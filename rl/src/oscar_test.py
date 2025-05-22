import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
import numpy as np
import random
import os
from collections import deque, namedtuple

# Assuming your til_environment is accessible
# from til_environment import gridworld # You'll need to import your actual environment

# --- Configuration ---
# Environment specific (match your game)
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS_PER_EPISODE = 100 # Max steps in one round of the game
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay

# --- Dreamer Configuration (Example Values - NEEDS EXTENSIVE TUNING) ---
class DreamerConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # RSSM (Recurrent State Space Model) Sizes
        self.rssm_deter_size = 256  # Size of the deterministic part of the RSSM state
        self.rssm_stoch_size = 32   # Size of the stochastic part of the RSSM state (latent variable z)
        self.rssm_hidden_size = 256 # Hidden size for GRU/transition model parts within RSSM

        # Encoder/Decoder
        self.embedding_size = 1024 # Size of the observation embedding produced by the Encoder
        self.cnn_depth = 32        # Base channel depth for CNN layers in Encoder/Decoder
        self.model_hidden_size = 400 # Hidden layer size for Decoders (Image, Reward, Continue)

        # Actor-Critic (operates in the learned latent space)
        self.actor_hidden_size = 400
        self.value_hidden_size = 400
        self.gamma = 0.995          # Discount factor for critic and GAE
        self.lambda_ = 0.95         # Lambda for Generalized Advantage Estimation (GAE)
        self.imagination_horizon = 15 # How many steps to "imagine" forward for actor-critic updates

        # Learning Rates
        self.world_model_lr = 2e-4  # Learning rate for Encoder, RSSM, Decoders
        self.actor_lr = 4e-5        # Learning rate for Actor
        self.value_lr = 4e-5        # Learning rate for Critic

        # Training Hyperparameters
        self.batch_size = 16          # Number of sequences sampled from replay buffer per training step
        self.sequence_length = 50     # Length of each sequence sampled from replay buffer
        self.replay_buffer_capacity = int(1e6) # Total number of transitions (adjust based on sequence_length)
                                               # Effective sequence capacity is capacity / sequence_length
        self.train_every_env_steps = 1000 # How many environment steps before triggering a training phase
        self.num_train_updates_per_phase = 100 # Number of gradient updates during one training phase
        
        self.free_nats = 3.0          # KL balancing: min KL divergence value for the loss
        self.kl_scale = 1.0           # Scale factor for the KL divergence loss term
        self.grad_clip_norm = 100.0   # Max norm for gradient clipping
        
        # Observation structure (derived from DreamerObservationWrapper)
        self.obs_image_shape = (10, 7, 5) # Channels, Height, Width (for processed viewcone)
        self.obs_vector_dim = 8           # Dimension of the processed vector features

        self.action_noise_std = 0.3 # For continuous actions; for discrete, actor's stochasticity provides exploration
        self.target_update_interval = 100 # Interval (in training updates) for updating target critic

# Initialize config globally or pass around as needed
config = DreamerConfig()
DEVICE = config.device

# --- Dreamer Observation Wrapper ---
# This class processes the raw observation from your environment into a format
# suitable for Dreamer's Encoder (separate image-like and vector features).
class DreamerObservationWrapper:
    def __init__(self, env_max_steps=MAX_STEPS_PER_EPISODE, env_grid_size_x=MAP_SIZE_X, env_grid_size_y=MAP_SIZE_Y):
        self._max_steps = env_max_steps
        self._grid_size_x = env_grid_size_x
        self._grid_size_y = env_grid_size_y
        self.image_channels = config.obs_image_shape[0] # Should be 10

    def _process_viewcone(self, viewcone_obs_np):
        # viewcone_obs_np is expected to be a 7x5 numpy array from the environment
        # Output: (Channels, Height, Width) = (10, 7, 5)
        processed_viewcone = np.zeros((7, 5, self.image_channels), dtype=np.float32) # H, W, C

        for r in range(viewcone_obs_np.shape[0]): # Should be 7
            for c in range(viewcone_obs_np.shape[1]): # Should be 5
                val = viewcone_obs_np[r, c]
                
                # Bits 0-1: Tile type (one-hot encoded into channels 0-3)
                tile_type = val & 0b11
                if 0 <= tile_type < 4: # Ensure tile_type is within expected range for one-hot
                    processed_viewcone[r, c, tile_type] = 1.0
                
                # Bit 2: Scout (channel 4)
                if (val >> 2) & 1: processed_viewcone[r, c, 4] = 1.0
                # Bit 3: Guard (channel 5)
                if (val >> 3) & 1: processed_viewcone[r, c, 5] = 1.0
                # Bit 4: Right wall (channel 6)
                if (val >> 4) & 1: processed_viewcone[r, c, 6] = 1.0
                # Bit 5: Bottom wall (channel 7)
                if (val >> 5) & 1: processed_viewcone[r, c, 7] = 1.0
                # Bit 6: Left wall (channel 8)
                if (val >> 6) & 1: processed_viewcone[r, c, 8] = 1.0
                # Bit 7: Top wall (channel 9)
                if (val >> 7) & 1: processed_viewcone[r, c, 9] = 1.0
        
        return np.transpose(processed_viewcone, (2, 0, 1)) # Transpose to C, H, W

    def process_observation(self, raw_obs_dict):
        # Ensure viewcone is a numpy array
        viewcone_np = np.array(raw_obs_dict['viewcone']) if isinstance(raw_obs_dict['viewcone'], list) else raw_obs_dict['viewcone']
        processed_viewcone_tensor = torch.from_numpy(self._process_viewcone(viewcone_np)).float()

        # Process direction (one-hot)
        direction_one_hot = np.zeros(4, dtype=np.float32)
        direction_val = raw_obs_dict.get('direction', 0)
        if 0 <= direction_val < 4:
            direction_one_hot[direction_val] = 1.0
        
        # Process scout role
        scout_role = np.array([raw_obs_dict.get('scout', 0.0)], dtype=np.float32)
        
        # Process location (normalize to [-1, 1])
        loc = raw_obs_dict.get('location', [0,0])
        loc_x = loc[0] / (self._grid_size_x -1) if self._grid_size_x > 1 else 0.0
        loc_y = loc[1] / (self._grid_size_y -1) if self._grid_size_y > 1 else 0.0
        normalized_location = (np.array([loc_x, loc_y], dtype=np.float32) * 2.0) - 1.0

        # Process step (normalize to [-1, 1])
        step_val = raw_obs_dict.get('step', 0)
        normalized_step = (np.array([step_val / self._max_steps], dtype=np.float32) * 2.0) - 1.0 if self._max_steps > 0 else np.array([0.0], dtype=np.float32)

        vector_features_np = np.concatenate([
            direction_one_hot, scout_role, normalized_location, normalized_step
        ])
        vector_features_tensor = torch.from_numpy(vector_features_np).float()
        
        # Ensure vector dimension matches config
        if vector_features_tensor.shape[0] != config.obs_vector_dim:
            raise ValueError(f"Processed vector dimension mismatch. Expected {config.obs_vector_dim}, got {vector_features_tensor.shape[0]}")

        return {
            'image': processed_viewcone_tensor, # Tensor (C,H,W)
            'vector': vector_features_tensor    # Tensor (VecDim,)
        }

# --- Sequence Replay Buffer (Uniform Sampling) ---
# Stores sequences of (observation_dict, action_one_hot, reward, done)
class SequenceReplayBuffer:
    def __init__(self, capacity_transitions, sequence_length, obs_image_shape, obs_vector_dim, action_dim, device):
        self.capacity_sequences = capacity_transitions // sequence_length
        self.sequence_length = sequence_length
        self.device = device
        self.action_dim = action_dim

        # Pre-allocate memory for efficiency
        self.observations_image = np.empty((self.capacity_sequences, sequence_length, *obs_image_shape), dtype=np.float32)
        self.observations_vector = np.empty((self.capacity_sequences, sequence_length, obs_vector_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity_sequences, sequence_length, action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity_sequences, sequence_length, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity_sequences, sequence_length, 1), dtype=np.bool_)

        self.buffer_idx = 0 # Points to the next sequence slot to fill
        self.is_full = False
        self._current_sequence_steps = [] # Temporarily stores steps for the current sequence being built

    def add_step(self, processed_obs_dict, action_scalar, reward, done):
        # Convert action_scalar to one-hot numpy array
        action_one_hot_np = np.zeros(self.action_dim, dtype=np.float32)
        action_one_hot_np[action_scalar] = 1.0

        self._current_sequence_steps.append({
            'image': processed_obs_dict['image'].cpu().numpy(), # Store as numpy to save GPU memory
            'vector': processed_obs_dict['vector'].cpu().numpy(),
            'action': action_one_hot_np,
            'reward': np.array([reward], dtype=np.float32),
            'done': np.array([done], dtype=np.bool_)
        })

        # If the current sequence reaches the desired length, store it
        if len(self._current_sequence_steps) == self.sequence_length:
            self._store_completed_sequence()
            self._current_sequence_steps = [] # Reset for the next sequence

    def _store_completed_sequence(self):
        if not self._current_sequence_steps: return # Should not happen if called correctly

        for t_step, experience in enumerate(self._current_sequence_steps):
            self.observations_image[self.buffer_idx, t_step] = experience['image']
            self.observations_vector[self.buffer_idx, t_step] = experience['vector']
            self.actions[self.buffer_idx, t_step] = experience['action']
            self.rewards[self.buffer_idx, t_step] = experience['reward']
            self.dones[self.buffer_idx, t_step] = experience['done']
        
        self.buffer_idx = (self.buffer_idx + 1) % self.capacity_sequences
        if self.buffer_idx == 0 and not self.is_full: # Wrapped around
            self.is_full = True
            print("Sequence Replay Buffer is now full.")

    def sample(self, batch_size):
        current_num_sequences = self.capacity_sequences if self.is_full else self.buffer_idx
        if current_num_sequences < batch_size:
            return None # Not enough sequences to form a batch

        # Sample random sequence indices
        seq_indices = np.random.randint(0, current_num_sequences, size=batch_size)
        
        # Retrieve batch data and convert to tensors on the correct device
        obs_img_batch = torch.tensor(self.observations_image[seq_indices], dtype=torch.float32).to(self.device)
        obs_vec_batch = torch.tensor(self.observations_vector[seq_indices], dtype=torch.float32).to(self.device)
        actions_batch = torch.tensor(self.actions[seq_indices], dtype=torch.float32).to(self.device)
        rewards_batch = torch.tensor(self.rewards[seq_indices], dtype=torch.float32).to(self.device)
        # Convert boolean 'dones' to float for model compatibility (e.g., in loss calculations)
        dones_batch = torch.tensor(self.dones[seq_indices], dtype=torch.float32).to(self.device)

        return {'image': obs_img_batch, 'vector': obs_vec_batch}, actions_batch, rewards_batch, dones_batch

    def __len__(self):
        # Returns the number of *complete sequences* stored
        return self.capacity_sequences if self.is_full else self.buffer_idx

# --- Dreamer Model Components ---

# Encoder: Processes observations into embeddings
class Encoder(nn.Module):
    def __init__(self, img_shape, vec_dim, cnn_depth, embedding_size):
        super().__init__()
        channels, height, width = img_shape
        # CNN for image features (viewcone)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, cnn_depth, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cnn_depth, 2 * cnn_depth, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * cnn_depth, 4 * cnn_depth, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            # Add another conv if H,W are large, or if smaller, maybe fewer layers.
            # For 7x5, this might be enough. Consider a final conv to reduce H,W if needed.
            nn.Flatten() # Flatten the output of CNN
        )
        # Calculate CNN output dimension dynamically
        with torch.no_grad():
            dummy_img_input = torch.zeros(1, *img_shape) # Batch_size=1, C, H, W
            cnn_out_dim = self.cnn(dummy_img_input).shape[1]

        # MLP for vector features
        self.vector_mlp = nn.Sequential(
            nn.Linear(vec_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64,64), nn.ReLU(inplace=True) # Added another layer
        )
        
        # Combiner MLP to produce the final observation embedding
        self.combiner = nn.Linear(cnn_out_dim + 64, embedding_size)

    def forward(self, obs_dict_batch):
        # obs_dict_batch: {'image': tensor, 'vector': tensor}
        # Tensors can be (B, S, C, H, W) and (B, S, VecDim) for sequences,
        # or (B, C, H, W) and (B, VecDim) for single steps.
        img_obs, vec_obs = obs_dict_batch['image'], obs_dict_batch['vector']
        
        has_sequence_dim = img_obs.ndim == 5 # Check if sequence dimension (S) is present
        B = img_obs.shape[0]
        S = img_obs.shape[1] if has_sequence_dim else 1

        # Reshape if sequence: (B, S, ...) -> (B*S, ...)
        if has_sequence_dim:
            img_obs = img_obs.reshape(B * S, *img_obs.shape[2:])
            vec_obs = vec_obs.reshape(B * S, *vec_obs.shape[2:])
        
        img_feat = self.cnn(img_obs)         # (B*S, CnnOutDim)
        vec_feat = self.vector_mlp(vec_obs)  # (B*S, 64)
        
        combined_features = torch.cat([img_feat, vec_feat], dim=-1)
        embedding = self.combiner(combined_features) # (B*S, EmbeddingSize)

        # Reshape back if it was a sequence: (B*S, EmbeddingSize) -> (B, S, EmbeddingSize)
        if has_sequence_dim:
            embedding = embedding.reshape(B, S, -1)
            
        return embedding

# RSSM: Core recurrent model for learning environment dynamics
class RSSM(nn.Module):
    def __init__(self, action_dim, obs_embed_dim, config_obj): # Pass full config object
        super().__init__()
        self.config = config_obj # Store config for easy access to its parameters
        self.action_dim = action_dim
        self.obs_embed_dim = obs_embed_dim

        # RNN cell (e.g., GRU) for the deterministic path of the RSSM
        self.rnn_cell = nn.GRUCell(self.config.rssm_hidden_size, self.config.rssm_deter_size)
        
        # Linear layer to process input for RNN: concat(prev_stochastic_state, prev_action)
        self.fc_rnn_input = nn.Linear(self.config.rssm_stoch_size + action_dim, self.config.rssm_hidden_size)
        
        # Posterior: q(z_t | h_t, e_t) - Infers stochastic state from deterministic state and current observation embedding
        self.fc_posterior_input = nn.Linear(self.config.rssm_deter_size + obs_embed_dim, self.config.rssm_hidden_size)
        self.fc_posterior_mean_std = nn.Linear(self.config.rssm_hidden_size, 2 * self.config.rssm_stoch_size) # Outputs mean and log_std

        # Prior: p(z_t | h_t) - Predicts next stochastic state from current deterministic state (used for imagination)
        self.fc_prior_input = nn.Linear(self.config.rssm_deter_size, self.config.rssm_hidden_size)
        self.fc_prior_mean_std = nn.Linear(self.config.rssm_hidden_size, 2 * self.config.rssm_stoch_size) # Outputs mean and log_std

    def initial_state(self, batch_size, device):
        # Returns initial hidden state (h) and stochastic state (z) for the RSSM
        return (torch.zeros(batch_size, self.config.rssm_deter_size, device=device),
                torch.zeros(batch_size, self.config.rssm_stoch_size, device=device))

    def observe_step(self, obs_embed_t, prev_action_one_hot, prev_rssm_state, is_first_step_mask):
        # obs_embed_t: Embedding of current observation e_t (B, EmbedDim)
        # prev_action_one_hot: Previous action a_{t-1} (B, ActionDim)
        # prev_rssm_state: Tuple (h_{t-1}, z_{t-1})
        # is_first_step_mask: Boolean tensor (B,) indicating if this is the first step in a sequence
        
        h_prev, z_prev = prev_rssm_state
        
        # If it's the first step of any sequence in the batch, reset its state
        if is_first_step_mask.any():
            h_prev = h_prev.clone(); z_prev = z_prev.clone() # Avoid in-place modification issues
            h_prev[is_first_step_mask] = 0.0
            z_prev[is_first_step_mask] = 0.0
            # Also zero out the previous action for the first step
            prev_action_one_hot = prev_action_one_hot.clone()
            prev_action_one_hot[is_first_step_mask] = 0.0

        # Compute input for RNN cell: processed concat(z_{t-1}, a_{t-1})
        rnn_input = F.relu(self.fc_rnn_input(torch.cat([z_prev, prev_action_one_hot], dim=-1)))
        # Update deterministic state: h_t = GRU(h_{t-1}, rnn_input)
        h_t = self.rnn_cell(rnn_input, h_prev)

        # Posterior: q(z_t | h_t, e_t)
        posterior_features = F.relu(self.fc_posterior_input(torch.cat([h_t, obs_embed_t], dim=-1)))
        post_mean, post_log_std = self.fc_posterior_mean_std(posterior_features).chunk(2, dim=-1)
        post_std = F.softplus(post_log_std) + 0.1 # Ensure std_dev is positive and non-zero
        posterior_dist = D.Normal(post_mean, post_std)
        z_t_posterior = posterior_dist.rsample() # Sample z_t using reparameterization trick

        # Prior: p(z_t | h_t) - used for KL divergence loss and for imagination
        prior_features = F.relu(self.fc_prior_input(h_t))
        prior_mean, prior_log_std = self.fc_prior_mean_std(prior_features).chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1
        prior_dist = D.Normal(prior_mean, prior_std)
        
        return (h_t, z_t_posterior), (posterior_dist, prior_dist) # Current RSSM state and distributions

    def imagine_step(self, prev_action_one_hot, prev_rssm_state):
        # Used for actor-critic training; predicts next state based on action only (no observation)
        h_prev, z_prev = prev_rssm_state
        rnn_input = F.relu(self.fc_rnn_input(torch.cat([z_prev, prev_action_one_hot], dim=-1)))
        h_t = self.rnn_cell(rnn_input, h_prev)
        
        # Predict z_t using the prior p(z_t | h_t)
        prior_features = F.relu(self.fc_prior_input(h_t))
        prior_mean, prior_log_std = self.fc_prior_mean_std(prior_features).chunk(2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 0.1
        prior_dist = D.Normal(prior_mean, prior_std)
        z_t_prior = prior_dist.sample() # Sample from the prior distribution for imagination
        
        return (h_t, z_t_prior)

    def observe_sequence(self, obs_embed_sequence, action_sequence, initial_rssm_h, initial_rssm_z):
        # obs_embed_sequence: (B, S, EmbedDim), action_sequence: (B, S, ActionDim)
        # initial_rssm_h, initial_rssm_z: Initial deterministic and stochastic states (B, Dim)
        
        batch_size, seq_len, _ = obs_embed_sequence.shape
        h_t, z_t = initial_rssm_h, initial_rssm_z # Current RSSM states, iterates through sequence
        
        list_h_states, list_z_posterior_states, list_posterior_dists, list_prior_dists = [], [], [], []

        for t in range(seq_len):
            # Action at step t-1 leads to observation embedding at step t.
            # For the first step (t=0), the "previous action" can be a zero vector.
            prev_action = action_sequence[:, t-1] if t > 0 else torch.zeros_like(action_sequence[:,0])
            
            # Mask indicating if this is the first step (t=0) for each item in the batch
            is_first_mask = torch.tensor([t == 0] * batch_size, device=obs_embed_sequence.device)
            
            (h_t, z_t), (posterior_dist_t, prior_dist_t) = self.observe_step(
                obs_embed_sequence[:, t], prev_action, (h_t, z_t), is_first_mask
            )
            
            list_h_states.append(h_t)
            list_z_posterior_states.append(z_t)
            list_posterior_dists.append(posterior_dist_t)
            list_prior_dists.append(prior_dist_t)
            
        # Stack collected states and distributions along the sequence dimension
        h_states_seq = torch.stack(list_h_states, dim=1)             # (B, S, DeterDim)
        z_posterior_states_seq = torch.stack(list_z_posterior_states, dim=1) # (B, S, StochDim)
        
        # Reconstruct distributions from stacked means and stddevs
        posterior_dists_seq = D.Normal(
            torch.stack([d.mean for d in list_posterior_dists], dim=1),
            torch.stack([d.stddev for d in list_posterior_dists], dim=1)
        )
        prior_dists_seq = D.Normal(
            torch.stack([d.mean for d in list_prior_dists], dim=1),
            torch.stack([d.stddev for d in list_prior_dists], dim=1)
        )
                
        return (h_states_seq, z_posterior_states_seq), (posterior_dists_seq, prior_dists_seq)

# Decoder: Reconstructs observations, rewards, etc., from RSSM states
class Decoder(nn.Module):
    def __init__(self, rssm_state_dim, hidden_dim, output_shape_or_dim, is_image_decoder=False):
        super().__init__()
        self.rssm_state_dim = rssm_state_dim # deter_size + stoch_size
        self.is_image_decoder = is_image_decoder

        # Common MLP layers
        self.fc_layers = nn.Sequential(
            nn.Linear(rssm_state_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)
        )
        
        if self.is_image_decoder:
            # Image decoder needs deconvolutional layers (ConvTranspose2d)
            # This is an example, architecture depends heavily on cnn_depth and target image_shape
            # It should roughly mirror the Encoder's CNN part.
            self.img_output_shape = output_shape_or_dim # (Channels, H, W)
            # Project to a base size for deconv, e.g., 4*cnn_depth * (H/4) * (W/4)
            # This part requires careful design to match Encoder's output shape before flatten.
            # For simplicity, let's assume a linear layer to a flattened feature map, then reshape and deconv.
            # Example: if Encoder's CNN output before flatten was (4*depth, H_cnn, W_cnn)
            
            # Simplified: Calculate a reasonable flat size for the deconv base
            # This is a heuristic and might need adjustment based on CNN architecture
            # Aim for a small spatial dimension before deconv layers
            self.deconv_base_h = max(1, self.img_output_shape[1] // 4) 
            self.deconv_base_w = max(1, self.img_output_shape[2] // 4)
            self.deconv_base_channels = 4 * config.cnn_depth # Match deep_conv_channels in encoder
            self.deconv_base_size_flat = self.deconv_base_channels * self.deconv_base_h * self.deconv_base_w
            
            self.fc_to_deconv_base = nn.Linear(hidden_dim, self.deconv_base_size_flat)
            
            # Example Deconvolutional Layers - this needs to be designed carefully
            # to upsample from (deconv_base_channels, deconv_base_h, deconv_base_w) to img_output_shape
            # For now, using a linear layer as a placeholder if deconv is too complex to generalize here.
            # A real implementation would use nn.ConvTranspose2d layers.
            self.deconv_layers = nn.Sequential(
                 nn.Linear(self.deconv_base_size_flat, np.prod(self.img_output_shape)) # Fallback
                # Example of a ConvTranspose2d layer (needs more to reach target size):
                # nn.ConvTranspose2d(self.deconv_base_channels, config.cnn_depth * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(config.cnn_depth * 2, config.cnn_depth, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(config.cnn_depth, self.img_output_shape[0], kernel_size=3, stride=1, padding=1)
            )
        else:
            # For scalar (reward) or vector (continue probability)
            self.output_layer = nn.Linear(hidden_dim, output_shape_or_dim) # output_shape_or_dim is an int (e.g., 1)

    def forward(self, h_state, z_state):
        # h_state, z_state can be (B,S,Dim) or (B*S,Dim) or (B,Dim)
        # Concatenate deterministic and stochastic parts of RSSM state
        rssm_features = torch.cat([h_state, z_state], dim=-1)
        hidden_output = self.fc_layers(rssm_features)
        
        if self.is_image_decoder:
            deconv_input_flat = self.fc_to_deconv_base(hidden_output)
            # If using actual ConvTranspose2d, reshape deconv_input_flat to (B*S, C_base, H_base, W_base)
            # deconv_input_spatial = deconv_input_flat.view(-1, self.deconv_base_channels, self.deconv_base_h, self.deconv_base_w)
            # flat_pixel_output = self.deconv_layers(deconv_input_spatial).flatten(start_dim=1) # then flatten again if deconv_layers output spatial

            # Using the simplified linear output for now
            flat_pixel_output = self.deconv_layers(deconv_input_flat) # if deconv_layers is just Linear
            
            original_batch_shape = h_state.shape[:-1] # (B,S) or (B) or (B*S) if already flat
            if not original_batch_shape: # if input was (StateDim)
                original_batch_shape = (1,) if h_state.ndim == 1 else h_state.shape[0:1]


            # Ensure flat_pixel_output has a batch dimension before view
            if flat_pixel_output.ndim == 1 and np.prod(original_batch_shape) == 1 : # single item, single batch
                 flat_pixel_output = flat_pixel_output.unsqueeze(0)
            elif flat_pixel_output.ndim == len(original_batch_shape) + 1: # (B*S, FlatPixels)
                pass # Already has batch dim relative to original_batch_shape for pixels
            
            try:
                output = flat_pixel_output.view(*original_batch_shape, *self.img_output_shape)
            except RuntimeError as e:
                print(f"Error during Decoder view operation: {e}")
                print(f"original_batch_shape: {original_batch_shape}, img_output_shape: {self.img_output_shape}")
                print(f"flat_pixel_output shape: {flat_pixel_output.shape}")
                raise e

            return torch.sigmoid(output) # Pixels usually in [0,1]
        else:
            # For reward or continue flag (scalar output)
            return self.output_layer(hidden_output)

# Actor: Learns the policy in the latent space of the RSSM
class Actor(nn.Module):
    def __init__(self, rssm_state_dim, action_dim, hidden_size):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(rssm_state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim) # Outputs logits for discrete actions
        )
    def forward(self, h_state, z_state):
        rssm_features = torch.cat([h_state, z_state], dim=-1)
        action_logits = self.fc_layers(rssm_features)
        # Return a categorical distribution for discrete actions
        return D.Categorical(logits=action_logits)

# Critic: Learns a value function in the latent space of the RSSM
class Critic(nn.Module):
    def __init__(self, rssm_state_dim, hidden_size):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(rssm_state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1) # Outputs a single value
        )
    def forward(self, h_state, z_state):
        rssm_features = torch.cat([h_state, z_state], dim=-1)
        return self.fc_layers(rssm_features)


# --- Dreamer Training Agent ---
class DreamerTrainingAgent:
    def __init__(self, agent_config, obs_processor_instance, model_load_path=None, model_save_path="dreamer_agent.pth"):
        self.config = agent_config
        self.device = self.config.device
        self.obs_processor = obs_processor_instance
        self.model_save_path = model_save_path
        
        self.action_dim = OUTPUT_ACTIONS # From global scope
        self.rssm_state_dim = self.config.rssm_deter_size + self.config.rssm_stoch_size

        # Initialize Model Components
        self.encoder = Encoder(self.config.obs_image_shape, self.config.obs_vector_dim, 
                               self.config.cnn_depth, self.config.embedding_size).to(self.device)
        self.rssm = RSSM(self.action_dim, self.config.embedding_size, self.config).to(self.device)
        
        self.image_decoder = Decoder(self.rssm_state_dim, self.config.model_hidden_size, 
                                     self.config.obs_image_shape, is_image_decoder=True).to(self.device)
        self.reward_decoder = Decoder(self.rssm_state_dim, self.config.model_hidden_size, 1).to(self.device)
        self.continue_decoder = Decoder(self.rssm_state_dim, self.config.model_hidden_size, 1).to(self.device) # Predicts (1-done)

        self.actor = Actor(self.rssm_state_dim, self.action_dim, self.config.actor_hidden_size).to(self.device)
        self.critic = Critic(self.rssm_state_dim, self.config.value_hidden_size).to(self.device)
        
        # Target critic for stabilizing critic updates (common in actor-critic methods)
        self.target_critic = Critic(self.rssm_state_dim, self.config.value_hidden_size).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False) # Target network is not trained directly

        # Optimizers
        self.world_model_params = (list(self.encoder.parameters()) + list(self.rssm.parameters()) +
                                   list(self.image_decoder.parameters()) + list(self.reward_decoder.parameters()) +
                                   list(self.continue_decoder.parameters()))
        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.world_model_optimizer = optim.Adam(self.world_model_params, lr=self.config.world_model_lr, eps=1e-5)
        self.actor_optimizer = optim.Adam(self.actor_params, lr=self.config.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_params, lr=self.config.value_lr, eps=1e-5)

        # Replay Buffer
        self.memory = SequenceReplayBuffer(
            self.config.replay_buffer_capacity, self.config.sequence_length,
            self.config.obs_image_shape, self.config.obs_vector_dim, self.action_dim, self.device
        )
        
        # State for interaction loop (for a single agent/environment instance)
        self._current_rssm_state_h = None 
        self._current_rssm_state_z = None
        self._prev_action_one_hot_for_env_step = None # For selecting next action

        self._total_train_updates = 0 # Counter for training updates (gradient steps)

        if model_load_path and os.path.exists(model_load_path):
            self.load_model_checkpoint(model_load_path)
        else:
            print("Initializing new Dreamer agent with random weights.")
        
        print(f"Dreamer agent initialized. Using device: {self.device}")

    def reset_agent_state_for_episode(self):
        # Called at the beginning of each new environment episode
        initial_h, initial_z = self.rssm.initial_state(batch_size=1, device=self.device)
        self._current_rssm_state_h = initial_h
        self._current_rssm_state_z = initial_z
        self._prev_action_one_hot_for_env_step = torch.zeros(1, self.action_dim, device=self.device)

    def select_action_for_env(self, processed_obs_dict, is_training_mode=True):
        # processed_obs_dict contains {'image': tensor, 'vector': tensor} for current step
        with torch.no_grad():
            if self._current_rssm_state_h is None: # Should have been called by reset_agent_state_for_episode
                self.reset_agent_state_for_episode()

            # Add batch dimension to observation tensors (as models expect batched input)
            obs_image_tensor = processed_obs_dict['image'].unsqueeze(0).to(self.device)
            obs_vector_tensor = processed_obs_dict['vector'].unsqueeze(0).to(self.device)
            
            # Encode current observation
            current_obs_embed = self.encoder({'image': obs_image_tensor, 'vector': obs_vector_tensor}) # (1, EmbedDim)
            
            # Update RSSM state using current observation and previous action
            # is_first_step_mask is False here as we are stepping within an episode.
            # reset_agent_state_for_episode handles the true first step.
            (h_t, z_t), _ = self.rssm.observe_step(
                current_obs_embed, 
                self._prev_action_one_hot_for_env_step, # Action that led to current_obs_embed
                (self._current_rssm_state_h, self._current_rssm_state_z), 
                is_first_step_mask=torch.tensor([False], device=self.device) 
            )
            
            # Update agent's recurrent state
            self._current_rssm_state_h, self._current_rssm_state_z = h_t, z_t

            # Get action from Actor based on current RSSM state
            action_distribution = self.actor(h_t, z_t)
            if is_training_mode:
                action_tensor = action_distribution.sample() # Sample action for exploration
            else: 
                action_tensor = torch.argmax(action_distribution.probs, dim=-1, keepdim=True) # Greedy for evaluation
            
            action_scalar = action_tensor.squeeze().cpu().item() # Convert to scalar for env.step()
            
            # Store this action (one-hot) to be used as "previous action" in the next call
            self._prev_action_one_hot_for_env_step = F.one_hot(action_tensor, num_classes=self.action_dim).float() # (1, ActionDim)
            
            return action_scalar

    def store_env_step_in_buffer(self, processed_obs_dict, action_scalar, reward, done):
        # This method is called *after* env.step() with action_scalar
        # processed_obs_dict is the observation *before* taking action_scalar
        # reward and done are the results *of* taking action_scalar
        self.memory.add_step(processed_obs_dict, action_scalar, reward, done)

    def train_agent_model_phase(self):
        # Check if enough sequences are in the buffer to form a batch
        if len(self.memory) < self.config.batch_size : 
             print(f"Skipping training: Not enough sequences in buffer ({len(self.memory)}/{self.config.batch_size}).")
             return False 

        print(f"\n--- Training Dreamer Phase (Total Updates So Far: {self._total_train_updates}) ---")
        avg_actor_losses, avg_value_losses, avg_model_losses = [], [], []

        # Perform multiple gradient updates in this training phase
        for i_update in range(self.config.num_train_updates_per_phase):
            self._total_train_updates += 1
            
            # Sample a batch of sequences from the replay buffer
            batch_data = self.memory.sample(self.config.batch_size)
            if batch_data is None: continue # Should not happen if length check is correct

            obs_batch_dict, actions_one_hot_batch, rewards_batch, dones_batch = batch_data
            # obs_batch_dict contains {'image': (B,S,C,H,W), 'vector': (B,S,VecDim)}
            # actions_one_hot_batch: (B,S,ActionDim), rewards_batch: (B,S,1), dones_batch: (B,S,1)

            # --- 1. World Model Training ---
            self.world_model_optimizer.zero_grad()
            
            # Encode all observations in the sampled sequences
            embedded_obs_sequence = self.encoder(obs_batch_dict) # (B, S, EmbedDim)
            
            # Initialize RSSM hidden states for sequence processing
            initial_h_rssm, initial_z_rssm = self.rssm.initial_state(self.config.batch_size, self.device)
            
            # Unroll RSSM over the sequences to get latent states and distributions
            (h_states_sequence, z_posterior_states_sequence), (posterior_dists_sequence, prior_dists_sequence) = \
                self.rssm.observe_sequence(embedded_obs_sequence, actions_one_hot_batch, initial_h_rssm, initial_z_rssm)
            # h_states_sequence, z_posterior_states_sequence are (B, S, Dim)
            # posterior_dists_sequence, prior_dists_sequence are D.Normal with batch_shape (B,S) event_shape (StochDim)

            # Calculate World Model Losses
            # Flatten states for decoders: (B, S, Dim) -> (B*S, Dim)
            B, S, _ = h_states_sequence.shape # B = batch_size, S = sequence_length
            flat_h_states = h_states_sequence.reshape(B*S, -1)
            flat_z_posterior_states = z_posterior_states_sequence.reshape(B*S, -1)
            
            # a) Image Reconstruction Loss (using Image Decoder)
            predicted_images_flat = self.image_decoder(flat_h_states, flat_z_posterior_states) # (B*S, C,H,W)
            true_images_flat = obs_batch_dict['image'].reshape(B*S, *self.config.obs_image_shape)
            # MSE loss for image reconstruction, sum over pixels, mean over batch*sequence
            recon_loss_image = F.mse_loss(predicted_images_flat, true_images_flat, reduction='none').sum(dim=[1,2,3]).mean()
            
            # b) Reward Prediction Loss (using Reward Decoder)
            predicted_rewards_flat = self.reward_decoder(flat_h_states, flat_z_posterior_states) # (B*S, 1)
            true_rewards_flat = rewards_batch.reshape(B*S, 1)
            reward_pred_loss = F.mse_loss(predicted_rewards_flat, true_rewards_flat) # Default reduction is mean

            # c) Continue Prediction Loss (using Continue Decoder, predicts 1-done)
            predicted_continue_logits_flat = self.continue_decoder(flat_h_states, flat_z_posterior_states) # (B*S, 1)
            true_continue_targets_flat = (1.0 - dones_batch.reshape(B*S, 1)) # Target is 1 if not done, 0 if done
            # Binary Cross Entropy with Logits for continue prediction
            continue_pred_loss = F.binary_cross_entropy_with_logits(predicted_continue_logits_flat, true_continue_targets_flat) # Default reduction is mean

            # d) KL Divergence Loss (between prior and posterior distributions of z)
            kl_div_elementwise = D.kl.kl_divergence(posterior_dists_sequence, prior_dists_sequence) # (B, S, StochDim)
            kl_loss_unbalanced = kl_div_elementwise.sum(dim=-1) # Sum over stoch_dim -> (B,S)
            # Apply free_nats: max(free_nats, kl_loss_value) before summing over batch and sequence, then mean
            kl_loss_balanced = torch.max(torch.full_like(kl_loss_unbalanced, self.config.free_nats), kl_loss_unbalanced).mean() 

            # Total World Model Loss
            world_model_loss = recon_loss_image + reward_pred_loss + continue_pred_loss + self.config.kl_scale * kl_loss_balanced
            
            world_model_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model_params, self.config.grad_clip_norm)
            self.world_model_optimizer.step()
            avg_model_losses.append(world_model_loss.item())

            # --- 2. Actor-Critic Training in Imagination ---
            # Detach initial states for imagination from world model computation graph
            # These are the posterior states from the observed sequences, reshaped for imagination start
            imag_initial_h_flat = h_states_sequence.detach().reshape(B*S, -1) 
            imag_initial_z_flat = z_posterior_states_sequence.detach().reshape(B*S, -1)

            current_imag_h = imag_initial_h_flat
            current_imag_z = imag_initial_z_flat

            imagined_h_traj, imagined_z_traj, imagined_actions_log_probs_traj, imagined_rewards_pred_traj = [], [], [], []
            
            # Imagine forward for H steps
            for _ in range(self.config.imagination_horizon):
                imagined_h_traj.append(current_imag_h)
                imagined_z_traj.append(current_imag_z)

                # Get action from Actor based on current imagined state
                action_dist_imag = self.actor(current_imag_h, current_imag_z)
                action_sample_imag = action_dist_imag.sample() # (B*S,)
                action_one_hot_imag = F.one_hot(action_sample_imag, num_classes=self.action_dim).float() # (B*S, ActionDim)
                
                imagined_actions_log_probs_traj.append(action_dist_imag.log_prob(action_sample_imag)) # (B*S,)

                # Predict reward for this imagined state-action using Reward Decoder
                reward_pred_imag = self.reward_decoder(current_imag_h, current_imag_z).squeeze(-1) # (B*S,)
                imagined_rewards_pred_traj.append(reward_pred_imag)

                # Step imagination forward using RSSM's imagine_step
                current_imag_h, current_imag_z = self.rssm.imagine_step(action_one_hot_imag, (current_imag_h, current_imag_z))
            
            # Stack imagined trajectories: (B*S, Horizon, Dim) or (B*S, Horizon)
            imag_h_stacked = torch.stack(imagined_h_traj, dim=1)
            imag_z_stacked = torch.stack(imagined_z_traj, dim=1)
            imag_actions_log_probs_stacked = torch.stack(imagined_actions_log_probs_traj, dim=1)
            imag_rewards_stacked = torch.stack(imagined_rewards_pred_traj, dim=1)

            # Calculate Lambda Returns for Critic targets
            with torch.no_grad():
                # Value of the state *after* the last imagined action (s_H), from Target Critic
                last_imagined_value = self.target_critic(current_imag_h, current_imag_z).squeeze(-1) # (B*S,)
                
                lambda_returns = torch.zeros_like(imag_rewards_stacked) # (B*S, Horizon)
                
                next_val_for_lambda = last_imagined_value
                for t in reversed(range(self.config.imagination_horizon)):
                    # Predicted continue probability for current imagined step (s_t)
                    continue_prob_imag = torch.sigmoid(self.continue_decoder(imag_h_stacked[:,t], imag_z_stacked[:,t])).squeeze(-1) # (B*S,)
                    
                    # Lambda return: r_t + gamma * cont_t * ( (1-lambda)*V_target(s_{t+1}) + lambda*LambdaRet(s_{t+1}) )
                    # V_target(s_{t+1}) is next_val_for_lambda if using critic, or next_return_component if using lambda
                    # Dreamer typically uses: discount * ( (1-lambda)*V(next_state) + lambda*next_lambda_return )
                    # Here, V(next_state) is `self.target_critic(imag_h_stacked[:, t+1], ...)` if t < H-1, else `last_imagined_value`
                    # And `next_lambda_return` is `next_val_for_lambda` which carries the recursive lambda sum.
                    
                    # Value of the *next* state in the imagined trajectory (s_{t+1})
                    # If t is the last step of horizon (H-1), next_state_value is V(s_H) = last_imagined_value
                    # Otherwise, it's V(s_{t+1}) from target_critic
                    if t < self.config.imagination_horizon - 1:
                         # This was an error in previous logic, should be the state that results from action at t
                         # The current_imag_h, current_imag_z *after* the loop are s_H.
                         # So, imag_h_stacked[:, t+1] is s_{t+1}
                        next_state_target_value = self.target_critic(imag_h_stacked[:, t+1], imag_z_stacked[:, t+1]).squeeze(-1)
                    else: # t == H-1
                        next_state_target_value = last_imagined_value


                    lambda_returns[:, t] = (imag_rewards_stacked[:, t] + 
                                           self.config.gamma * continue_prob_imag * ( (1 - self.config.lambda_) * next_state_target_value + 
                                             self.config.lambda_ * next_val_for_lambda )
                                          )
                    next_val_for_lambda = lambda_returns[:, t] # Update for next iteration (t-1)
            
            # Actor Loss: Maximize expected lambda returns
            # Advantages are (lambda_returns - V_critic(s_t)).detach() for stability
            advantages = (lambda_returns - self.critic(imag_h_stacked, imag_z_stacked).squeeze(-1)).detach()
            actor_loss = - (imag_actions_log_probs_stacked * advantages).mean()

            # Critic Loss: MSE between critic's value predictions and lambda returns
            critic_values_pred = self.critic(imag_h_stacked, imag_z_stacked).squeeze(-1) # (B*S, Horizon)
            value_loss = F.mse_loss(critic_values_pred, lambda_returns)

            # Update Actor and Critic
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True) # Retain graph because value_loss uses shared imagined states
            nn.utils.clip_grad_norm_(self.actor_params, self.config.grad_clip_norm)
            self.actor_optimizer.step()
            avg_actor_losses.append(actor_loss.item())

            self.critic_optimizer.zero_grad()
            value_loss.backward() 
            nn.utils.clip_grad_norm_(self.critic_params, self.config.grad_clip_norm)
            self.critic_optimizer.step()
            avg_value_losses.append(value_loss.item())

            # Soft update Target Critic
            if self._total_train_updates % self.config.target_update_interval == 0:
                with torch.no_grad():
                    for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
                        tau = 0.01 
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        if avg_model_losses: # Check if any updates were made
             print(f"Update Iter {i_update+1}/{self.config.num_train_updates_per_phase} of Phase: WM Loss: {np.mean(avg_model_losses):.4f}, Actor Loss: {np.mean(avg_actor_losses):.4f}, Critic Loss: {np.mean(avg_value_losses):.4f}")
        return True # Indicates training occurred

    def save_model_checkpoint(self):
        checkpoint_payload = {
            'encoder_state_dict': self.encoder.state_dict(),
            'rssm_state_dict': self.rssm.state_dict(),
            'image_decoder_state_dict': self.image_decoder.state_dict(),
            'reward_decoder_state_dict': self.reward_decoder.state_dict(),
            'continue_decoder_state_dict': self.continue_decoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'world_model_optimizer_state_dict': self.world_model_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            '_total_train_updates': self._total_train_updates,
            # Optionally save replay buffer state if needed, but can be large
        }
        torch.save(checkpoint_payload, self.model_save_path)
        print(f"Dreamer model checkpoint saved to {self.model_save_path}")

    def load_model_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.rssm.load_state_dict(checkpoint['rssm_state_dict'])
            self.image_decoder.load_state_dict(checkpoint['image_decoder_state_dict'])
            self.reward_decoder.load_state_dict(checkpoint['reward_decoder_state_dict'])
            self.continue_decoder.load_state_dict(checkpoint['continue_decoder_state_dict'])
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            
            self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self._total_train_updates = checkpoint.get('_total_train_updates', 0)
            print(f"Dreamer model checkpoint loaded from {path}. Resuming from {self._total_train_updates} training updates.")
        except Exception as e:
            print(f"Error loading Dreamer model checkpoint from {path}: {e}. Model might be incompatible or path incorrect.")


# --- Main Training Loop (Adapted for Dreamer Agent) ---
def main_dreamer_training_loop(env_module_name, num_episodes=2000, novice_setting=False, 
                               load_checkpoint_path=None, save_checkpoint_path="dreamer_agent_checkpoint.pth"):
    
    # Dynamically import the environment module
    try:
        # This assumes 'til_environment.gridworld' is the structure
        # For example, env_module_name could be "til_environment.gridworld"
        parts = env_module_name.split('.')
        module_path = '.'.join(parts[:-1])
        env_creator_name = parts[-1]
        
        # Ensure the module_path is not empty if env_module_name is just "gridworld"
        if not module_path and len(parts) > 1: # e.g. "gridworld.env"
             module_path = parts[0] # then module_path is "gridworld"
             env_creator_name = parts[1] # and env_creator_name is "env"
        elif not module_path and len(parts) == 1: # e.g. "gridworld" and we assume gridworld.env()
            module_path = env_module_name
            env_creator_name = "env" # Assuming the function is called 'env' inside the module
            print(f"Assuming environment creator is {module_path}.{env_creator_name}()")


        env_package = __import__(module_path, fromlist=[env_creator_name if len(parts)>1 else ''])
        
        if len(parts) > 1 and hasattr(env_package, env_creator_name):
             env_module_obj = getattr(env_package, env_creator_name) # This could be 'gridworld' from 'til_environment'
        else: # If env_module_name was just "gridworld", env_package is the gridworld module itself
             env_module_obj = env_package

        if hasattr(env_module_obj, 'env'):
            env_creator_func = getattr(env_module_obj, 'env')
        else:
            raise AttributeError(f"Could not find 'env' function in module object {env_module_obj}")

    except ImportError as e:
        print(f"Error importing environment module '{env_module_name}' (path: '{module_path}'): {e}")
        print("Please ensure the environment is correctly installed and accessible.")
        return
    except AttributeError as e:
        print(f"Error accessing '.env()' method. Path: '{env_module_name}', Module Obj: {env_module_obj if 'env_module_obj' in locals() else 'N/A'}: {e}")
        return

    # Initialize PettingZoo environment
    # render_mode=None for training, "human" for visualization (if supported)
    env = env_creator_func(env_wrappers=[], render_mode=None, novice=novice_setting)
    
    # Identify the agent we are training (assuming it's the first one listed)
    if not env.possible_agents:
        print("Error: No possible agents found in the environment.")
        env.close()
        return
    my_trainable_agent_id = env.possible_agents[0] 
    print(f"Starting Dreamer training for agent: {my_trainable_agent_id}")
    print(f"Action space for {my_trainable_agent_id}: {env.action_space(my_trainable_agent_id)}")

    # Initialize Observation Processor and Dreamer Agent
    obs_processor = DreamerObservationWrapper()
    dreamer_agent = DreamerTrainingAgent(config, obs_processor, 
                                         model_load_path=load_checkpoint_path, 
                                         model_save_path=save_checkpoint_path)
    
    episode_scores_deque = deque(maxlen=100) # For tracking average score over recent episodes
    total_environment_steps_collected = 0

    for i_episode in range(1, num_episodes + 1):
        env.reset(seed=42 + i_episode) # Seed for reproducibility
        dreamer_agent.reset_agent_state_for_episode() # Reset Dreamer's internal recurrent state
        
        current_episode_score_for_my_agent = 0
        processed_observation_for_my_agent = None # Stores s_t for (s_t, a_t, r_{t+1}, d_{t+1}) tuple

        for agent_step_id in env.agent_iter(): # Iterates through agents until episode ends
            raw_observation_dict, reward, termination, truncation, info = env.last()
            is_done_for_current_agent_turn = termination or truncation

            if agent_step_id == my_trainable_agent_id:
                # This is our agent's turn.
                # `raw_observation_dict` is s_{t+1} (if previous action was by this agent) or s_0.
                # `reward` and `is_done_for_current_agent_turn` are r_{t+1} and d_{t+1} resulting from a_t.

                # If `processed_observation_for_my_agent` (s_t) exists, it means an action a_t was taken.
                # We can now form the tuple (s_t, a_t, r_{t+1}, d_{t+1}) and store it.
                if processed_observation_for_my_agent is not None and hasattr(dreamer_agent, '_last_action_scalar_for_buffer'):
                    dreamer_agent.store_env_step_in_buffer(
                        processed_observation_for_my_agent, # This was s_t
                        dreamer_agent._last_action_scalar_for_buffer, # This was a_t
                        reward, # This is r_{t+1}
                        is_done_for_current_agent_turn # This is d_{t+1}
                    )
                
                current_episode_score_for_my_agent += reward
                # total_environment_steps_collected is incremented when our agent acts

                if is_done_for_current_agent_turn:
                    env.step(None) # Our agent is done, pass None to PettingZoo
                    # The final transition was stored above.
                    # No need to break from agent_iter here, PettingZoo handles done agents.
                else:
                    total_environment_steps_collected += 1 # Increment only when agent takes a non-None step
                    # 1. Process current raw observation (which is s_{t+1} or s_0)
                    obs_dict_for_processing = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in raw_observation_dict.items()}
                    current_processed_obs_for_action = dreamer_agent.obs_processor.process_observation(obs_dict_for_processing)
                    
                    # 2. Select action a_{t+1} based on current_processed_obs_for_action (s_{t+1})
                    action_to_take = dreamer_agent.select_action_for_env(current_processed_obs_for_action, is_training_mode=True)
                    env.step(action_to_take)
                    
                    # 3. For the *next* iteration of our agent, current_processed_obs_for_action will be s_t,
                    #    and action_to_take will be a_t.
                    processed_observation_for_my_agent = current_processed_obs_for_action
                    dreamer_agent._last_action_scalar_for_buffer = action_to_take

                # Periodically trigger a training phase
                if total_environment_steps_collected > 0 and \
                   total_environment_steps_collected % config.train_every_env_steps == 0:
                    if len(dreamer_agent.memory) >= config.batch_size : # Check if enough sequences
                         dreamer_agent.train_agent_model_phase()

            else: # It's another agent's turn
                if is_done_for_current_agent_turn:
                    env.step(None) # Other agent is done
                else:
                    # Other agents take random actions
                    action_space_other = env.action_space(agent_step_id)
                    env.step(action_space_other.sample())
            
            if not env.agents: # Episode ends if all live agents are done
                break
        
        episode_scores_deque.append(current_episode_score_for_my_agent)
        average_score_recent = np.mean(episode_scores_deque)
        
        print(f"\rEpisode {i_episode}/{num_episodes} | Score: {current_episode_score_for_my_agent:.2f} | "
              f"Avg Score (100ep): {average_score_recent:.2f} | Total Env Steps: {total_environment_steps_collected}", end="")
        
        if i_episode % 20 == 0: # Print full line and save checkpoint less frequently
            print(f"\rEpisode {i_episode}/{num_episodes} | Score: {current_episode_score_for_my_agent:.2f} | "
                  f"Avg Score (100ep): {average_score_recent:.2f} | Total Env Steps: {total_environment_steps_collected}")
            if total_environment_steps_collected > config.batch_size * config.sequence_length : # Avoid saving too early if buffer not filled
                dreamer_agent.save_model_checkpoint()
            
    env.close()
    if total_environment_steps_collected > config.batch_size * config.sequence_length :
        dreamer_agent.save_model_checkpoint() # Final save after all episodes
    print("\nDreamer training loop finished.")
    return episode_scores_deque


if __name__ == '__main__':
    import time
    training_start_time = time.time()
    
    # --- HOW TO USE ---
    # 1. Ensure your environment module (e.g., 'til_environment.gridworld') is accessible.
    #    It should be in your PYTHONPATH or the same directory.
    #    The module must have an 'env()' function that returns a PettingZoo AECEnv.
    # 2. Install PyTorch: pip install torch numpy
    # 3. Run this script.

    # Example usage:
    # Replace "til_environment.gridworld" with the actual import path to your environment's .env() creator
    # e.g., if your file is my_env.py and contains gridworld.env(), it might be "my_env.gridworld"
    # If the file is gridworld.py and has env(), it's "gridworld.env" (if gridworld.py is a module)
    # Or just "gridworld" if gridworld.py is directly in PYTHONPATH and you call gridworld.env()
    
    # IMPORTANT: Adjust this path to correctly point to your environment module and the .env() function.
    # If your file is `til_environment/gridworld.py` and it contains `def env():`,
    # and `til_environment` is a package (has __init__.py), then use "til_environment.gridworld"
    # and the code will attempt to call `til_environment.gridworld.env()`.
    
    environment_module_path = "til_environment.gridworld" # Adjust this if your structure is different
                                     
    main_dreamer_training_loop(
        env_module_name=environment_module_path,
        num_episodes=10000,       # Dreamer typically requires many episodes/steps
        novice_setting=True,      # Start with the simpler novice map if available
        load_checkpoint_path=None, # Set to "dreamer_agent_final.pth" to resume training
        save_checkpoint_path="dreamer_agent_final.pth"
    )

    training_end_time = time.time()
    total_training_duration_minutes = (training_end_time - training_start_time) / 60
    print(f"Total Training Time = {total_training_duration_minutes:.2f} minutes")

