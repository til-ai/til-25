import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque, namedtuple

# --- Configuration (Adjust these as needed) ---
# Environment specific (match your game - from til_environment.gridworld)
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS_PER_EPISODE = 100 # NUM_ITERS in environment
VIEWCONE_CHANNELS = 8 # Each tile in viewcone is unpacked into 8 features
VIEWCONE_HEIGHT = 7   # self.viewcone_length = self.viewcone[2] + self.viewcone[3] + 1 = 2+4+1 = 7
VIEWCONE_WIDTH = 5    # self.viewcone_width = self.viewcone[0] + self.viewcone[1] + 1 = 2+2+1 = 5
OTHER_FEATURES_SIZE = 4 + 2 + 1 + 1 # direction_one_hot (4) + norm_location (2) + scout_role (1) + norm_step (1)

# Neural Network Hyperparameters
CNN_OUTPUT_CHANNELS_1 = 16
CNN_OUTPUT_CHANNELS_2 = 32
KERNEL_SIZE_1 = (3, 3) # For 7x5 input, (H,W)
STRIDE_1 = 1
KERNEL_SIZE_2 = (3, 3) # For smaller feature map
STRIDE_2 = 1
# Calculate flattened CNN output size dynamically in the model's __init__
# HIDDEN_LAYER_1_SIZE will be based on CNN output + OTHER_FEATURES_SIZE
MLP_HIDDEN_LAYER_1_SIZE = 128 # Size of the first MLP layer after concatenating CNN output and other features
MLP_HIDDEN_LAYER_2_SIZE = 128 # Size of the second MLP layer
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay
DROPOUT_RATE = 0.2 # Dropout rate for MLP layers

# Training Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size for training
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 1e-4    # Learning rate for the optimizer (reduced as per common advice)
WEIGHT_DECAY = 1e-5     # L2 regularization
TARGET_UPDATE_EVERY = 1000 # How often to update the target network (in global steps)
UPDATE_EVERY = 4        # How often to run a learning step (in agent steps within an episode)

# Epsilon-greedy exploration parameters (for training)
EPSILON_START = 1.0
EPSILON_END = 0.05 # Slightly higher end for continuous exploration on new maps
EPSILON_DECAY_RATE = 0.999 # Slower decay rate
MIN_EPSILON_FRAMES = int(5e4) # Number of frames to reach a significant portion of decay

# PER Parameters
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = int(1e5)
PER_EPSILON = 1e-6

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SumTree and PrioritizedReplayBuffer (Same as rl_agent_training_v1) ---
Experience = namedtuple("Experience", field_names=["state_viewcone", "state_other", "action", "reward", "next_state_viewcone", "next_state_other", "done"])

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state_viewcone, state_other, action, reward, next_state_viewcone, next_state_other, done):
        experience = Experience(state_viewcone, state_other, action, reward, next_state_viewcone, next_state_other, done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size, beta=PER_BETA_START):
        batch_idx = np.empty(batch_size, dtype=np.int32)
        batch_data = np.empty(batch_size, dtype=object)
        weights = np.empty(batch_size, dtype=np.float32)
        priority_segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            sampling_probabilities = priority / self.tree.total_priority
            weights[i] = np.power(self.tree.n_entries * sampling_probabilities, -beta)
            batch_idx[i] = index
            batch_data[i] = data
        
        weights /= weights.max()

        # Unpack experiences
        states_viewcone, states_other, actions, rewards, next_states_viewcone, next_states_other, dones = zip(*[e for e in batch_data])

        states_viewcone = torch.from_numpy(np.array(states_viewcone)).float().to(DEVICE) # N, C, H, W
        states_other = torch.from_numpy(np.array(states_other)).float().to(DEVICE)       # N, OTHER_FEATURES_SIZE
        actions = torch.from_numpy(np.vstack(actions)).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states_viewcone = torch.from_numpy(np.array(next_states_viewcone)).float().to(DEVICE)
        next_states_other = torch.from_numpy(np.array(next_states_other)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(DEVICE)
        
        return (states_viewcone, states_other, actions, rewards, next_states_viewcone, next_states_other, dones), batch_idx, torch.from_numpy(weights).float().to(DEVICE)

    def update_priorities(self, batch_indices, td_errors):
        priorities = np.abs(td_errors) + PER_EPSILON
        priorities = np.power(priorities, self.alpha)
        for idx, priority in zip(batch_indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

# --- CNN-DQN Model ---
class CNNDQN(nn.Module):
    def __init__(self, viewcone_channels, viewcone_height, viewcone_width, other_features_size, mlp_hidden1, mlp_hidden2, num_actions, dropout_rate):
        super(CNNDQN, self).__init__()
        self.viewcone_channels = viewcone_channels
        self.viewcone_height = viewcone_height
        self.viewcone_width = viewcone_width
        self.other_features_size = other_features_size

        # CNN for viewcone processing
        self.conv1 = nn.Conv2d(viewcone_channels, CNN_OUTPUT_CHANNELS_1, kernel_size=KERNEL_SIZE_1, stride=STRIDE_1, padding=1) # Padding to maintain size
        self.relu_conv1 = nn.ReLU()
        # Calculate H_out, W_out after conv1
        h_out1 = (viewcone_height + 2 * 1 - KERNEL_SIZE_1[0]) // STRIDE_1 + 1
        w_out1 = (viewcone_width + 2 * 1 - KERNEL_SIZE_1[1]) // STRIDE_1 + 1
        
        self.conv2 = nn.Conv2d(CNN_OUTPUT_CHANNELS_1, CNN_OUTPUT_CHANNELS_2, kernel_size=KERNEL_SIZE_2, stride=STRIDE_2, padding=1)
        self.relu_conv2 = nn.ReLU()
        # Calculate H_out, W_out after conv2
        h_out2 = (h_out1 + 2 * 1 - KERNEL_SIZE_2[0]) // STRIDE_2 + 1
        w_out2 = (w_out1 + 2 * 1 - KERNEL_SIZE_2[1]) // STRIDE_2 + 1

        # Calculate the flattened size of the CNN output
        self.cnn_output_flat_size = CNN_OUTPUT_CHANNELS_2 * h_out2 * w_out2
        print(f"CNN output HxW: {h_out2}x{w_out2}, Flattened CNN output size: {self.cnn_output_flat_size}")


        # MLP for combined features
        self.fc1_mlp = nn.Linear(self.cnn_output_flat_size + other_features_size, mlp_hidden1)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2_mlp = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc_output = nn.Linear(mlp_hidden2, num_actions)

    def forward(self, viewcone_input, other_features_input):
        # CNN path
        # Input viewcone_input: (N, C, H, W) -> (N, 8, 7, 5)
        x_cnn = self.relu_conv1(self.conv1(viewcone_input))
        x_cnn = self.relu_conv2(self.conv2(x_cnn))
        x_cnn_flat = x_cnn.view(-1, self.cnn_output_flat_size) # Flatten CNN output

        # Concatenate CNN output with other features
        # other_features_input: (N, other_features_size)
        combined_features = torch.cat((x_cnn_flat, other_features_input), dim=1)

        # MLP path
        x = self.relu_fc1(self.fc1_mlp(combined_features))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2_mlp(x))
        x = self.dropout2(x)
        
        return self.fc_output(x)

# --- Trainable RL Agent ---
class TrainableRLAgent:
    def __init__(self, model_load_path=None, model_save_path="trained_cnn_dqn_model.pth"):
        self.device = DEVICE
        print(f"Using device: {self.device}")

        self.policy_net = CNNDQN(VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH, 
                                 OTHER_FEATURES_SIZE, MLP_HIDDEN_LAYER_1_SIZE, 
                                 MLP_HIDDEN_LAYER_2_SIZE, OUTPUT_ACTIONS, DROPOUT_RATE).to(self.device)
        self.target_net = CNNDQN(VIEWCONE_CHANNELS, VIEWCONE_HEIGHT, VIEWCONE_WIDTH, 
                                 OTHER_FEATURES_SIZE, MLP_HIDDEN_LAYER_1_SIZE, 
                                 MLP_HIDDEN_LAYER_2_SIZE, OUTPUT_ACTIONS, DROPOUT_RATE).to(self.device)
        
        if model_load_path and os.path.exists(model_load_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_load_path, map_location=self.device))
                print(f"Loaded pre-trained policy_net from {model_load_path}")
            except Exception as e:
                print(f"Error loading model from {model_load_path}: {e}. Initializing with random weights.")
                self.policy_net.apply(self._initialize_weights)
        else:
            print(f"No model path provided or path '{model_load_path}' does not exist. Initializing policy_net with random weights.")
            self.policy_net.apply(self._initialize_weights)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        
        self.model_save_path = model_save_path
        self.t_step_episode = 0 # Counter for triggering learning within an episode
        self.beta = PER_BETA_START
        self.beta_increment_per_sampling = (1.0 - PER_BETA_START) / PER_BETA_FRAMES

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        tile_features = []
        # Bit 0: Tile type bit 0
        tile_features.append(float(tile_value & 0b01)) 
        # Bit 1: Tile type bit 1
        tile_features.append(float((tile_value & 0b10) >> 1))
        # Bit 2: Scout present
        tile_features.append(float((tile_value >> 2) & 1))
        # Bit 3: Guard present
        tile_features.append(float((tile_value >> 3) & 1))
        # Bit 4: Right wall
        tile_features.append(float((tile_value >> 4) & 1))
        # Bit 5: Bottom wall
        tile_features.append(float((tile_value >> 5) & 1))
        # Bit 6: Left wall
        tile_features.append(float((tile_value >> 6) & 1))
        # Bit 7: Top wall
        tile_features.append(float((tile_value >> 7) & 1))
        return tile_features # 8 features

    def process_observation(self, observation_dict):
        # 1. Viewcone (Height x Width) -> (Channels, Height, Width)
        raw_viewcone = observation_dict.get("viewcone", np.zeros((VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.uint8))
        # Ensure raw_viewcone has expected dimensions if it's from env (H, W)
        if raw_viewcone.shape != (VIEWCONE_HEIGHT, VIEWCONE_WIDTH):
             # Pad or truncate if necessary, or raise error
            print(f"Warning: Viewcone shape mismatch. Expected ({VIEWCONE_HEIGHT},{VIEWCONE_WIDTH}), got {raw_viewcone.shape}. Using zeros.")
            raw_viewcone = np.zeros((VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.uint8)

        processed_viewcone_channels = [] # List of channel arrays
        for r in range(VIEWCONE_HEIGHT):
            for c in range(VIEWCONE_WIDTH):
                tile_value = raw_viewcone[r, c]
                unpacked_features = self._unpack_viewcone_tile(tile_value) # 8 features
                if c == 0 and r == 0: # Initialize channels
                    for _ in range(VIEWCONE_CHANNELS):
                        processed_viewcone_channels.append(np.zeros((VIEWCONE_HEIGHT, VIEWCONE_WIDTH), dtype=np.float32))
                for channel_idx in range(VIEWCONE_CHANNELS):
                    processed_viewcone_channels[channel_idx][r, c] = unpacked_features[channel_idx]
        
        state_viewcone_np = np.array(processed_viewcone_channels) # Shape: (C, H, W) -> (8, 7, 5)

        # 2. Other features
        other_features_list = []
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4: direction_one_hot[direction] = 1.0
        other_features_list.extend(direction_one_hot)

        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        other_features_list.extend([norm_x, norm_y])

        scout_role = float(observation_dict.get("scout", 0))
        other_features_list.append(scout_role)

        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else 0.0
        other_features_list.append(norm_step)
        
        state_other_np = np.array(other_features_list, dtype=np.float32)
        
        return state_viewcone_np, state_other_np

    def select_action(self, state_viewcone_np, state_other_np, epsilon=0.0):
        if random.random() > epsilon:
            state_viewcone_tensor = torch.from_numpy(state_viewcone_np).float().unsqueeze(0).to(self.device)
            state_other_tensor = torch.from_numpy(state_other_np).float().unsqueeze(0).to(self.device)
            
            self.policy_net.eval() 
            with torch.no_grad():
                action_values = self.policy_net(state_viewcone_tensor, state_other_tensor)
            self.policy_net.train() 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(OUTPUT_ACTIONS))

    def step(self, state_viewcone, state_other, action, reward, next_state_viewcone, next_state_other, done, global_step_counter):
        self.memory.add(state_viewcone, state_other, action, reward, next_state_viewcone, next_state_other, done)
        
        self.t_step_episode = (self.t_step_episode + 1) % UPDATE_EVERY
        if self.t_step_episode == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences, indices, weights = self.memory.sample(BATCH_SIZE, beta=self.beta)
                self.learn(experiences, indices, weights, GAMMA)
                self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
    
    def learn(self, experiences, indices, importance_sampling_weights, gamma):
        states_viewcone, states_other, actions, rewards, next_states_viewcone, next_states_other, dones = experiences

        # Double DQN: action selection from policy_net, evaluation from target_net
        q_next_actions_policy = self.policy_net(next_states_viewcone, next_states_other).detach().max(1)[1].unsqueeze(1)
        q_targets_next = self.target_net(next_states_viewcone, next_states_other).detach().gather(1, q_next_actions_policy)
        
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.policy_net(states_viewcone, states_other).gather(1, actions)

        td_errors = (q_targets - q_expected).abs().cpu().detach().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        # Apply reward clipping for loss calculation if desired (e.g. rewards_clipped = torch.clamp(rewards, -1, 1))
        # Here, using original rewards for loss.
        loss = (importance_sampling_weights * nn.MSELoss(reduction='none')(q_expected, q_targets)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Target network updated at global step.")

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def reset_episode_counters(self):
        self.t_step_episode = 0


# --- Main Training Loop (Example) ---
def train_agent(env_module, num_episodes=10000, novice_track=False, load_model_from=None, save_model_to="trained_cnn_dqn_agent.pth"):
    env = env_module.env(env_wrappers=[], render_mode=None, novice=novice_track) # Use raw env for direct obs
    
    my_agent_id = env.possible_agents[0] 
    print(f"Training agent: {my_agent_id}")

    agent = TrainableRLAgent(model_load_path=load_model_from, model_save_path=save_model_to)
    
    scores_deque = deque(maxlen=100)
    scores = []
    epsilon = EPSILON_START
    global_total_steps = 0 # For target network updates and epsilon decay over total experience

    for i_episode in range(1, num_episodes + 1):
        env.reset() 
        agent.reset_episode_counters()
        
        current_rewards_this_episode = {id: 0 for id in env.possible_agents}
        last_processed_observation_for_my_agent = None # Stores (prev_state_vc, prev_state_other, prev_action)
        
        # PettingZoo iteration
        for pet_agent_id in env.agent_iter():
            observation_dict_raw, reward, termination, truncation, info = env.last()
            
            for ag_id in env.agents:
                 current_rewards_this_episode[ag_id] += env.rewards.get(ag_id, 0) # Accumulate PZ step rewards

            done = termination or truncation

            if done:
                action_to_take = None
            elif pet_agent_id == my_agent_id:
                # It's our agent's turn
                obs_dict_processed_for_agent = {
                    k: v if isinstance(v, (int, float, np.ndarray)) else v.tolist() 
                    for k, v in observation_dict_raw.items()
                }
                current_state_vc_np, current_state_other_np = agent.process_observation(obs_dict_processed_for_agent)
                
                if last_processed_observation_for_my_agent is not None:
                    prev_s_vc, prev_s_other, prev_a = last_processed_observation_for_my_agent
                    # The reward for (s,a) is `reward` from env.last() of *this* step
                    agent.step(prev_s_vc, prev_s_other, prev_a, reward, current_state_vc_np, current_state_other_np, done, global_total_steps)
                
                action_to_take = agent.select_action(current_state_vc_np, current_state_other_np, epsilon)
                last_processed_observation_for_my_agent = (current_state_vc_np, current_state_other_np, action_to_take)
                global_total_steps += 1

            else: # Other agents
                if env.action_space(pet_agent_id) is not None:
                     action_to_take = env.action_space(pet_agent_id).sample()
                else:
                    action_to_take = None
            
            env.step(action_to_take)
            
            if done and pet_agent_id == my_agent_id and last_processed_observation_for_my_agent is not None:
                prev_s_vc, prev_s_other, prev_a = last_processed_observation_for_my_agent
                final_next_s_vc = np.zeros_like(prev_s_vc) 
                final_next_s_other = np.zeros_like(prev_s_other)
                agent.step(prev_s_vc, prev_s_other, prev_a, reward, final_next_s_vc, final_next_s_other, True, global_total_steps)
                last_processed_observation_for_my_agent = None 
            
            # Target network update based on global steps
            if global_total_steps % TARGET_UPDATE_EVERY == 0 and global_total_steps > 0:
                agent.update_target_net()

        # End of episode
        episode_score = current_rewards_this_episode[my_agent_id]
        scores_deque.append(episode_score)
        scores.append(episode_score)
        
        # Decay epsilon based on global steps or episodes
        if global_total_steps > MIN_EPSILON_FRAMES : # Start decay after some initial exploration
             epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)
        # Alternatively, decay per episode: epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_PER_EPISODE)

        print(f'\rEpisode {i_episode}\tAvg Score: {np.mean(scores_deque):.2f}\tEpsilon: {epsilon:.4f}\tGlobal Steps: {global_total_steps}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAvg Score: {np.mean(scores_deque):.2f}\tEpsilon: {epsilon:.4f}\tGlobal Steps: {global_total_steps}')
            agent.save_model()
            
    env.close()
    return scores


if __name__ == '__main__':
    try:
        from til_environment import gridworld 
        print("Successfully imported til_environment.gridworld")
        
        trained_scores = train_agent(
            gridworld, 
            num_episodes=200, # Increased episodes
            novice_track=False, # Set to False for varying maps
            load_model_from=None, 
            save_model_to="my_wargame_cnn_agent.pth"
        )
        print("Training finished.")

    except ImportError:
        print("Could not import 'til_environment.gridworld'.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

