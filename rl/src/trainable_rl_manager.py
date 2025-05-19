import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque, namedtuple

# --- Configuration (Adjust these as needed) ---
# Environment specific (match your game)
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS_PER_EPISODE = 100 # Max steps in one round of the game

# Neural Network Hyperparameters (from rl_agent_python_v1)
INPUT_FEATURES = 288  # 7*5*8 (viewcone) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay

# Training Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 32         # Minibatch size for training
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 1e-4    # Learning rate for the optimizer
TARGET_UPDATE_EVERY = 100 # How often to update the target network (in steps)
UPDATE_EVERY = 4        # How often to run a learning step (in steps)

# Epsilon-greedy exploration parameters (for training)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995 # Multiplicative decay factor per episode/fixed number of steps

# PER Parameters
PER_ALPHA = 0.6  # Prioritization exponent (0 for uniform, 1 for full prioritization)
PER_BETA_START = 0.4 # Initial importance sampling exponent
PER_BETA_FRAMES = int(1e5) # Number of frames over which beta is annealed to 1.0
PER_EPSILON = 1e-6 # Small constant to ensure non-zero priority

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SumTree for Prioritized Replay Buffer ---
class SumTree:
    """
    A SumTree is a binary tree data structure where the value of a parent node
    is the sum of its children. It is used for efficient sampling from a
    distribution. Leaf nodes store priorities, and internal nodes store sums.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # Tree storage
        self.data = np.zeros(capacity, dtype=object) # Data storage (transitions)
        self.data_pointer = 0 # Current position to write new data
        self.n_entries = 0 # Current number of entries in the buffer

    def add(self, priority, data):
        """Add priority score and data to the tree."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0 # Cycle back to the beginning

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """Update priority of a node and propagate changes upwards."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate the change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        """Find sample on leaf node based on a cumulative sum value."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree): # Reached leaf
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

# --- Prioritized Replay Buffer ---
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha # Controls how much prioritization is used (0=uniform, 1=full)
        self.max_priority = 1.0 # Max priority for new experiences

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to the buffer with max priority."""
        experience = Experience(state, action, reward, next_state, done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size, beta=PER_BETA_START):
        """
        Samples a batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to sample.
            beta (float): Importance-sampling exponent.
        Returns:
            tuple: (experiences, indices, weights)
        """
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
            weights[i] = np.power(self.tree.n_entries * sampling_probabilities, -beta) # (N * P(i))^-beta
            batch_idx[i] = index
            batch_data[i] = data
        
        weights /= weights.max() # Normalize for stability

        states, actions, rewards, next_states, dones = zip(*[e for e in batch_data])

        states = torch.from_numpy(np.vstack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(actions)).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones), batch_idx, torch.from_numpy(weights).float().to(DEVICE)

    def update_priorities(self, batch_indices, td_errors):
        """Updates the priorities of sampled experiences."""
        priorities = np.abs(td_errors) + PER_EPSILON # Add epsilon to ensure non-zero priority
        priorities = np.power(priorities, self.alpha)
        
        for idx, priority in zip(batch_indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority) # Update max_priority

    def __len__(self):
        return self.tree.n_entries

# --- Deep Q-Network (DQN) Model (same as in rl_agent_python_v1) ---
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Trainable RL Agent ---
class TrainableRLAgent:
    def __init__(self, model_load_path=None, model_save_path="trained_dqn_model.pth"):
        self.device = DEVICE
        print(f"Using device: {self.device}")

        self.policy_net = DQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_ACTIONS).to(self.device)
        self.target_net = DQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_ACTIONS).to(self.device)
        
        if model_load_path and os.path.exists(model_load_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_load_path, map_location=self.device))
                print(f"Loaded pre-trained policy_net from {model_load_path}")
            except Exception as e:
                print(f"Error loading model from {model_load_path}: {e}. Initializing with random weights.")
                self.policy_net.apply(self._initialize_weights)
        else:
            print(f"No model path provided or path {model_load_path} does not exist. Initializing policy_net with random weights.")
            self.policy_net.apply(self._initialize_weights)

        self.target_net.load_state_dict(self.policy_net.state_dict()) # Initialize target_net with policy_net weights
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        
        self.model_save_path = model_save_path
        self.t_step = 0 # Counter for triggering learning and target network updates
        self.beta = PER_BETA_START
        self.beta_increment_per_sampling = (1.0 - PER_BETA_START) / PER_BETA_FRAMES


    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value): # Same as in rl_agent_python_v1
        tile_features = []
        tile_features.append(float(tile_value & 0b01)) 
        tile_features.append(float((tile_value & 0b10) >> 1))
        for i in range(2, 8):
            tile_features.append(float((tile_value >> i) & 1))
        return tile_features

    def process_observation(self, observation_dict): # Same as in rl_agent_python_v1
        processed_features = []
        viewcone = observation_dict.get("viewcone", [])
        for r in range(7):
            for c in range(5):
                tile_value = viewcone[r][c] if r < len(viewcone) and c < len(viewcone[r]) else 0
                processed_features.extend(self._unpack_viewcone_tile(tile_value))
        
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4: direction_one_hot[direction] = 1.0
        processed_features.extend(direction_one_hot)

        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        processed_features.extend([norm_x, norm_y])

        scout_role = float(observation_dict.get("scout", 0))
        processed_features.append(scout_role)

        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else 0.0
        processed_features.append(norm_step)
        
        # Ensure correct feature length (should be INPUT_FEATURES)
        if len(processed_features) != INPUT_FEATURES:
            # This indicates an issue with feature processing or constants
            raise ValueError(f"Feature length mismatch. Expected {INPUT_FEATURES}, got {len(processed_features)}")

        return np.array(processed_features, dtype=np.float32) # Return as numpy array for buffer

    def select_action(self, state_np, epsilon=0.0):
        """
        Selects an action using epsilon-greedy policy.
        Args:
            state_np (np.ndarray): Processed state as a numpy array.
            epsilon (float): Exploration rate.
        Returns:
            int: Selected action.
        """
        if random.random() > epsilon:
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
            self.policy_net.eval() # Set to evaluation mode for action selection
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            self.policy_net.train() # Set back to training mode
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(OUTPUT_ACTIONS))

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences, indices, weights = self.memory.sample(BATCH_SIZE, beta=self.beta)
                self.learn(experiences, indices, weights, GAMMA)
                # Anneal beta
                self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)


        # Update target network
        if self.t_step % TARGET_UPDATE_EVERY == 0 : # A bit confusing, t_step is modulo UPDATE_EVERY
                                                    # A global step counter might be better here.
                                                    # Let's assume a global step counter is incremented elsewhere
                                                    # and this check is done based on that.
                                                    # For now, this will update target less frequently.
            self.update_target_net()


    def learn(self, experiences, indices, importance_sampling_weights, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Q_targets = r + γ * Q_target(s', argmax_a Q_policy(s', a))
        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            indices (np.ndarray): indices of these experiences in the SumTree
            importance_sampling_weights (torch.Tensor): weights for IS
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from policy network
        # This is for Double DQN: action selection from policy_net, evaluation from target_net
        q_next_policy = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
        # Get Q values for next_states from target_net using actions selected by policy_net
        q_targets_next = self.target_net(next_states).detach().gather(1, q_next_policy)
        
        # Compute Q targets for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from policy_net
        q_expected = self.policy_net(states).gather(1, actions)

        # Compute TD errors for PER
        td_errors = (q_targets - q_expected).abs().cpu().detach().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        # Compute loss (element-wise multiplication with IS weights)
        loss = (importance_sampling_weights * nn.MSELoss(reduction='none')(q_expected, q_targets)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()


    def update_target_net(self):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        # For hard update:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # print("Updated target network.")

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def reset_state(self): # For compatibility with potential stateful components, not used in this DQN
        pass


# --- Main Training Loop (Example) ---
def train_agent(env_module, num_episodes=2000, novice_track=False, load_model_from=None, save_model_to="trained_dqn_agent.pth"):
    """
    Main training loop for the RL agent.
    Args:
        env_module: The imported environment module (e.g., til_environment.gridworld).
        num_episodes (int): Number of episodes to train for.
        novice_track (bool): If true, uses the novice environment settings.
        load_model_from (str, optional): Path to load a pre-trained model.
        save_model_to (str): Path to save the trained model.
    """
    # Initialize your game environment
    # This assumes your 'til_environment.gridworld' has an 'env' function
    # that returns a PettingZoo-like environment.
    env = env_module.env(env_wrappers=[], render_mode=None, novice=novice_track)
    
    # Assuming your agent is always the first one in possible_agents
    # Adjust if your setup is different or if you want to train a specific agent
    my_agent_id = env.possible_agents[0] 
    print(f"Training agent: {my_agent_id}")
    print(f"Action space for {my_agent_id}: {env.action_space(my_agent_id)}")
    print(f"Observation space for {my_agent_id}: {env.observation_space(my_agent_id)}")

    agent = TrainableRLAgent(model_load_path=load_model_from, model_save_path=save_model_to)
    
    scores_deque = deque(maxlen=100) # For tracking recent scores
    scores = [] # List of scores from all episodes
    epsilon = EPSILON_START
    total_steps_taken = 0

    for i_episode in range(1, num_episodes + 1):
        env.reset() # Reset environment at the start of each episode
        agent.reset_state() # Reset agent's internal state if any (not for this DQN)
        
        # The environment interaction loop from your test script
        current_rewards_this_episode = {agent_id: 0 for agent_id in env.possible_agents}
        
        # Get initial observation for our agent
        # This part needs careful handling with PettingZoo's agent_iter
        # We need to get the first observation for our agent
        
        # The loop below processes all agents. We only train `my_agent_id`.
        # We need to store the state for `my_agent_id` to pass to `agent.step`
        
        last_observation_for_my_agent = None
        
        for pet_agent_id in env.agent_iter(): # PettingZoo's iterator
            observation, reward, termination, truncation, info = env.last()
            
            # Accumulate rewards for all agents for this step
            for ag_id in env.agents: # env.agents are live agents in current step
                 current_rewards_this_episode[ag_id] += env.rewards.get(ag_id, 0)

            done = termination or truncation

            if done: # If an agent is done, it might not take an action
                action = None # PettingZoo expects None if agent is done
            elif pet_agent_id == my_agent_id:
                # It's our agent's turn
                # 1. Process observation
                obs_dict = {k: v if isinstance(v, (int, float)) else v.tolist() for k, v in observation.items()}
                current_state_np = agent.process_observation(obs_dict)
                
                # 2. Store previous transition if available
                if last_observation_for_my_agent is not None:
                    # last_observation_for_my_agent = (prev_state, prev_action, prev_reward_for_my_agent)
                    prev_state_np, prev_action, prev_reward = last_observation_for_my_agent
                    # The reward for the (s,a) pair is what we received *after* taking action 'a' in state 's'
                    # which is the 'reward' variable from env.last() *now*
                    agent.step(prev_state_np, prev_action, reward, current_state_np, done)
                    total_steps_taken +=1

                # 3. Select action
                action = agent.select_action(current_state_np, epsilon)
                
                # 4. Store current state, action, and this step's reward for the *next* transition
                last_observation_for_my_agent = (current_state_np, action, reward) # reward here is for the current (s,a)

            else:
                # Other agents take random actions (or use their own policies if implemented)
                if env.action_space(pet_agent_id) is not None:
                     action = env.action_space(pet_agent_id).sample()
                else:
                    action = None # Should not happen if agent is not done

            env.step(action) # Step the environment with the chosen action (or None)
            
            if done and pet_agent_id == my_agent_id and last_observation_for_my_agent is not None:
                # If our agent is done, we need to record the final transition
                prev_state_np, prev_action, _ = last_observation_for_my_agent 
                # The final reward is `reward` from env.last() when done is true
                # The next_state is not critical as it's a terminal state, can be zeros or current_state_np
                final_next_state_np = np.zeros_like(prev_state_np) # Or current_state_np
                agent.step(prev_state_np, prev_action, reward, final_next_state_np, True)
                total_steps_taken +=1
                last_observation_for_my_agent = None # Reset for next episode start

        # End of episode
        episode_score = current_rewards_this_episode[my_agent_id]
        scores_deque.append(episode_score)
        scores.append(episode_score)
        
        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon) # Decay epsilon

        print(f'\rEpisode {i_episode}\tAverage Score (last 100): {np.mean(scores_deque):.2f}\tEpsilon: {epsilon:.4f}\tTotal Steps: {total_steps_taken}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score (last 100): {np.mean(scores_deque):.2f}\tEpsilon: {epsilon:.4f}\tTotal Steps: {total_steps_taken}')
            agent.save_model()
        
        if np.mean(scores_deque) >= 200.0: # Example condition to stop training
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
            agent.save_model()
            break
            
    env.close()
    return scores


if __name__ == '__main__':
    # --- HOW TO USE ---
    # 1. Make sure you have 'til_environment' and its 'gridworld' module accessible.
    #    (e.g., it's in your PYTHONPATH or the same directory)
    # 2. Install PyTorch: pip install torch
    # 3. Run this script: python your_training_script_name.py
    import time
    start = time.time()
    # Example:
    try:
        from til_environment import gridworld # Assuming this is your environment module
        print("Successfully imported til_environment.gridworld")
        
        # Start training
        # Set load_model_from to a .pth file to continue training or fine-tune
        # Set save_model_to to where you want the final model to be saved
        trained_scores = train_agent(
            gridworld, 
            num_episodes=100000, # Adjust as needed
            novice_track=False, # Or True for the Novice track map
            load_model_from="agent_66k_eps.pth", # "trained_dqn_agent.pth" to resume
            save_model_to="agent_166k_eps.pth"
        )
        print("Training finished.")

        # You can add plotting for scores if you like:
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(trained_scores)), trained_scores)
        # plt.ylabel('Score')
        # plt.xlabel('Episode #')
        # plt.show()

    except ImportError:
        print("Could not import 'til_environment.gridworld'.")
        print("Please ensure the environment module is correctly set up and accessible.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    end = time.time()
    print("Time taken = ", end-start)
    print(f"Time taken = {end-start:.4f} seconds")