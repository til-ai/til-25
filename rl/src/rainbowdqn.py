import json
import torch
import torch.nn as nn
import torch.nn.functional as F # Added for Dueling DQN
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque, namedtuple
import time # For timing the script
import traceback # For detailed error printing

# --- Configuration (Adjust these as needed) ---
# Environment specific (match your game)
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS_PER_EPISODE = 100 # Max steps in one round of the game

# Neural Network Hyperparameters
INPUT_FEATURES = 288  # 7*5*8 (viewcone) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 256 # Common hidden layer for Dueling DQN
HIDDEN_LAYER_2_SIZE = 256 # Common hidden layer for Dueling DQN
VALUE_STREAM_HIDDEN_SIZE = 128 # Hidden layer size for the value stream
ADVANTAGE_STREAM_HIDDEN_SIZE = 128 # Hidden layer size for the advantage stream
OUTPUT_ACTIONS = 5   # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay

# Training Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 32         # Minibatch size for training
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 1e-4    # Learning rate for the optimizer
TARGET_UPDATE_EVERY = 1000 # How often to update the target network (in learning steps)
UPDATE_EVERY = 4        # How often to run a learning step (in global environment steps)

# Epsilon-greedy exploration parameters (for training)
EPSILON_START = 0.015
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.999 # Multiplicative decay factor per episode

# PER Parameters
PER_ALPHA = 0.6  # Prioritization exponent (0 for uniform, 1 for full prioritization)
PER_BETA_START = 0.4 # Initial importance sampling exponent
PER_BETA_FRAMES = int(1e5) # Number of frames over which beta is annealed to 1.0
PER_EPSILON = 1e-6 # Small constant to ensure non-zero priority

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SumTree for Prioritized Replay Buffer ---
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

# --- Prioritized Replay Buffer ---
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size, beta=PER_BETA_START):
        batch_idx = np.empty(batch_size, dtype=np.int32)
        batch_data = np.empty(batch_size, dtype=object)
        weights = np.empty(batch_size, dtype=np.float32)

        if self.tree.n_entries == 0:
             return None, None, None # Cannot sample if buffer is empty

        priority_segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            # Ensure value does not exceed total_priority, especially if total_priority is very small
            value = np.random.uniform(a, min(b, self.tree.total_priority)) 
            
            index, priority, data = self.tree.get_leaf(value)
            
            if priority == 0 or self.tree.total_priority == 0: # Avoid division by zero
                sampling_probabilities = PER_EPSILON / self.tree.capacity # Smallest possible probability
            else:
                sampling_probabilities = priority / self.tree.total_priority

            if self.tree.n_entries == 0 or sampling_probabilities == 0:
                 weights[i] = 1.0 # Default weight if something is wrong
            else:
                 weights[i] = np.power(self.tree.n_entries * sampling_probabilities, -beta)

            batch_idx[i] = index
            batch_data[i] = data
        
        if weights.max() > 0: # Normalize weights
             weights /= weights.max()
        else: # If all weights somehow became zero
             weights = np.ones_like(weights, dtype=np.float32) / batch_size


        states, actions, rewards, next_states, dones = zip(*[e for e in batch_data])

        states = torch.from_numpy(np.vstack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(actions)).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones), batch_idx, torch.from_numpy(weights).float().to(DEVICE)

    def update_priorities(self, batch_indices, td_errors):
        priorities = np.abs(td_errors) + PER_EPSILON
        priorities = np.power(priorities, self.alpha)
        for idx, priority in zip(batch_indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

# --- Dueling Deep Q-Network (DQN) Model ---
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, common_hidden_dim1, common_hidden_dim2,
                 value_stream_hidden, advantage_stream_hidden, output_dim):
        super(DuelingDQN, self).__init__()
        self.output_dim = output_dim

        # Common feature learning layers
        self.fc1 = nn.Linear(input_dim, common_hidden_dim1)
        self.fc2 = nn.Linear(common_hidden_dim1, common_hidden_dim2)

        # Value stream
        self.value_fc1 = nn.Linear(common_hidden_dim2, value_stream_hidden)
        self.value_fc2 = nn.Linear(value_stream_hidden, 1) # Outputs V(s)

        # Advantage stream
        self.advantage_fc1 = nn.Linear(common_hidden_dim2, advantage_stream_hidden)
        self.advantage_fc2 = nn.Linear(advantage_stream_hidden, output_dim) # Outputs A(s,a)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value) 

        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# --- Trainable RL Agent ---
class TrainableRLAgent:
    def __init__(self, model_load_path=None, model_save_path="trained_rainbow_dqn_model.pth"):
        self.device = DEVICE
        print(f"Using device: {self.device}")

        self.policy_net = DuelingDQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE,
                                     VALUE_STREAM_HIDDEN_SIZE, ADVANTAGE_STREAM_HIDDEN_SIZE,
                                     OUTPUT_ACTIONS).to(self.device)
        self.target_net = DuelingDQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE,
                                     VALUE_STREAM_HIDDEN_SIZE, ADVANTAGE_STREAM_HIDDEN_SIZE,
                                     OUTPUT_ACTIONS).to(self.device)
        
        if model_load_path and os.path.exists(model_load_path):
            try:
                # Load the state dict. If keys don't match (e.g. loading a non-Dueling model into DuelingDQN),
                # this will raise an error. The catch block will then initialize randomly.
                self.policy_net.load_state_dict(torch.load(model_load_path, map_location=self.device))
                print(f"Loaded pre-trained policy_net from {model_load_path}")
            except Exception as e:
                print(f"Error loading model from {model_load_path}: {e}. Initializing policy_net with random weights.")
                self.policy_net.apply(self._initialize_weights)
        else:
            if model_load_path: # Only print if a path was given but not found
                 print(f"Model path '{model_load_path}' does not exist.")
            print("Initializing policy_net with random weights.")
            self.policy_net.apply(self._initialize_weights)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        
        self.model_save_path = model_save_path
        self.total_env_steps = 0 
        self.learning_steps = 0  
        self.beta = PER_BETA_START
        # Calculate beta increment based on total frames/steps for annealing, not per sampling
        # If PER_BETA_FRAMES is the total number of environment steps over which to anneal beta
        self.beta_increment_on_learn = (1.0 - PER_BETA_START) / (PER_BETA_FRAMES / UPDATE_EVERY) if PER_BETA_FRAMES > 0 else 0


    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        tile_features = []
        # Bits 0, 1: Tile type (Value of last 2 bits (tile & 0b11))
        # 0 (00): No vision, 1 (01): Empty tile, 2 (10): Recon, 3 (11): Mission
        # Representing this as two binary features as in the original template:
        tile_features.append(float(tile_value & 0b01))       # Bit 0
        tile_features.append(float((tile_value & 0b10) >> 1)) # Bit 1
        
        # Bits 2-7: Occupancy and walls
        for i in range(2, 8): # Bit index 2 to 7
            tile_features.append(float((tile_value >> i) & 1))
        return tile_features # Total 8 features per tile

    def process_observation(self, observation_dict):
        processed_features = []
        viewcone = observation_dict.get("viewcone", [])
        for r in range(7): # 7 rows
            for c in range(5): # 5 columns
                tile_value = 0 
                if r < len(viewcone) and c < len(viewcone[r]):
                    tile_value = viewcone[r][c]
                processed_features.extend(self._unpack_viewcone_tile(tile_value))
        # Expected: 7 * 5 * 8 = 280 features

        direction = observation_dict.get("direction", 0) # 0:R, 1:D, 2:L, 3:U
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4: direction_one_hot[direction] = 1.0
        processed_features.extend(direction_one_hot) # 4 features

        location = observation_dict.get("location", [0, 0]) # (x,y)
        # Normalize location from 0 to MAP_SIZE-1 to 0 to 1.0
        norm_x = location[0] / (MAP_SIZE_X - 1.0) if MAP_SIZE_X > 1 else 0.0
        norm_y = location[1] / (MAP_SIZE_Y - 1.0) if MAP_SIZE_Y > 1 else 0.0
        processed_features.extend([norm_x, norm_y]) # 2 features

        scout_role = float(observation_dict.get("scout", 0)) # 1 if scout, 0 if guard
        processed_features.append(scout_role) # 1 feature

        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else 0.0
        processed_features.append(norm_step) # 1 feature
        
        if len(processed_features) != INPUT_FEATURES:
            raise ValueError(f"Feature length mismatch. Expected {INPUT_FEATURES}, got {len(processed_features)}. "
                             f"Viewcone data example (first row if exists): {viewcone[0] if viewcone else 'empty'}")
        return np.array(processed_features, dtype=np.float32)

    def select_action(self, state_np, epsilon=0.0):
        if random.random() > epsilon:
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
            self.policy_net.eval() 
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            self.policy_net.train() 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(OUTPUT_ACTIONS))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.total_env_steps += 1
        
        if self.total_env_steps % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences, indices, weights = self.memory.sample(BATCH_SIZE, beta=self.beta)
                if experiences: 
                    self.learn(experiences, indices, weights, GAMMA)
                    self.learning_steps += 1
                    self.beta = min(1.0, self.beta + self.beta_increment_on_learn)

                    if self.learning_steps % TARGET_UPDATE_EVERY == 0:
                        self.update_target_net()

    def learn(self, experiences, indices, importance_sampling_weights, gamma):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            q_next_policy_actions = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
            q_targets_next = self.target_net(next_states).detach().gather(1, q_next_policy_actions)

        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.policy_net(states).gather(1, actions)
        td_errors = (q_targets - q_expected).abs().cpu().detach().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)

        loss = (importance_sampling_weights * F.mse_loss(q_expected, q_targets, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Updated target network at learning step {self.learning_steps}, total env steps {self.total_env_steps}.")

    def save_model(self):
        try:
            torch.save(self.policy_net.state_dict(), self.model_save_path)
            print(f"Model saved to {self.model_save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def reset_state(self):
        pass

# --- Main Training Loop (Example, adapted from your template) ---
def train_agent(env_module, num_episodes=2000, novice_track=False, load_model_from=None, save_model_to="trained_rainbow_agent.pth"):
    # Initialize game environment using PettingZoo AEC API style
    env = env_module.env(env_wrappers=[], render_mode=None, novice=novice_track) 
    
    # Determine the agent ID to train. Assuming it's the first one.
    # This might need adjustment if the environment has a fixed ID for the controllable agent.
    my_agent_id = env.possible_agents[0] 
    print(f"Training agent: {my_agent_id}")
    # print(f"Action space for {my_agent_id}: {env.action_space(my_agent_id)}")
    # print(f"Observation space for {my_agent_id}: {env.observation_space(my_agent_id)}")

    agent = TrainableRLAgent(model_load_path=load_model_from, model_save_path=save_model_to)
    
    scores_deque = deque(maxlen=100) # For tracking recent scores
    all_episode_scores = [] # List of scores from all episodes
    epsilon = EPSILON_START

    for i_episode in range(1, num_episodes + 1):
        env.reset() # Reset environment at the start of each episode
        agent.reset_state() 
        
        current_episode_rewards_for_my_agent = 0.0
        # Store (state, action) for my_agent_id to form a transition later
        last_state_action_tuple_for_my_agent = None 

        # PettingZoo AEC API loop
        for pet_agent_id in env.agent_iter(): # Iterates through agents whose turn it is
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if pet_agent_id == my_agent_id:
                current_episode_rewards_for_my_agent += reward # Accumulate reward for my agent from its last action

            if done: # Agent is done for this episode
                action_to_take = None # PettingZoo expects None if agent is done
                if pet_agent_id == my_agent_id and last_state_action_tuple_for_my_agent:
                    # This is the final transition for our agent
                    prev_state_np, prev_action = last_state_action_tuple_for_my_agent
                    
                    # Process the terminal observation to get the final next_state
                    # If observation is None (can happen if agent is already removed), use a zero vector or prev_state_np
                    if observation is not None:
                        obs_dict_terminal = {k: v if isinstance(v, (int, float, bool)) else (v.tolist() if hasattr(v, 'tolist') else v) for k, v in observation.items()}
                        final_next_state_np = agent.process_observation(obs_dict_terminal)
                    else: # Fallback if terminal observation is None
                        final_next_state_np = np.zeros_like(prev_state_np)

                    agent.step(prev_state_np, prev_action, reward, final_next_state_np, True) # True for done
                    last_state_action_tuple_for_my_agent = None
            elif pet_agent_id == my_agent_id:
                # It's our agent's turn and it's not done
                obs_dict = {k: v if isinstance(v, (int, float, bool)) else (v.tolist() if hasattr(v, 'tolist') else v) for k, v in observation.items()}
                current_state_np = agent.process_observation(obs_dict)

                if last_state_action_tuple_for_my_agent is not None:
                    # We have a previous state and action. 'reward' is outcome of that. 'current_state_np' is s'.
                    prev_state_np, prev_action = last_state_action_tuple_for_my_agent
                    agent.step(prev_state_np, prev_action, reward, current_state_np, False) # False for done

                action_to_take = agent.select_action(current_state_np, epsilon)
                last_state_action_tuple_for_my_agent = (current_state_np, action_to_take)
            else:
                # Other agents' turns (e.g., fixed policy, random, or other learning agents)
                if env.action_space(pet_agent_id) is not None:
                     action_to_take = env.action_space(pet_agent_id).sample() # Example: random action
                else: # Should not happen if agent is not done and has an action space
                    action_to_take = None 
            
            env.step(action_to_take) # Step the environment with the chosen action
            
            # If all agents are done, the env.agents list will be empty, and agent_iter will stop.
            if not env.agents: 
                 break
        
        # End of episode
        scores_deque.append(current_episode_rewards_for_my_agent)
        all_episode_scores.append(current_episode_rewards_for_my_agent)
        
        epsilon = max(EPSILON_END, EPSILON_DECAY_RATE * epsilon) # Decay epsilon

        print(f'\rEpisode {i_episode}\tAvg Score (Last 100): {np.mean(scores_deque):.2f}\tEpsilon: {epsilon:.4f}\tTotal Env Steps: {agent.total_env_steps}\tLearning Steps: {agent.learning_steps}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAvg Score (Last 100): {np.mean(scores_deque):.2f}\tEpsilon: {epsilon:.4f}\tTotal Env Steps: {agent.total_env_steps}\tLearning Steps: {agent.learning_steps}')
            if agent.model_save_path: # Save model if path is provided
                 agent.save_model()
        
        # Example condition to stop training (adjust as needed)
        # Based on game rewards: Recon=1, Challenge=5, Captured=-50.
        # A positive average score over 100 episodes would be a good start.
        if len(scores_deque) == 100 and np.mean(scores_deque) >= 80.0: # Example target average score
            print(f'\nEnvironment potentially solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
            if agent.model_save_path:
                agent.save_model()
            break
            
    env.close()
    return all_episode_scores


# --- Dummy Environment for Testing (til_environment.gridworld placeholder) ---
# This is kept for reference or if you need to test without the actual environment.
# To use it, you would need to modify the `if __name__ == '__main__':` block.
try:
    import gymnasium as gym # PettingZoo often uses gymnasium.spaces
except ImportError:
    print("Warning: gymnasium not installed. Dummy environment might not be fully functional if used.")
    gym = None # Set to None if not available

class DummyGridworldEnv:
    def __init__(self, env_wrappers=None, render_mode=None, novice=False):
        self.possible_agents = [f"agent_{i}" for i in range(4)] # Typically 4 agents
        self.agent_selection = None
        self.agents = [] 
        self._agent_selector_iter = None # Iterator for agent turns

        if gym:
            self.observation_spaces = {
                agent_id: gym.spaces.Dict({
                    "viewcone": gym.spaces.Box(0, 255, shape=(7,5), dtype=np.uint8),
                    "direction": gym.spaces.Discrete(4),
                    "scout": gym.spaces.Discrete(2),
                    "location": gym.spaces.Box(low=np.array([0,0]), high=np.array([MAP_SIZE_X-1, MAP_SIZE_Y-1]), dtype=np.int16),
                    "step": gym.spaces.Discrete(MAX_STEPS_PER_EPISODE + 1)
                }) for agent_id in self.possible_agents
            }
            self.action_spaces = {
                agent_id: gym.spaces.Discrete(OUTPUT_ACTIONS) for agent_id in self.possible_agents
            }
        else: # Fallback if gymnasium is not installed
            self.observation_spaces = {agent_id: None for agent_id in self.possible_agents}
            self.action_spaces = {agent_id: None for agent_id in self.possible_agents}

        self._rewards = {agent_id: 0 for agent_id in self.possible_agents}
        self._terminations = {agent_id: False for agent_id in self.possible_agents}
        self._truncations = {agent_id: False for agent_id in self.possible_agents}
        self._infos = {agent_id: {} for agent_id in self.possible_agents}
        self.current_step_count = 0
        self.novice_mode = novice
        # print(f"DummyEnv Initialized. Novice: {self.novice_mode}")

    def observation_space(self, agent): return self.observation_spaces[agent]
    def action_space(self, agent): return self.action_spaces[agent]

    def _generate_dummy_obs(self, agent_id):
        viewcone = np.random.randint(0, 256, size=(7,5), dtype=np.uint8)
        # Example: make one tile a recon (value 2) and one a mission (value 3)
        if random.random() < 0.5: viewcone[random.randint(0,6)][random.randint(0,4)] = 2 
        if random.random() < 0.5: viewcone[random.randint(0,6)][random.randint(0,4)] = 3
        if random.random() < 0.3: viewcone[random.randint(0,6)][random.randint(0,4)] |= 128 # Top wall
        return {
            "viewcone": viewcone, "direction": random.randint(0,3),
            "scout": 1 if agent_id == self.possible_agents[0] else 0, # Agent 0 is scout
            "location": np.array([random.randint(0, MAP_SIZE_X-1), random.randint(0, MAP_SIZE_Y-1)]),
            "step": self.current_step_count
        }

    def reset(self, seed=None, options=None):
        if seed is not None: random.seed(seed); np.random.seed(seed)
        self.agents = list(self.possible_agents)
        self._agent_selector_iter = iter(self.agents)
        self.agent_selection = next(self._agent_selector_iter)
        self.current_step_count = 0
        self._rewards = {agent_id: 0 for agent_id in self.possible_agents}
        self._terminations = {agent_id: False for agent_id in self.possible_agents}
        self._truncations = {agent_id: False for agent_id in self.possible_agents}
        self._infos = {agent_id: {} for agent_id in self.possible_agents}
        # Per PettingZoo AEC API, reset() does not return obs. First obs is from last().

    def last(self):
        agent_id = self.agent_selection
        obs = self._generate_dummy_obs(agent_id) if not (self._terminations[agent_id] or self._truncations[agent_id]) else None
        return obs, self._rewards[agent_id], self._terminations[agent_id], self._truncations[agent_id], self._infos[agent_id]

    def step(self, action):
        if self._terminations[self.agent_selection] or self._truncations[self.agent_selection]:
            # If agent is done, remove from active agents list for this step cycle
            if self.agent_selection in self.agents: self.agents.remove(self.agent_selection)
            # Then, select next agent or end iteration if all are done
            if self.agents:
                self.agent_selection = next(self._agent_selector_iter, None) # Get next from current iter
                if self.agent_selection is None: # Reached end of current agent list
                    self._agent_selector_iter = iter(self.agents) # New iterator for remaining agents
                    self.agent_selection = next(self._agent_selector_iter)
            else: # No agents left
                self.agent_selection = None
            return

        current_agent = self.agent_selection
        self._rewards[current_agent] = 0 # Reset reward for this agent for this step
        if action is not None:
            if action == 0: self._rewards[current_agent] += random.uniform(0, 0.1) # Move
            if random.random() < 0.05 : self._rewards[current_agent] += 1 # Recon
            if random.random() < 0.01 : self._rewards[current_agent] += 5 # Challenge
            obs_data = self._generate_dummy_obs(current_agent)
            if obs_data["scout"] == 1 and random.random() < 0.005: # Scout captured
                self._rewards[current_agent] -= 50
                self._terminations[current_agent] = True
                for ag_idx, ag in enumerate(self.possible_agents):
                    if self._generate_dummy_obs(ag)["scout"] == 0: self._rewards[ag] += 50 # Guards get reward

        # Logic to advance global step: if current_agent is the last in the original possible_agents list
        # This is a simplification. Real PZ envs handle this more robustly.
        if self.possible_agents.index(current_agent) == len(self.possible_agents) -1:
            self.current_step_count += 1

        if self.current_step_count >= MAX_STEPS_PER_EPISODE:
            for ag in self.agents: self._truncations[ag] = True # Truncate all agents

        # Select next agent
        try:
            self.agent_selection = next(self._agent_selector_iter)
        except StopIteration: # Finished one pass over all currently active agents
            self._agent_selector_iter = iter(self.agents) # Reset iterator with current list of active agents
            self.agent_selection = next(self._agent_selector_iter, None) # Get first agent for next pass, or None if no agents

    def agent_iter(self):
        # Yields agents as long as there are active agents and steps remaining.
        self._agent_selector_iter = iter(self.agents) # Start with a fresh iterator over current agents
        self.agent_selection = next(self._agent_selector_iter, None)

        while self.agent_selection is not None:
            yield self.agent_selection
            # After yield, main loop calls last() and step(). step() updates agent_selection.
            if not self.agents or all(self._terminations[ag] or self._truncations[ag] for ag in self.agents):
                break # All agents are done

    def close(self): pass # print("DummyEnv closed.")

    # Static method to be callable like `gridworld.env()`
    @staticmethod
    def env(*args, **kwargs):
        return DummyGridworldEnv(*args, **kwargs)


if __name__ == '__main__':
    script_start_time = time.time() # Renamed to avoid conflict if 'start' is used elsewhere

    # --- User's Main Execution Block ---
    try:
        from til_environment import gridworld # Assuming this is your environment module
        print("Successfully imported til_environment.gridworld")
        current_env_module = gridworld
        
    except ImportError:
        print("Could not import 'til_environment.gridworld'.")
        print("Please ensure the environment module is correctly set up and accessible.")
        print("Attempting to use DummyGridworldEnv for basic testing (if available).")
        # Fallback to dummy environment if the main one is not found
        if 'DummyGridworldEnv' in globals() and gym is not None:
            print("Using DummyGridworldEnv as a fallback.")
            current_env_module = DummyGridworldEnv # Use the class itself as it has a static 'env' method
        else:
            print("DummyGridworldEnv is not available or gymnasium is not installed. Exiting.")
            current_env_module = None
            exit()
    
    if current_env_module:
        try:
            # Start training
            # Set load_model_from to a .pth file to continue training or fine-tune
            # Set save_model_to to where you want the final model to be saved
            trained_scores = train_agent(
                current_env_module, 
                num_episodes=100000, # Adjust as needed (e.g., 500 for quick dummy test)
                novice_track=False, # Or True for the Novice track map
                load_model_from="rb_agent_100k_eps.pth", # "trained_dqn_agent.pth" or your DuelingDQN model
                save_model_to="rb_agent_100k_v2_eps.pth" # Output for the new DuelingDQN model
            )
            print("Training finished.")

            # You can add plotting for scores if you like:
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10,6))
            # plt.plot(np.arange(1, len(trained_scores)+1), trained_scores, label='Episode Score')
            # if len(trained_scores) >= 100:
            #   moving_avg = [np.mean(trained_scores[max(0, i-99):i+1]) for i in range(len(trained_scores))]
            #   plt.plot(np.arange(1, len(trained_scores)+1), moving_avg, label='100-episode Avg', color='red', linestyle='--')
            # plt.ylabel('Score')
            # plt.xlabel('Episode #')
            # plt.title('RL Agent Training Progress')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig("training_scores_plot_rainbow.png") # Save the plot
            # plt.show()

        except Exception as e:
            print(f"An error occurred during training: {e}")
            traceback.print_exc()

    script_end_time = time.time() # Renamed
    print(f"Total script time = {script_end_time - script_start_time:.4f} seconds")
