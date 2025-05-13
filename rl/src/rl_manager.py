import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flask import Flask, request, jsonify
import random

# --- Configuration ---
# Neural Network Hyperparameters
INPUT_FEATURES = 288  # Calculated below: 7*5*8 (viewcone) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 128
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay

# Game Environment Constants
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS = 100

# Agent settings
EPSILON_INFERENCE = 0.01 # Small epsilon for some exploration even during inference, or 0 for pure exploitation

# --- Deep Q-Network (DQN) Model ---
class DQN(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for Q-value approximation.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): The input state features.
        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# --- RL Agent ---
class RLManager:
    """
    The Reinforcement Learning Agent.
    It processes observations, uses a DQN to select actions, and can be reset.
    """
    def __init__(self, model_path=None):
        """
        Initializes the RL Agent.
        Args:
            model_path (str, optional): Path to a pre-trained model file. Defaults to None (random initialization).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = DQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_ACTIONS).to(self.device)

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}. Using random weights.")
                # Fallback to random weights if loading fails
                self.model.apply(self._initialize_weights)
        else:
            print("No model path provided. Initializing model with random weights.")
            self.model.apply(self._initialize_weights)

        self.model.eval()  # Set the model to evaluation mode

    def _initialize_weights(self, m):
        """
        Initializes weights of the neural network layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        """
        Unpacks a single tile's 8-bit integer value from the viewcone into a feature vector.
        - Bits 0-1: Tile type (0:No vision, 1:Empty, 2:Recon, 3:Mission)
        - Bit 2: Scout present
        - Bit 3: Guard present
        - Bit 4: Right wall
        - Bit 5: Bottom wall
        - Bit 6: Left wall
        - Bit 7: Top wall

        Returns a list of 8 binary features.
        """
        features = []
        
        # Tile type (2 bits) - represented as one-hot encoding (4 features)
        tile_type = tile_value & 0b11
        # features.extend([1.0 if i == tile_type else 0.0 for i in range(4)]) # One-hot for tile type
        # For simplicity, let's use the 2 bits directly, or scale them.
        # Or, let's use the 8-bit decomposition as planned.
        
        # Bits 0-1: Tile type (0:No vision, 1:Empty, 2:Recon, 3:Mission)
        # We can one-hot encode this part.
        type_feature = [0.0] * 4
        if tile_type < 4: # Ensure tile_type is within expected range
            type_feature[tile_type] = 1.0
        features.extend(type_feature)


        # Bit 2: Scout present
        features.append(1.0 if (tile_value >> 2) & 1 else 0.0)
        # Bit 3: Guard present
        features.append(1.0 if (tile_value >> 3) & 1 else 0.0)
        # Bit 4: Right wall
        features.append(1.0 if (tile_value >> 4) & 1 else 0.0)
        # Bit 5: Bottom wall
        features.append(1.0 if (tile_value >> 5) & 1 else 0.0)
        # Bit 6: Left wall
        features.append(1.0 if (tile_value >> 6) & 1 else 0.0)
        # Bit 7: Top wall
        features.append(1.0 if (tile_value >> 7) & 1 else 0.0)
        
        # The above gives 4 (tile_type one-hot) + 6 (occupancy/walls) = 10 features per tile.
        # Let's stick to the original plan of 8 features per tile for INPUT_FEATURES = 288 calculation.
        # Re-doing the feature extraction for 8 features per tile:
        # Feature 1,2: Tile type (raw bits, or scaled)
        # For simplicity and directness with the 8-bit description:
        tile_features = []
        tile_features.append(float(tile_value & 0b01)) # bit 0
        tile_features.append(float((tile_value & 0b10) >> 1)) # bit 1
        for i in range(2, 8): # bits 2 through 7
            tile_features.append(float((tile_value >> i) & 1))
        
        return tile_features # This will be 8 features

    def process_observation(self, observation_dict):
        """
        Converts the raw observation dictionary into a flat feature tensor for the DQN.
        Args:
            observation_dict (dict): The observation dictionary from the game.
        Returns:
            torch.Tensor: A flat tensor representing the state.
        """
        processed_features = []

        # 1. Viewcone (7x5 grid)
        viewcone = observation_dict.get("viewcone", [])
        viewcone_features = []
        for r in range(7): # 7 rows
            for c in range(5): # 5 columns
                tile_value = viewcone[r][c] if r < len(viewcone) and c < len(viewcone[r]) else 0 # Handle potential malformed viewcone
                viewcone_features.extend(self._unpack_viewcone_tile(tile_value))
        processed_features.extend(viewcone_features) # 7 * 5 * 8 = 280 features

        # 2. Direction (0:R, 1:D, 2:L, 3:U) - One-hot encode
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4:
            direction_one_hot[direction] = 1.0
        processed_features.extend(direction_one_hot) # 4 features

        # 3. Location (x,y) - Normalize
        location = observation_dict.get("location", [0, 0])
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        processed_features.extend([norm_x, norm_y]) # 2 features

        # 4. Scout (1 if scout, 0 if guard)
        scout_role = float(observation_dict.get("scout", 0))
        processed_features.append(scout_role) # 1 feature

        # 5. Step - Normalize
        step = observation_dict.get("step", 0)
        norm_step = step / MAX_STEPS if MAX_STEPS > 0 else 0.0
        processed_features.append(norm_step) # 1 feature
        
        # Total features should match INPUT_FEATURES (280 + 4 + 2 + 1 + 1 = 288)
        if len(processed_features) != INPUT_FEATURES:
            print(f"Warning: Feature length mismatch. Expected {INPUT_FEATURES}, got {len(processed_features)}")
            # Pad or truncate if necessary, though ideally this shouldn't happen with correct processing
            if len(processed_features) < INPUT_FEATURES:
                processed_features.extend([0.0] * (INPUT_FEATURES - len(processed_features)))
            else:
                processed_features = processed_features[:INPUT_FEATURES]


        return torch.tensor(processed_features, dtype=torch.float32, device=self.device).unsqueeze(0) # Add batch dimension

    def select_action(self, observation_dict):
        """Selects an action based on the current observation.
        Uses epsilon-greedy for exploration if epsilon > 0, otherwise greedy.
        Args:
            observation_dict (dict): The observation dictionary.
        Returns:
            int: The selected action (0-4). See `rl/README.md` for the options.
        """
        if random.random() < EPSILON_INFERENCE:
            # Explore: select a random action
            return random.randint(0, OUTPUT_ACTIONS - 1)
        else:
            # Exploit: select the best action based on Q-values
            with torch.no_grad(): # No need to track gradients during inference
                state_tensor = self.process_observation(observation_dict)
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values, dim=1).item() # Get the action with the highest Q-value
                return action

    def reset(self):
        """
        Resets any persistent state information in the agent.
        For this stateless DQN inference agent, this might not do much.
        If using RNN layers or episodic memory, clear them here.
        """
        print("Agent state reset.")
        # Example: if self.model has recurrent layers, reset hidden states:
        # if hasattr(self.model, 'reset_hidden_states'):
        #     self.model.reset_hidden_states()
        pass



