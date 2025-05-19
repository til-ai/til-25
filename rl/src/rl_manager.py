import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# --- Configuration ---
# Neural Network Hyperparameters
INPUT_FEATURES = 288  # Calculated below: 7*5*8 (viewcone) + 4 (direction) + 2 (location) + 1 (scout) + 1 (step)
HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 256
HIDDEN_LAYER_3_SIZE = 256
OUTPUT_ACTIONS = 5  # 0:Forward, 1:Backward, 2:TurnL, 3:TurnR, 4:Stay

# Game Environment Constants
MAP_SIZE_X = 16
MAP_SIZE_Y = 16
MAX_STEPS = 100
MAX_STEPS_PER_EPISODE = 100 # Max steps in one round of the game

# Agent settings
EPSILON_INFERENCE = 0.01 # Small epsilon for some exploration even during inference, or 0 for pure exploitation

# --- Deep Q-Network (DQN) Model (same as in rl_agent_python_v1) ---
# Ensure this class is defined and available
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# --- RL Agent (Inference Only) ---
class RLManager:
    """
    The Reinforcement Learning Agent for inference using a pre-trained model.
    It processes observations and uses a DQN to select actions.
    """
    def __init__(self, model_path="agent03_100k_eps.pth"):
        """
        Initialises the RL Agent.
        Args:
            model_path (str): Path to a pre-trained model file (.pth).
        """
        # Ensure DEVICE is defined (e.g., globally or passed in)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Assuming these constants are defined globally or imported
        # INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS
        try:
             self.model = DQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS).to(self.device)
        except NameError as e:
             print(f"Error: Required constant not defined: {e}")
             print("Please ensure INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS are defined.")
             raise

        if model_path and os.path.exists(model_path):
            try:
                # Load the state_dict onto the correct device
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}. Initialising with random weights.")
                # Fallback to random weights if loading fails or model mismatch
                self.model.apply(self._initialise_weights)
        else:
            print(f"No model path provided or path {model_path} does not exist. Initialising model with random weights.")
            # Initialise with random weights if no path is given or file not found
            self.model.apply(self._initialise_weights)

        self.model.eval()  # Set the model to evaluation mode (disables dropout, batch norm stats etc.)

    def _initialise_weights(self, m):
        """
        Initialises weights of the neural network layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _unpack_viewcone_tile(self, tile_value):
        """
        Unpacks a single tile's 8-bit integer value from the viewcone into a feature vector.
        Returns a list of 8 binary features.
        """
        tile_features = []
        tile_features.append(float(tile_value & 0b01)) # bit 0 (Tile Type Bit 0)
        tile_features.append(float((tile_value & 0b10) >> 1)) # bit 1 (Tile Type Bit 1)
        for i in range(2, 8): # bits 2 through 7 (Scout, Guard, Walls)
            tile_features.append(float((tile_value >> i) & 1))

        return tile_features # This will be 8 features per tile

    def process_observation(self, observation_dict):
        """
        Processes the observation dictionary into a flat feature vector (numpy array).
        Matches the feature engineering used during training.
        """
        processed_features = []

        # Process viewcone (7x5 grid, 8 features per tile)
        viewcone = observation_dict.get("viewcone", [[0]*5 for _ in range(7)]) # Default to empty if missing
        for r in range(7):
            for c in range(5):
                # Ensure we don't go out of bounds if viewcone dimensions vary unexpectedly
                tile_value = viewcone[r][c] if r < len(viewcone) and c < len(viewcone[r]) else 0
                processed_features.extend(self._unpack_viewcone_tile(tile_value))

        # Process direction (one-hot encoding)
        direction = observation_dict.get("direction", 0)
        direction_one_hot = [0.0] * 4
        if 0 <= direction < 4:
             direction_one_hot[direction] = 1.0
        processed_features.extend(direction_one_hot)

        # Process location (normalised)
        location = observation_dict.get("location", [0, 0])
        # Assuming MAP_SIZE_X and MAP_SIZE_Y are defined
        norm_x = location[0] / MAP_SIZE_X if MAP_SIZE_X > 0 else 0.0
        norm_y = location[1] / MAP_SIZE_Y if MAP_SIZE_Y > 0 else 0.0
        processed_features.extend([norm_x, norm_y])

        # Process scout role
        scout_role = float(observation_dict.get("scout", 0))
        processed_features.append(scout_role)

        # Process step (normalised)
        step = observation_dict.get("step", 0)
        # Assuming MAX_STEPS_PER_EPISODE is defined
        norm_step = step / MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else 0.0
        processed_features.append(norm_step)

        # Ensure correct feature length
        # Assuming INPUT_FEATURES is defined and correct
        if len(processed_features) != INPUT_FEATURES:
             print(f"Warning: Feature length mismatch. Expected {INPUT_FEATURES}, got {len(processed_features)}")
             # Depending on strictness, you might raise an error here
             # raise ValueError(f"Feature length mismatch. Expected {INPUT_FEATURES}, got {len(processed_features)}")
             # For now, return what we have, but this indicates a problem in feature engineering or constant definition

        return np.array(processed_features, dtype=np.float32) # Return as numpy array

    def rl(self, observation_dict):
        """Selects an action based on the current observation using the loaded DQN model.
        Uses epsilon-greedy for exploration during inference/testing if EPSILON_INFERENCE > 0.
        For pure greedy inference, set EPSILON_INFERENCE = 0.

        Args:
            observation_dict (dict): The observation dictionary provided by the environment.
        Returns:
            int: The selected action (0 to OUTPUT_ACTIONS-1).
                 See environment documentation for action mapping.
        """
        # Assuming EPSILON_INFERENCE is defined
        if random.random() < EPSILON_INFERENCE:
            # Explore: select a random action
            return random.randint(0, OUTPUT_ACTIONS - 1)
        else:
            # Exploit: select the best action based on Q-values from the model
            state_np = self.process_observation(observation_dict)
            # Convert numpy array to torch tensor and move to the appropriate device
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)

            with torch.no_grad(): # Important: disable gradient calculation for inference
                # Get Q values from the model
                q_values = self.model(state_tensor)
                # Select the action with the maximum Q-value
                action = torch.argmax(q_values, dim=1).item()

            return action

    def reset_state(self):
        """
        Resets any internal state if the agent was stateful (e.g., for RNNs).
        Not strictly necessary for this feedforward DQN, but included for compatibility.
        """
        pass # No state to reset for this simple DQN


# Example Usage (requires constants and a dummy observation):
if __name__ == "__main__":
    # Define necessary constants for demonstration
    INPUT_FEATURES = 7 * 5 * 8 + 4 + 2 + 1 + 1 # 7x5 grid, 8 bits per tile + direction (4) + location (2) + scout (1) + step (1)
    HIDDEN_LAYER_1_SIZE = 256 # Example size
    HIDDEN_LAYER_2_SIZE = 128 # Example size
    HIDDEN_LAYER_3_SIZE = 64  # Example size
    OUTPUT_ACTIONS = 5      # Example: e.g., [Move N, Move E, Move S, Move W, Stay]
    MAP_SIZE_X = 10         # Example map size
    MAP_SIZE_Y = 10         # Example map size
    MAX_STEPS_PER_EPISODE = 100 # Example max steps
    EPSILON_INFERENCE = 0.0   # Set to 0 for pure greedy inference

    # Create a dummy model file for testing load
    # In a real scenario, this would be a model trained by TrainableRLAgent
    dummy_model_path = "dummy_agent_model.pth"
    if not os.path.exists(dummy_model_path):
        print(f"Creating a dummy model file at {dummy_model_path}...")
        dummy_model_train = DQN(INPUT_FEATURES, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE, OUTPUT_ACTIONS)
        torch.save(dummy_model_train.state_dict(), dummy_model_path)
        print("Dummy model created.")

    # Initialise the RLManager, attempting to load the dummy model
    agent = RLManager(model_path=dummy_model_path)

    # Create a dummy observation dictionary
    # This must match the structure expected by process_observation
    dummy_observation = {
        "viewcone": [[random.randint(0, 255) for _ in range(5)] for _ in range(7)],
        "direction": random.randint(0, 3),
        "location": [random.randint(0, MAP_SIZE_X - 1), random.randint(0, MAP_SIZE_Y - 1)],
        "scout": random.choice([0, 1]),
        "step": random.randint(0, MAX_STEPS_PER_EPISODE - 1)
    }

    # Perform inference
    print("\nPerforming inference with a dummy observation:")
    selected_action = agent.rl(dummy_observation)
    print(f"Observation: {dummy_observation}")
    print(f"Selected Action: {selected_action}")

    # Example with a non-existent path to show fallback
    print("\nInitialising with a non-existent path (will use random weights):")
    agent_random = RLManager(model_path="non_existent_model.pth")
    selected_action_random = agent_random.rl(dummy_observation)
    print(f"Selected Action (random init): {selected_action_random}")

    # Clean up dummy file
    # if os.path.exists(dummy_model_path):
    #     os.remove(dummy_model_path)
    #     print(f"\nCleaned up dummy model file {dummy_model_path}")