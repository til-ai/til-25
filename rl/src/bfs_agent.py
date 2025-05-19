import collections
# from til_environment import gridworld # Assuming this is how you import the environment
from til_environment import gridworld
# --- END MOCK ---

# --- Constants based on the game specification ---
MAP_WIDTH = 16
MAP_HEIGHT = 16
ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_STAY = 4
DIR_RIGHT = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_UP = 3
TILE_TYPE_MASK = 0b11
TILE_NO_VISION = 0
TILE_EMPTY = 1
TILE_RECON = 2
TILE_MISSION = 3
OCC_SCOUT_BIT = 2
OCC_GUARD_BIT = 3
WALL_RIGHT_BIT = 4
WALL_BOTTOM_BIT = 5
WALL_LEFT_BIT = 6
WALL_TOP_BIT = 7
VIEWCONE_DEPTH = 7
VIEWCONE_WIDTH = 5
VIEWCONE_AGENT_RELATIVE_DEPTH = 2 
VIEWCONE_AGENT_RELATIVE_WIDTH = 2

class RuleBasedExploringAgent:
    def __init__(self, agent_id="agent"):
        self.agent_id = agent_id 
        self.map_size = MAP_WIDTH
        self.visited_map = [[False for _ in range(self.map_size)] for _ in range(self.map_size)]
        self.known_terrain = [[{'N_wall': None, 'E_wall': None, 'S_wall': None, 'W_wall': None} 
                               for _ in range(self.map_size)] for _ in range(self.map_size)]
        self.current_path_to_target = []
        self.last_action = None
        self.expected_next_pos = None
        # print(f"[{self.agent_id}] RuleBasedExploringAgent initialized.")

    def _transform_viewcone_to_absolute(self, r_vc, c_vc, agent_abs_x, agent_abs_y, agent_abs_dir):
        df = r_vc - VIEWCONE_AGENT_RELATIVE_DEPTH 
        dr = c_vc - VIEWCONE_AGENT_RELATIVE_WIDTH 
        abs_x, abs_y = agent_abs_x, agent_abs_y

        if agent_abs_dir == DIR_RIGHT: abs_x += df; abs_y -= dr
        elif agent_abs_dir == DIR_DOWN: abs_x += dr; abs_y += df
        elif agent_abs_dir == DIR_LEFT: abs_x -= df; abs_y += dr
        elif agent_abs_dir == DIR_UP: abs_x -= dr; abs_y -= df
        return abs_x, abs_y

    def _update_known_terrain(self, agent_x, agent_y, agent_dir, viewcone_flat):
        if not isinstance(viewcone_flat, list) or len(viewcone_flat) != VIEWCONE_DEPTH * VIEWCONE_WIDTH:
            return

        viewcone_grid = [viewcone_flat[i*VIEWCONE_WIDTH : (i+1)*VIEWCONE_WIDTH] 
                         for i in range(VIEWCONE_DEPTH)]

        for r_vc in range(VIEWCONE_DEPTH):
            for c_vc in range(VIEWCONE_WIDTH):
                tile_val = viewcone_grid[r_vc][c_vc]
                abs_x, abs_y = self._transform_viewcone_to_absolute(r_vc, c_vc, agent_x, agent_y, agent_dir)

                if not (0 <= abs_x < self.map_size and 0 <= abs_y < self.map_size):
                    continue
                
                current_cell_terrain = self.known_terrain[abs_x][abs_y]
                if current_cell_terrain['N_wall'] is None:
                    current_cell_terrain['N_wall'] = bool(tile_val & (1 << WALL_TOP_BIT))
                if current_cell_terrain['E_wall'] is None:
                    current_cell_terrain['E_wall'] = bool(tile_val & (1 << WALL_RIGHT_BIT))
                if current_cell_terrain['S_wall'] is None:
                    current_cell_terrain['S_wall'] = bool(tile_val & (1 << WALL_BOTTOM_BIT))
                if current_cell_terrain['W_wall'] is None:
                    current_cell_terrain['W_wall'] = bool(tile_val & (1 << WALL_LEFT_BIT))

                if current_cell_terrain['N_wall'] and abs_y > 0 and self.known_terrain[abs_x][abs_y-1]['S_wall'] is None:
                    self.known_terrain[abs_x][abs_y-1]['S_wall'] = True
                if current_cell_terrain['E_wall'] and abs_x < self.map_size - 1 and self.known_terrain[abs_x+1][abs_y]['W_wall'] is None:
                    self.known_terrain[abs_x+1][abs_y]['W_wall'] = True
                if current_cell_terrain['S_wall'] and abs_y < self.map_size - 1 and self.known_terrain[abs_x][abs_y+1]['N_wall'] is None:
                    self.known_terrain[abs_x][abs_y+1]['N_wall'] = True
                if current_cell_terrain['W_wall'] and abs_x > 0 and self.known_terrain[abs_x-1][abs_y]['E_wall'] is None:
                    self.known_terrain[abs_x-1][abs_y]['E_wall'] = True

    def _is_wall_between(self, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) + abs(dy) != 1: return True 

        if dx == 1: 
            return self.known_terrain[x1][y1]['E_wall'] is True or \
                   (0 <= x2 < self.map_size and self.known_terrain[x2][y1]['W_wall'] is True)
        if dx == -1: 
            return self.known_terrain[x1][y1]['W_wall'] is True or \
                   (0 <= x2 < self.map_size and self.known_terrain[x2][y1]['E_wall'] is True)
        if dy == 1: 
            return self.known_terrain[x1][y1]['S_wall'] is True or \
                   (0 <= y2 < self.map_size and self.known_terrain[x1][y2]['N_wall'] is True)
        if dy == -1: 
            return self.known_terrain[x1][y1]['N_wall'] is True or \
                   (0 <= y2 < self.map_size and self.known_terrain[x1][y2]['S_wall'] is True)
        return True 

    def _bfs_to_unvisited(self, start_x, start_y):
        q = collections.deque([((start_x, start_y), [])]) 
        bfs_visited_coords = {(start_x, start_y)}

        while q:
            (curr_x, curr_y), path = q.popleft()

            if not self.visited_map[curr_x][curr_y] and (curr_x, curr_y) != (start_x, start_y):
                return path + [(curr_x, curr_y)]

            for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]: 
                next_x, next_y = curr_x + dx, curr_y + dy

                if (0 <= next_x < self.map_size and 
                    0 <= next_y < self.map_size and
                    (next_x, next_y) not in bfs_visited_coords):
                    
                    wall_blocks = False
                    if dx == 1 and self.known_terrain[curr_x][curr_y]['E_wall'] is True: wall_blocks = True
                    elif dx == -1 and self.known_terrain[curr_x][curr_y]['W_wall'] is True: wall_blocks = True
                    elif dy == 1 and self.known_terrain[curr_x][curr_y]['S_wall'] is True: wall_blocks = True
                    elif dy == -1 and self.known_terrain[curr_x][curr_y]['N_wall'] is True: wall_blocks = True
                    
                    if not wall_blocks:
                        bfs_visited_coords.add((next_x, next_y))
                        q.append(((next_x, next_y), path + [(curr_x, curr_y)]))
        return None

    def _get_action_to_reach_adjacent(self, curr_x, curr_y, curr_dir, target_x, target_y):
        dx = target_x - curr_x
        dy = target_y - curr_y

        if abs(dx) + abs(dy) != 1: return ACTION_STAY 

        target_dir = -1
        if dx == 1: target_dir = DIR_RIGHT
        elif dx == -1: target_dir = DIR_LEFT
        elif dy == 1: target_dir = DIR_DOWN
        elif dy == -1: target_dir = DIR_UP

        if curr_dir == target_dir:
            if self._is_wall_between(curr_x, curr_y, target_x, target_y):
                self.current_path_to_target = [] 
                return ACTION_TURN_LEFT 
            return ACTION_FORWARD
        
        if (curr_dir - 1 + 4) % 4 == target_dir: return ACTION_TURN_LEFT
        if (curr_dir + 1) % 4 == target_dir: return ACTION_TURN_RIGHT
        return ACTION_TURN_LEFT


    def _get_expected_next_pos(self, current_pos_tuple, current_dir, action):
        # Ensure current_pos is a tuple for calculations
        x, y = current_pos_tuple 
        if action == ACTION_FORWARD:
            if current_dir == DIR_RIGHT: return (x + 1, y)
            if current_dir == DIR_LEFT:  return (x - 1, y)
            if current_dir == DIR_DOWN:  return (x, y + 1)
            if current_dir == DIR_UP:    return (x, y - 1)
        elif action == ACTION_BACKWARD: 
            if current_dir == DIR_RIGHT: return (x - 1, y)
            if current_dir == DIR_LEFT:  return (x + 1, y)
            if current_dir == DIR_DOWN:  return (x, y - 1)
            if current_dir == DIR_UP:    return (x, y + 1)
        return current_pos_tuple # Return tuple if no move

    def act(self, observation):
        # *** FIX FOR ValueError: Ensure my_pos is a tuple ***
        my_pos = tuple(observation['location']) 
        my_dir = observation['direction']
        viewcone_data = observation['viewcone']
        
        if self.last_action in [ACTION_FORWARD, ACTION_BACKWARD] and \
           self.expected_next_pos is not None and \
           self.expected_next_pos == my_pos and \
           self.current_path_to_target and \
           my_pos == self.current_path_to_target[0]: # Check if we specifically reached the target cell
            self.current_path_to_target.pop(0)
        elif self.last_action in [ACTION_FORWARD, ACTION_BACKWARD] and \
             self.expected_next_pos is not None and self.expected_next_pos != my_pos:
            self.current_path_to_target = [] 

        if 0 <= my_pos[0] < self.map_size and 0 <= my_pos[1] < self.map_size:
             self.visited_map[my_pos[0]][my_pos[1]] = True
        self._update_known_terrain(my_pos[0], my_pos[1], my_dir, viewcone_data)

        if not self.current_path_to_target:
            path = self._bfs_to_unvisited(my_pos[0], my_pos[1])
            if path and len(path) > 1: 
                self.current_path_to_target = path[1:] 
            else:
                self.current_path_to_target = []

        action_to_take = ACTION_STAY 
        self.expected_next_pos = my_pos # Default if no move action, ensure it's a tuple

        if self.current_path_to_target:
            next_cell_in_path = self.current_path_to_target[0]
            action_to_take = self._get_action_to_reach_adjacent(my_pos[0], my_pos[1], my_dir, 
                                                                next_cell_in_path[0], next_cell_in_path[1])
            
            # Pass my_pos (which is now guaranteed to be a tuple)
            potential_next_pos_after_action = self._get_expected_next_pos(my_pos, my_dir, action_to_take)
            
            # The comparison potential_next_pos_after_action != my_pos should now work correctly
            if action_to_take in [ACTION_FORWARD, ACTION_BACKWARD] and potential_next_pos_after_action != my_pos:
                 if self._is_wall_between(my_pos[0], my_pos[1], potential_next_pos_after_action[0], potential_next_pos_after_action[1]):
                    self.current_path_to_target = [] 
                    action_to_take = ACTION_TURN_LEFT 
                    self.expected_next_pos = my_pos # Reset expected_next_pos as we are turning
                 else:
                    self.expected_next_pos = potential_next_pos_after_action
            elif action_to_take in [ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_STAY]:
                self.expected_next_pos = my_pos # If turning or staying, expected position is current position
        else: 
            action_to_take = ACTION_TURN_LEFT
            self.expected_next_pos = my_pos

        self.last_action = action_to_take
        return action_to_take

# --- Main Execution Logic ---
if __name__ == "__main__":
    import os
    import imageio
    
    env = gridworld.env(
        env_wrappers=[],
        render_mode="rgb_array", 
        debug=False, 
        novice=False, 
    )
    
    env.reset(seed=42)
    agent_policies = {} # Store policy for the scout agent(s)
    
    # Create video folder if needed
    video_folder = "./bfs_rl_renders"
    os.makedirs(video_folder, exist_ok=True)
    episode_frames = []

    # The game ends after 100 time steps or scout capture.
    # The PettingZoo loop handles iteration until termination/truncation.
    try:
        for agent_id in env.agent_iter(): # agent_id can be 'player_0', 'player_1', etc.
            observation, reward, termination, truncation, info = env.last()
            
            # Capture frame ONLY if recording is enabled for this episode
            if True:
                try:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                except Exception as e:
                    print(f"\nWarning: Could not render frames")
                    traceback.print_exc()

            if termination or truncation:
                # print(f"Game step {observation.get('step', 'N/A')}: Agent {agent_id} terminated or truncated. Reward: {reward}. Ending episode.")
                # Action for terminated agent might be needed if env expects it
                # env.step(None) or env.action_space(agent_id).sample() or ACTION_STAY
                # However, the loop usually breaks here.
                break 
            else:
                # *** MODIFICATION: Control scout only ***
                is_scout = observation.get('scout') == 1

                if is_scout:
                    if agent_id not in agent_policies:
                        print(f"Game step {observation.get('step', 'N/A')}: Creating RuleBasedExploringAgent for SCOUT: {agent_id}")
                        agent_policies[agent_id] = RuleBasedExploringAgent(agent_id=agent_id)
                    
                    current_policy = agent_policies[agent_id]
                    action = current_policy.act(observation)
                else: # It's a Guard
                    # print(f"Game step {observation.get('step', 'N/A')}: Agent {agent_id} is a Guard. Action: STAY")
                    action = ACTION_STAY # Guards will do nothing
            
            env.step(action)
            
        # Save video at the end of the episode ONLY if frames were collected
        if video_folder and len(episode_frames) > 0: # Check the flag
            video_path = os.path.join(video_folder, f"episode_1.mp4") # Use 6 digits for episode number
            try:
                # imageio needs the frames to be in (T, H, W, C) format, which env.render() provides
                imageio.mimsave(video_path, episode_frames, fps=30) # Adjust fps as needed
                print(f"\nSaved video") # Print on a new line after progress
            except Exception as e:
                print(f"\nWarning: Could not save video to {video_path}: {e}")
                traceback.print_exc()
        else:
            print(f"num frames: {len(episode_frames)}")

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        env.close() # This will print "MockEnv closed." if using mock
        print("Main script finished.")


