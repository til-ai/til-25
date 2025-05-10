import json
import os

import requests
from dotenv import load_dotenv
from til_environment import gridworld

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

NUM_ROUNDS = 8


def main(novice: bool):
    env = gridworld.env(env_wrappers=[], render_mode=None, novice=novice)
    # be the agent at index 0
    _agent = env.possible_agents[0]
    rewards = {agent: 0 for agent in env.possible_agents}

    for _ in range(NUM_ROUNDS):
        env.reset()
        # reset endpoint
        _ = requests.post("http://localhost:5004/reset")

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            observation = {
                k: v if type(v) is int else v.tolist() for k, v in observation.items()
            }

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                action = None
            elif agent == _agent:
                response = requests.post(
                    "http://localhost:5004/rl",
                    data=json.dumps({"instances": [{"observation": observation}]}),
                )
                predictions = response.json()["predictions"]

                action = int(predictions[0]["action"])
            else:
                # take random action from other agents
                action = env.action_space(agent).sample()
            env.step(action)
    env.close()
    print(f"total rewards: {rewards[_agent]}")
    print(f"score: {rewards[_agent] / NUM_ROUNDS / 100}")


if __name__ == "__main__":
    main(TEAM_TRACK == "novice")
