# RL

## Environment Details

TODO!

### Observation Space

This environment uses a discrete observation space. Observations are composed as follows:

consist of 4 stacked frames of the following:

A 3×5 vision area centering on the agent, which can see 1 tile in each direction except forward, where they can see 3 tiles ahead. Each tile is essentially a `uint8` binary flag, where the last 2 bits denote tile information, while the other 6 bits have tile occupancy and wall information.

| Value of last 2 bits | Meaning            |
| -------------------- | ------------------ |
| 0                    | No vision          |
| 1                    | Empty tile         |
| 2                    | Recon (1 point)    |
| 3                    | Mission (5 points) |

| Power of 2 | Value | Meaning         |
| ---------- | ----- | --------------- |
| 2          | 4     | Scout           |
| 3          | 8     | Guard           |
| 4          | 16    | Has right wall  |
| 5          | 32    | Has bottom wall |
| 6          | 64    | Has left wall   |
| 7          | 128   | Hhas top wall   |

There is also a direction value provided:

| value | meaning |
| ----- | ------- |
| 0     | right   |
| 1     | down    |
| 2     | left    |
| 3     | up      |

There is also a 0-1 value representing whether the agent is currently a scout or a guard. This won't change within a round.

| value | meaning |
| ----- | ------- |
| 0     | scout   |
| 1     | guard   |

### Action Space

This environment uses a discrete action space.

On your agent's turn, it should choose from one of the following 5 actions, and their corresponding integer:

| Value (int) | Action        |
| ----------- | ------------- |
| 0           | Move forward  |
| 1           | Move backward |
| 2           | Turn left     |
| 3           | Turn right    |
| 4           | Stay          |

The `forward` and `backward` actions move one grid square in the direction the agent is facing (or in the opposite direction, in the case of `backward`). The `left` and `right` actions rotate the agent 90 degrees to each direction. The `stay` action results in the agent staying in place.

### Rewards

Note that, for the purpose of your training loop, you are able to modify the reward function however you wish to shape the training of your agent(s). However, for the Qualifier evaluations, these are the rewards that will be used.

| Outcome                          | Scout Reward | Guard Reward |
| -------------------------------- | ------------ | ------------ |
| Scout acquires a recon point     | 1            | 0            |
| Scout acquires a special mission | 5            | 0            |
| Scout captured by Guard          | -50          | 50           |

For the Semi-Finals or Grand Finals, the scoring will be similar, but the exact amount of points received by your team from special missions will be dependent on your other models' performance on the special mission.

## Notes

The container containing your trained RL agent is likely to be independent of the environment, and thus can have a more limited set of requirements/packages installed solely for inference.
