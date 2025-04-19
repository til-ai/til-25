# RL

## Environment Details

TODO!

### Observation Space

This environment uses a discrete observation space. Observations are composed as follows:

consist of 4 stacked frames of the following:

A 3×5 vision area centering on the agent, which can see 1 tile in each direction except forward, where they can see 3 tiles ahead. Each tile is essentially a `uint8` binary flag, where the last 2 bits denote tile information, while the other 6 bits have tile occupancy and wall information.

| value of last 2 bits | meaning            |
| -------------------- | ------------------ |
| 0                    | no vision          |
| 1                    | empty tile         |
| 2                    | recon (1 point)    |
| 3                    | mission (5 points) |

| power of 2 | value | meaning         |
| ---------- | ----- | --------------- |
| 2          | 4     | defender        |
| 3          | 8     | attacker        |
| 4          | 16    | has right wall  |
| 5          | 32    | has bottom wall |
| 6          | 64    | has left wall   |
| 7          | 128   | has top wall    |

There is also a direction value provided:

| value | meaning |
| ----- | ------- |
| 0     | right   |
| 1     | down    |
| 2     | left    |
| 3     | up      |

There is also a 0-1 value representing whether the agent is currently a defender or an attacker.

| value | meaning  |
| ----- | -------- |
| 0     | attacker |
| 1     | defender |

### Action Space

This environment uses a discrete action space.

On your agent's turn, it should choose from one of the following 5 actions, and their corresponding integer:

| int | action   |
| --- | -------- |
| 0   | forward  |
| 1   | backward |
| 2   | left     |
| 3   | right    |
| 4   | stay     |

The `forward` and `backward` actions move one grid square in the direction the agent is facing (or in the opposite direction, in the case of `backward`). The `left` and `right` actions rotate the agent 90 degrees to each direction. The `stay` action results in the agent staying in place.

## Notes

The container containing your trained RL agent is likely to be independent of the environment, and thus can have a more limited set of requirements/packages installed solely for inference.
