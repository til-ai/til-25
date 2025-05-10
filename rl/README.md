# RL

Your RL challenge is to direct your agent through the game map while interacting with other agents and completing challenges.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/rl` route on port `5004`. It is a JSON object structured as such:

```JSON
{
  "instances": [
    {
      "observation": {
        "viewcone": [[0, 0, ..., 0], [0, 0, ..., 0], ... , [0, 0, ..., 0]],
        "direction": 0,
        "location": [0, 0],
        "scout": 0,
        "step": 0
      }
    }
  ]
}
```

The observation is a representation of the inputs the agent senses in its environment. See the [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications) to learn how to interpret the observation.

The length of the `instances` array is 1.

During evaluation for Qualifiers, a GET request will be sent to the `/reset` route to signal that a round has ended, all agents are being reset to their starting positions (possibly with new roles), and any persistent state information your code may have stored must be cleared.

### Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        {
            "action": 0
        }
    ]
}
```

The action is an integer representing the next movement your agent intends to take. See the [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications) for a list of possible movements.
