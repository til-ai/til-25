# RL

Your RL challenge is to command your agent through the game map while interacting with other agents and completing challenges.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/rl` route on port `5004`.

The input is a JSON object with this structure:

<!-- TODO: update this and delete this comment. -->

```JSON
{
  "instances": [
    {
      "observation": {
        "ryan": 0,
        "todo": "this"
      }
    }
  ]
}
```

The observation is a representation of the inputs the agent senses in its environment. See the [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications) in the Wiki for the meaning of each key.

The length of the `"instances"` array is 1.

## Output

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

The action is an integer representing the next movement your agent intends to take. See the [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications) in the Wiki for the possible values.
