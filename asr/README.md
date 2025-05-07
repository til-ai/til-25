# ASR

Your ASR challenge is to transcribe a noisy recording of speech.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/asr` route on port `5001`.

The input is a JSON object with this structure:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_AUDIO"
    },
    ...
  ]
}
```

The `"b64"` key of each object in `"instances"` contains a base-64 encoded WAV audio file, single channel, sampled at 16 kHz. The audio is English speech to be transcribed.

The length of the `"instances"` array is indeterminate.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        "Predicted transcript.",
        ...
    ]
}
```

Each string in `"predictions"` is the ASR transcription for the corresponding audio file.

The order of predictions in `"predictions"` must match the order in the `"instances"` object in the input JSON. There must be one prediction for every input.
