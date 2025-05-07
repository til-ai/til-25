# OCR

Your OCR challenge is to read text in a scanned document.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/ocr` route on port `5003`.

The input is a JSON object with this structure:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    }
  ]
}
```

The `"b64"` key of each object in `"instances"` contains a base-64 encoded image in JPG format, at a size of 2481 by 3544 pixels (width by height), with single-channel (grayscale) 8-bit color depth. The image is a scan of a document with text.

The length of the `"instances"` array is indeterminate.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        "Predicted transcript one.",
        "Predicted transcript two."
    ]
}
```

Each string in `"predictions"` is the OCR prediction for the corresponding image.

The order of predictions in `"predictions"` must match the order in the `"instances"` object in the input JSON. There must be one prediction for every input.
