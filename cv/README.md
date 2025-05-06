# CV

Your CV challenge is to detect and classify objects in an image.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/cv` route on port `5002`.

The input is a JSON object with this structure:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    },
    ...
  ]
}
```

<!-- TODO: Check if this is still correct, and delete this comment. -->

The `"b64"` key of each object in `"instances"` contains a base-64 encoded image in JPG format, at a size of 1520 by 870 pixels (width by height), with 3-channel 8-bit color depth. The image is a scene in which to perform object detection and classification.

The length of the `"instances"` array is indeterminate.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        {
            "bbox": [x, y, w, h],
            "category_id": category_id
        },
        ...
    ]
}
```

Where `x`, `y`, `w`, `h`, and `category_id` are of type `int`, and represent the coordinates, dimensions, and predicted category of the detected object. See the [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications) in the Wiki for the meaning of each parameter.

The order of predictions in `"predictions"` must be the same as the order of objects in the `"instances"` object in the input JSON. There must be one prediction for every input.
