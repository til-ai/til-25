# CV

Your CV challenge is to detect and classify objects in an image.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/cv` route on port 5002. It is a JSON document structured as such:

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

The `b64` key of each object in the `instances` list contains the base64-encoded bytes of the input image in JPEG format. The length of the `instances` list is variable.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        [
            {
                "bbox": [x, y, w, h],
                "category_id": category_id
            },
            ...
        ],
        ...
    ]
}
```

where `x`, `y`, `w`, `h`, and `category_id` are defined as above.

If your model detects no objects in a scene, your handler should output an empty list for that scene.

The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where n is the number of input instances. The length of `predictions` must equal that of `instances`.
