from typing import Dict, List


class CVManager:
    def __init__(self):
        """
        Initializes the CVManager. This is where you should
        initialize your model and any static configurations.
        """

    def identify(self, image: bytes) -> List[Dict[str, int | List[int]]]:
        """
        Performs object detection, returning a list of predictions in the format
        [
            {
                "bbox": [x, y, w, h],
                "category_id": category_id
            },
            ...
        ]
        where:
        - (x, y) are the coordinates of the top left corner of the predicted bounding box,
        - (w, h) are its width and height in pixels,
        - category_id is the index of the predicted category of the enclosed object.

        Note that all bounding box coordinates are 0-indexed. That is, if (x, y) = (0, 0),
        then the top left corner of your bounding box is the top left corner of the image.
        """
        return []
