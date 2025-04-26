import base64

from CVManager import CVManager
from fastapi import FastAPI, Request

app = FastAPI()

cv_manager = CVManager()


@app.get("/health")
def health():
    """
    Healthcheck function for your model.
    """
    return {"message": "health ok"}


@app.post("/identify")
async def identify(instance: Request):
    """
    Performs Object Detection given an image frame.
    """
    # get base64 encoded string of image, convert back into bytes
    input_json = await instance.json()

    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        image_bytes = base64.b64decode(instance["b64"])

        bboxes = cv_manager.identify(image_bytes)
        predictions.append(bboxes)

    return {"predictions": predictions}
