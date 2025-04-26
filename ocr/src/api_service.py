import base64

from fastapi import FastAPI, Request
from OCRManager import OCRManager

app = FastAPI()

ocr_manager = OCRManager()


@app.get("/health")
def health():
    """
    Healthcheck function for your model.
    """
    return {"message": "health ok"}


@app.post("/ocr")
async def ocr(instance: Request):
    """
    Performs OCR on a given document image, returning the text found in the document.
    """
    # get base64 encoded string of image, convert back into bytes
    input_json = await instance.json()

    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        image_bytes = base64.b64decode(instance["b64"])

        bbox = ocr_manager.identify(image_bytes, instance["caption"])
        predictions.append(bbox)

    return {"predictions": predictions}
