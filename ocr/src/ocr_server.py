"""Runs the OCR server."""
import base64
from fastapi import FastAPI, Request
# Assuming ocr_manager.py is in the same directory (src/)
from .ocr_manager import OCRManager # Use relative import if ocr_manager is in the same package/dir

app = FastAPI()
# Initialize the manager when the application starts.
# This will load PaddleOCR models into memory.
manager = OCRManager()


@app.post("/ocr")
async def ocr(request: Request) -> dict[str, list[str]]:
    """Performs OCR on images of documents.

    Args:
        request: The API request. Contains a list of images, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` document text, in the same order as which appears in `request`.
    """
    try:
        inputs_json = await request.json()
    except Exception as e:
        return {"error": f"Invalid JSON payload: {str(e)}"}

    if "instances" not in inputs_json or not isinstance(inputs_json["instances"], list):
        return {"error": "Missing or invalid 'instances' field in JSON payload."}

    predictions = []
    for instance in inputs_json["instances"]:
        if not isinstance(instance, dict) or "b64" not in instance:
            predictions.append("Error: Invalid instance format, missing 'b64' key.")
            continue
        
        try:
            # Reads the base-64 encoded image and decodes it into bytes.
            image_bytes = base64.b64decode(instance["b64"])
        except Exception as e:
            predictions.append(f"Error: Could not decode base64 string: {str(e)}")
            continue
        
        # Performs OCR and appends the result.
        text = manager.ocr(image_bytes)
        predictions.append(text)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    # You could add a simple OCR test here if manager is initialized
    if manager and hasattr(manager, 'ocr_engine') and manager.ocr_engine:
        return {"message": "health ok, OCRManager initialized"}
    else:
        return {"message": "health warning, OCRManager not fully initialized"}