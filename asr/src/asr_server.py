"""Runs the ASR server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64
from fastapi import FastAPI, Request
from .asr_manager import ASRManager


app = FastAPI()
manager = ASRManager()


@app.post("/asr")
async def asr(request: Request) -> dict[str, list[str]]:
    """Performs ASR on audio files.

    Args:
        request: The API request. Contains a list of audio files, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` transcriptions, in the same order as which appears in `request`.
    """

    inputs_json = await request.json()

    predictions = []
    for instance in inputs_json["instances"]:

        # Reads the base-64 encoded audio and decodes it into bytes.
        audio_bytes = base64.b64decode(instance["b64"])

        # Performs ASR and appends the result.
        transcription = manager.asr(audio_bytes)
        predictions.append(transcription)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the server."""
    return {"message": "health ok"}
