from fastapi import FastAPI, Request

from RLManager import RLManager

app = FastAPI()

rl_manager = RLManager()


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/rl")
async def rl(request: Request):
    """
    Feeds observation into RL model
    Returns action taken given current observation (int)
    """

    # get observation, feed into model
    input_json = await request.json()

    predictions = []
    # each is a dict with one key "observation" and the value as a list of ints
    for instance in input_json["instances"]:
        predictions.append({"action": rl_manager.get_action(instance["observation"])})
    return {"predictions": predictions}
