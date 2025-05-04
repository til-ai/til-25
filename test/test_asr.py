import base64
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from .score_asr import asr_eval

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")


def main():
    input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)
    instances = []

    with open(input_dir / "asr.jsonl", "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "audio" / instance["audio"], "rb") as file:
                audio_bytes = file.read()
                instances.append(
                    {**instance, "b64": base64.b64encode(audio_bytes).decode("ascii")}
                )

    results = run_batched(instances)
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "asr_results.csv", index=False)
    # calculate eval
    eval_result = asr_eval(
        [result["transcript"] for result in results],
        [result["prediction"] for result in results],
    )
    print(f"1-WER: {eval_result}")


def run_batched(
    instances: list[dict[str, str | int]], batch_size: int = 4
) -> list[dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        response = requests.post(
            "http://localhost:5001/asr",
            data=json.dumps(
                {
                    "instances": [
                        {"key": _instance["key"], "b64": _instance["b64"]}
                        for _instance in _instances
                    ]
                }
            ),
        )
        _results = response.json()["predictions"]
        results.extend(
            [
                {
                    "key": _instances[i]["key"],
                    "transcript": _instances[i]["transcript"],
                    "prediction": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
