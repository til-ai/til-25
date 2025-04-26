import base64
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from scoring.vlm_eval import vlm_eval

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")


def main():
    input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")

    results_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    truths = []
    counter = 0

    with open(input_dir / "vlm.jsonl", "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "images" / instance["image"], "rb") as file:
                image_bytes = file.read()
                for annotation in instance["annotations"]:
                    instances.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "b64": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    )
                    truths.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "bbox": annotation["bbox"],
                        }
                    )
                    counter += 1

    assert len(truths) == len(instances)
    results = run_batched(instances)
    df = pd.DataFrame(results)
    assert len(truths) == len(results)
    df.to_csv(results_dir / "vlm_results.csv", index=False)
    # calculate eval
    eval_result = vlm_eval(
        [truth["bbox"] for truth in truths],
        [result["bbox"] for result in results],
    )
    print(f"IoU@0.5: {eval_result}")


def run_batched(
    instances: list[dict[str, str | int]], batch_size: int = 4
) -> list[dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        response = requests.post(
            "http://localhost:5004/identify",
            data=json.dumps(
                {
                    "instances": [
                        {field: _instance[field] for field in ("key", "caption", "b64")}
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
                    "bbox": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
