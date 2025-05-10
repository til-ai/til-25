
import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
import jiwer
from typing import Any
import requests
from dotenv import load_dotenv
import itertools
from tqdm import tqdm


load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 4


cer_transforms = jiwer.Compose([
    jiwer.SubstituteRegexes({"-": ""}),
    jiwer.RemoveWhiteSpace(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


def sample_generator(
        instances: Sequence[Mapping[str, Any]],
        data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / instance["document"], "rb") as file:
            document_bytes = file.read()
        yield {
            **instance,
            "b64": base64.b64encode(document_bytes).decode("ascii"),
        }


def score_ocr(preds: Sequence[str], ground_truth: Sequence[str]) -> float:
    return 1 - jiwer.cer(
        ground_truth,
        preds,
        truth_transform=cer_transforms,
        hypothesis_transform=cer_transforms,
    )


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/ocr")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "ocr.jsonl") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]

    batch_generator = itertools.batched(sample_generator(instances, data_dir), n=BATCH_SIZE)

    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        response = requests.post("http://localhost:5003/ocr", data=json.dumps({
            "instances": batch,
        }))
        results.extend(response.json()["predictions"])

    results_path = results_dir / "ocr_results.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)
    
    ground_truths = []
    for instance in instances:
        with open(data_dir / instance["contents"], "r") as file:
            document_contents = file.read().strip()
        ground_truths.append(document_contents)

    score = score_ocr(results, ground_truths)
    print("1 - CER:", score)


if __name__ == "__main__":
    main()
