import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
import jiwer
from typing import Any
import itertools
import requests
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 4


wer_transforms = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({"-": " "}),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


def sample_generator(
        instances: Sequence[Mapping[str, Any]],
        data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / instance["audio"], "rb") as audio_file:
            audio_bytes = audio_file.read()
        yield {
            "key": instance["key"],
            "b64": base64.b64encode(audio_bytes).decode("ascii"),
        }


def score_asr(truth: list[str], hypothesis: list[str]) -> float:
    return 1 - jiwer.wer(
        truth,
        hypothesis,
        truth_transform=wer_transforms,
        hypothesis_transform=wer_transforms,
    )


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/asr")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "asr.jsonl") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]

    batch_generator = itertools.batched(sample_generator(instances, data_dir), n=BATCH_SIZE)
    
    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        response = requests.post("http://localhost:5001/asr", data=json.dumps({
            "instances": batch,
        }))
        results.extend(response.json()["predictions"])

    results_path = results_dir / "asr_results.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)

    ground_truths = [instance["transcript"] for instance in instances]
    score = score_asr(results, ground_truths)
    print("1 - WER:", score)


if __name__ == "__main__":
    main()
