import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from collections import defaultdict
from typing import Any
import requests
from dotenv import load_dotenv
import itertools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import io

# SAHI imports with updated approach
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")
BATCH_SIZE = 4

class COCOPatched(COCO):
    def __init__(self, annotations):
        # The varnames here are disgusting, but they're used by other
        # non-overridden methods so don't touch them.
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        assert type(annotations) == dict, \
            f"Annotation format {type(annotations)} not supported"
        print("Annotations loaded.")
        self.dataset = annotations
        self.createIndex()

def sample_generator(
        instances: Sequence[Mapping[str, Any]],
        data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / "images" / instance["file_name"], "rb") as img_file:
            img_data = img_file.read()
            yield {
                "key": instance["id"],
                "b64": base64.b64encode(img_data).decode("ascii"),
                "file_path": str(data_dir / "images" / instance["file_name"])
            }

def score_cv(preds: Sequence[Mapping[str, Any]], ground_truth: Any) -> float:
    if not preds:
        return 0.
    
    ground_truth = COCOPatched(ground_truth)
    results = ground_truth.loadRes(preds)
    coco_eval = COCOeval(ground_truth, results, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0].item()

def setup_detection_model():
    """Set up the detection model using SAHI's AutoDetectionModel"""
    # Path to model
    model_path = Path(os.getcwd())/'src'/"yolo11_v1.pt"
    
    print(f"Setting up detection model from {model_path}")
    
    # Use AutoDetectionModel for YOLO11 as recommended in the documentation
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=0.3,
        device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    return detection_model

def process_image(image_path, detection_model, temp_dir="temp_images"):
    """Process a single image using SAHI sliced inference"""
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get predictions using SAHI sliced inference
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        
        # Convert to COCO predictions format
        coco_predictions = result.to_coco_predictions(image_id=0)
        
        # Format the results according to COCO format
        detections = []
        for pred in coco_predictions:
            # COCO format bbox is [x, y, width, height]
            detections.append({
                "bbox": pred["bbox"],
                "category_id": pred["category_id"],
                "score": pred["score"]
            })
        
        return detections
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def process_base64_image(b64_string, image_id, detection_model, temp_dir="temp_images"):
    """Process an image from base64 string using SAHI sliced inference"""
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Decode base64 to image
        img_bytes = base64.b64decode(b64_string)
        temp_path = f"{temp_dir}/{image_id}.jpg"
        
        with open(temp_path, "wb") as f:
            f.write(img_bytes)
        
        # Use the new SAHI API for sliced prediction
        result = get_sliced_prediction(
            temp_path,
            detection_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        
        # Convert to COCO predictions format
        coco_predictions = result.to_coco_predictions(image_id=image_id)
        
        # Format the results according to COCO format
        detections = []
        for pred in coco_predictions:
            detections.append({
                "bbox": pred["bbox"],
                "category_id": pred["category_id"],
                "score": pred["score"]
            })
        
        # Optional: Clean up temp file
        os.remove(temp_path)
        
        return detections
        
    except Exception as e:
        print(f"Error processing image ID {image_id}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    print("Setting up detection model...")
    detection_model = setup_detection_model()
    
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/cv")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "annotations.json", "r") as f:
        annotations = json.load(f)
    
    instances = annotations["images"]
    batch_generator = itertools.batched(sample_generator(instances, data_dir), n=BATCH_SIZE)
    
    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        batch_results = []
        
        for instance in batch:
            image_id = instance["key"]
            # Process image directly from file path if available (more efficient)
            if "file_path" in instance and os.path.exists(instance["file_path"]):
                detections = process_image(instance["file_path"], detection_model)
            else:
                # Fall back to processing from base64 if needed
                detections = process_base64_image(instance["b64"], image_id, detection_model)
            
            # Add image_id to each detection
            for detection in detections:
                results.append({
                    "image_id": image_id,
                    "score": float(detection["score"]),
                    "bbox": [float(x) for x in detection["bbox"]],  # Ensure all values are float
                    "category_id": int(detection["category_id"]),   # Ensure category_id is int
                })
    
    results_path = results_dir / "cv_results.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)
    
    mean_ap = score_cv(results, annotations)
    print("mAP@.5:.05:.95:", mean_ap)

if __name__ == "__main__":
    main()