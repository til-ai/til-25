import base64
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
import pprint

def debug_cv_predictions():
    # Load environment variables
    load_dotenv()
    TEAM_NAME = os.getenv("TEAM_NAME")
    TEAM_TRACK = os.getenv("TEAM_TRACK")
    
    # Paths
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/cv")
    
    # Load annotations
    with open(data_dir / "annotations.json", "r") as f:
        annotations = json.load(f)
    
    print(f"Total images: {len(annotations['images'])}")
    
    # Test only on a small sample (5 images)
    sample_size = 5
    instances = annotations['images'][:sample_size]
    
    # Create sample data
    samples = []
    for instance in instances:
        with open(data_dir / "images" / instance["file_name"], "rb") as img_file:
            img_data = img_file.read()
            samples.append({
                "key": instance["id"],
                "b64": base64.b64encode(img_data).decode("ascii"),
            })
    
    # Make prediction request
    print(f"Requesting predictions for {len(samples)} images...")
    response = requests.post("http://localhost:5002/cv", data=json.dumps({
        "instances": samples,
    }))
    
    # Get predictions
    batch_preds = response.json()["predictions"]
    
    # Debug: inspect the raw prediction format
    print("\n=== Raw Prediction Format ===")
    print(f"Number of prediction results: {len(batch_preds)}")
    if batch_preds:
        first_image_detections = batch_preds[0]
        print(f"Number of detections in first image: {len(first_image_detections)}")
        if first_image_detections:
            print("\nExample detection:")
            pprint.pprint(first_image_detections[0])
            
            # Specifically investigate the bbox field
            bbox = first_image_detections[0].get("bbox")
            print(f"\nBounding box type: {type(bbox)}")
            print(f"Bounding box value: {bbox}")
    
    # Process the results as in the original code, but with better error handling
    results = []
    for instance, single_image_detections in zip(samples, batch_preds):
        print(f"\nProcessing image ID: {instance['key']}")
        print(f"Number of detections: {len(single_image_detections)}")
        
        for i, detection in enumerate(single_image_detections):
            print(f"\n  Detection #{i+1}:")
            print(f"  Category ID: {detection.get('category_id')}")
            print(f"  Bbox type: {type(detection.get('bbox'))}")
            print(f"  Bbox value: {detection.get('bbox')}")
            
            # Try to properly format the bbox
            bbox = detection.get("bbox")
            if isinstance(bbox, dict):
                print("  Converting dict bbox to list format")
                # Try different possible dictionary formats
                if all(k in bbox for k in ['x', 'y', 'width', 'height']):
                    bbox_list = [bbox['x'], bbox['y'], bbox['width'], bbox['height']]
                elif all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
                    # Convert from x1,y1,x2,y2 to x,y,width,height
                    bbox_list = [
                        bbox['x1'], 
                        bbox['y1'], 
                        bbox['x2'] - bbox['x1'], 
                        bbox['y2'] - bbox['y1']
                    ]
                else:
                    print(f"  Unknown bbox format: {bbox}")
                    bbox_list = [0, 0, 0, 0]  # Default fallback
                
                print(f"  Converted bbox: {bbox_list}")
                
                # Add to results with converted bbox
                results.append({
                    "image_id": instance["key"],
                    "score": detection.get("score", 1.0),
                    "bbox": bbox_list,
                    "category_id": detection.get("category_id"),
                })
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                # Already in the correct format
                results.append({
                    "image_id": instance["key"],
                    "score": detection.get("score", 1.0),
                    "bbox": bbox,
                    "category_id": detection.get("category_id"),
                })
            else:
                print(f"  Unexpected bbox format: {bbox}")
    
    print("\n=== Processed Results ===")
    if results:
        print(f"Number of processed detections: {len(results)}")
        print("First processed result:")
        pprint.pprint(results[0])
    else:
        print("No results processed!")

if __name__ == "__main__":
    debug_cv_predictions()