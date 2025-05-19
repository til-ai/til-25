from pathlib import Path
import os
import json
from ultralytics import YOLO
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict
from sahi.utils.file import list_files
import albumentations as A
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr
import albumentations as A

# Monkey patch the Albumentations class to use our custom transforms
def custom_albumentations_init(self, p=1.0):
    """Initialize the transform object for YOLO bbox formatted params."""
    self.p = p
    self.transform = None
    self.contains_spatial = False  # Important to set this to False for non-spatial transforms
    
    prefix = colorstr("albumentations: ")
    try:
        # Define transformations that maintain image dimensions
        T = [
            A.RandomRain(p=0.2, slant_lower=-10, slant_upper=10, 
                         drop_length=20, drop_width=1, drop_color=(200, 200, 200), 
                         blur_value=5, brightness_coefficient=0.9),  
            # Removed Rotate with crop_border=True - this was causing variable sizes
            A.Blur(p=0.3),
            A.MedianBlur(p=0.3),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.2),
            # Add safe rotation that doesn't crop the image
            A.SafeRotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        ]
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo"))
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")

# Apply the monkey patch to replace the original __init__ method
Albumentations.__init__ = custom_albumentations_init

def get_dataset_path():
    return Path(os.getcwd()).parent.parent / 'advanced' / 'cv'

def get_output_path():
    # Make sure we write to a directory we have permission to modify
    return Path(os.getcwd()) / "yolo_dataset"

def train():
    # Load YOLO model and train
    output_path = get_output_path()
    output_path.mkdir(exist_ok=True)
    
    # Define metrics output file path
    metrics_file = output_path / "evaluation_metrics.json"
    sahi_metrics_file = output_path / "sahi_evaluation_metrics.json"
    
    # Using standard model name
    model = YOLO("yolo11_v1.pt")
    
    # Train the model
    # Now we can use 'albumentations' parameter because we patched the Albumentations class
    results = model.train(
        data=str(output_path / "dataset.yaml"),
        epochs=80,
        imgsz=640,
        name='yolo11_v2',
        save=True,
        save_period=10,
        augment=True,
    )
    
    # Save the trained model
    model_save_path = output_path / "best.pt"
    model.save(str(model_save_path))
    
    # Run standard validation and save metrics
    print("Running standard validation...")
    val_results = model.val()
    metrics = {
        "precision": float(val_results.box.map),    # mAP@0.5
        "recall": float(val_results.box.mr),        # mean Average Recall
        "mAP50": float(val_results.box.map50),      # mAP@0.5
        "mAP50-95": float(val_results.box.map),     # mAP@0.5:0.95
        "fitness": float(val_results.fitness)       # Overall fitness score
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Standard evaluation metrics saved to {metrics_file}")
    print(f"Training completed!")
    print(f"Model saved to: {model_save_path}")
    print(f"Training logs saved to: {output_path}/training_run")

if __name__ == "__main__":
    train()