"""Manages the CV model."""
from typing import Any
import os
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import uuid
import logging as log

class LetterboxTransform:
    """
    Letterbox transform that resizes and pads the image to maintain aspect ratio.
    Similar to YOLO's letterboxing approach.
    """
    def __init__(self, size=(640, 640)):
        self.size = size
        
    def __call__(self, img):
        # Calculate the padding needed to maintain aspect ratio
        width, height = img.size
        target_w, target_h = self.size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / width, target_h / height)
        
        # Calculate new size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image maintaining aspect ratio
        resized_img = F.resize(img, [new_height, new_width])
        
        # Calculate padding
        pad_width = target_w - new_width
        pad_height = target_h - new_height
        
        # Padding on each side
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        
        # Apply padding
        padded_img = F.pad(resized_img, [pad_left, pad_top, pad_right, pad_bottom], 0)
        
        return padded_img, scale, (pad_left, pad_top)

class CVManager:
    def __init__(self):
        # Load the YOLO model
        self.yolo_model = YOLO("yolo11_v3.pt")
        
        # SAHI-compatible detection model
        yolo_model_path = "yolo11_v3.pt"
        sahi_model_type = 'ultralytics'  
        confidence_threshold = 0.3
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.sahi_detection_model = AutoDetectionModel.from_pretrained(
                model_type=sahi_model_type,
                model_path=yolo_model_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
            log.info("sahi works")
        except Exception as e:
            log.info(f"Failed to initialize SAHI model: {e}")
            self.sahi_detection_model = None
            
        # Directory for temporarily storing input images for SAHI
        self.sahi_temp_dir = "temp_sahi_images"
        os.makedirs(self.sahi_temp_dir, exist_ok=True)
        
    def cv(self, image: bytes, use_sahi: bool = True) -> list[dict[str, Any]]:
        """Performs object detection on an image.
        
        Args:
            image: The image file in bytes.
            use_sahi: Whether to use SAHI for sliced inference (True) or regular YOLO (False).
            
        Returns:
            A list of predictions with required format:
            - bbox: [x, y, w, h] - Bounding box coordinates and dimensions in pixels
            - category_id: Index of the predicted category
            - score: Confidence score (only returned with SAHI)
        """       
        if use_sahi and self.sahi_detection_model:
            # Use SAHI approach with sliced inference
            return self._process_with_sahi(image)
        else:
            # Use standard YOLO approach
            return self._process_with_yolo(image)
            
    def _process_with_yolo(self, image: bytes) -> list[dict[str, Any]]:
        """Process image using standard YOLO inference."""
        # Process the image
        pil_image = Image.open(io.BytesIO(image)).convert('RGB')
        original_width, original_height = pil_image.size
        
        # Apply letterboxing transform
        letterbox = LetterboxTransform(size=(640, 640))
        padded_img, scale, padding = letterbox(pil_image)
        pad_left, pad_top = padding
        
        # Convert to tensor and prepare for model
        transform = T.Compose([T.ToTensor()])
        tensor = transform(padded_img).unsqueeze(0)  # Add batch dimension
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)
        
        # Run prediction
        result = self.yolo_model.predict(tensor)[0]
        boxes = result.boxes
        
        # Get coordinates (in xyxy format: x1, y1, x2, y2)
        bounding_boxes_xyxy = boxes.xyxy
        
        # Get confidence scores and classes
        confidence = boxes.conf
        classes = boxes.cls
        
        output = []
        
        # Process each detection
        for pred_index in range(len(boxes)):
            # Extract coordinates from model output (these are in padded image space)
            x1, y1, x2, y2 = [float(coord) for coord in bounding_boxes_xyxy[pred_index]]
            
            # Adjust for padding
            x1 = (x1 - pad_left) if x1 > pad_left else 0
            y1 = (y1 - pad_top) if y1 > pad_top else 0
            x2 = (x2 - pad_left) if x2 > pad_left else 0
            y2 = (y2 - pad_top) if y2 > pad_top else 0
            
            # Convert back to original image coordinates
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            
            # Calculate width and height
            w = x2 - x1
            h = y2 - y1
            
            # Get category ID and confidence
            category_id = int(classes[pred_index].item())
            score = float(confidence[pred_index].item())
            
            # Create prediction dictionary in required format
            pred_dict = {
                "bbox": [x1, y1, w, h],
                "category_id": category_id,
                "score": score
            }
            
            output.append(pred_dict)
            
        return output
    
    def _process_with_sahi(self, image: bytes) -> list[dict[str, Any]]:
        """Process image using SAHI sliced inference."""
        if not self.sahi_detection_model:
            print("SAHI model not initialized. Falling back to standard YOLO.")
            return self._process_with_yolo(image)
            
        try:
            # Save the image temporarily to work with SAHI
            temp_image_path = os.path.join(self.sahi_temp_dir, f"{uuid.uuid4()}.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(image)
                
            # Get predictions using SAHI sliced inference
            result = get_sliced_prediction(
                temp_image_path,
                self.sahi_detection_model,
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
                    "bbox": pred["bbox"],  # Already in [x, y, w, h] format
                    "category_id": pred["category_id"],
                    "score": pred["score"]
                })
            
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
            return detections
            
        except Exception as e:
            print(f"SAHI processing failed: {e}")
            print("Falling back to standard YOLO.")
            return self._process_with_yolo(image)