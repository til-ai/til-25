"""Manages the CV model."""
from typing import Any
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import math

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
        # Load YOLO model
        self.model = YOLO("yolo11_v2.pt")
        
    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.
        
        Args:
            image: The image file in bytes.
            
        Returns:
            A list of predictions with required format:
            - (x,y): Top-left corner coordinates of the bounding box (in pixels)
            - (w,h): Width and height of the bounding box (in pixels)
            - category_id: Index of the predicted category
        """       
        # Process the image
        processed_image, orig_h, orig_w, scale, padding = preprocess_image(image)
        pad_left, pad_top = padding
        
        # Run prediction
        result = self.model.predict(processed_image)[0]
        boxes = result.boxes
        
        # Get coordinates (in xyxy format: x1, y1, x2, y2)
        bounding_boxes_xyxy = boxes.xyxy
        
        # Get confidence scores and classes
        confidence = boxes.conf
        classes = boxes.cls
        
        output = []
        
        # Process each detection
        for pred_index in range(boxes.shape[0]):
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
            
            # Get category ID
            category_id = int(classes[pred_index].item())
            
            # Create prediction dictionary in required format
            pred_dict = {
                "bbox": [x1, y1, w, h],
                "category_id": category_id
            }
            print(pred_dict)
            output.append(pred_dict)
            
        return output

def preprocess_image(image: bytes):
    """
    Preprocess the image for the YOLO model with letterboxing to maintain aspect ratio.
    
    Args:
        image: Image bytes
        
    Returns:
        Processed tensor, original height, original width, scale, padding
    """
    # Convert bytes to PIL Image
    pil_image = Image.open(io.BytesIO(image)).convert('RGB')
    original_width, original_height = pil_image.size
    
    # Apply letterboxing transform (resize with padding to maintain aspect ratio)
    letterbox = LetterboxTransform(size=(640, 640))
    padded_img, scale, padding = letterbox(pil_image)
    
    # Convert to tensor
    transform = T.Compose([
        T.ToTensor(),
    ])
    tensor = transform(padded_img).unsqueeze(0)  # Add batch dimension
    
    # Move tensor to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return tensor.to(device), original_height, original_width, scale, padding