# src/ocr_manager.py

import os
import cv2
import numpy as np
import torch
from doctr.models import ocr_predictor, db_resnet50, crnn_vgg16_bn

class OCRManager:
    def __init__(self):
        print("Initializing OCRManager...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load custom detection and recognition model
        det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
        # det_params = torch.load('<path_to_pt>', map_location=device)
        # det_model.load_state_dict(det_params)
        reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
        # reco_params = torch.load('<path_to_pt>', map_location=device)
        # reco_model.load_state_dict(reco_params)
        self.ocr_engine = ocr_predictor(det_arch=det_model, reco_arch=reco_model, pretrained=False)
        print("docTR engine initialized successfully in OCRManager.")


    def ocr(self, image_bytes: bytes) -> str:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None: 
                print("OCRManager Error: Could not decode image from bytes.")
                return "Error: Could not decode image."

            image_height, image_width, _ = img_np.shape
            # print(f"OCRManager: Decoded image dimensions: {image_width}x{image_height}")

            # ocr_result_batch = self.ocr_engine.ocr(img_np, cls=False) # Pass NumPy array
            ocr_result_batch = self.ocr_engine([img_np]) # Pass NumPy array
            if not ocr_result_batch: 
                print("OCRManager: OCR returned no results.")
                return ""
            
            # Extract words from the OCR results
            # doctr output structure:
            # A list of doctr.documents.page.Page objects, one for each image in the batch.
            # Each Page object has a 'export()' method that returns a JSON-like dict.
            # We want the 'words' information from each block on each page.
            extracted_words = []
            for page in ocr_result_batch.pages:
                # page_words = []
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            extracted_words.append(word) # Only select words, no extra info
                            # page_words.append({
                            #     'word': word.value,
                            #     'confidence': word.confidence,
                            #     'box_2d': word.geometry # Keep original bounding box
                            # })
                # extracted_words.append(page_words)
            
            # Convert extracted words to English (Remove mispelled words etc.)
            
            full_text = " ".join(extracted_words)
            return full_text

        except Exception as e:
            print(f"Error in OCRManager.ocr: {e}")
            # import traceback # Uncomment for detailed stack trace during debugging
            # traceback.print_exc()
            return f"Error during OCR processing: {str(e)}"

