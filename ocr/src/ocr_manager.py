# src/ocr_manager.py

import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCRManager:
    def __init__(self):
        print("Initializing OCRManager...")
        
        # Construct model paths from environment variables
        models_base_dir = os.getenv('MODELS_BASE_DIR')
        if not models_base_dir:
            print("CRITICAL ERROR: MODELS_BASE_DIR environment variable not set.")
            # You might want to raise an exception here or have a fallback if appropriate
            # For this Docker setup, it should always be set.
            raise EnvironmentError("MODELS_BASE_DIR not set, cannot locate models.")

        det_model_dir = os.path.join(models_base_dir, os.getenv('DET_MODEL_SUBDIR', 'det/en/en_PP-OCRv3_det_infer'))
        rec_model_dir = os.path.join(models_base_dir, os.getenv('REC_MODEL_SUBDIR', 'rec/en/en_PP-OCRv4_rec_infer'))
        cls_model_dir = os.path.join(models_base_dir, os.getenv('CLS_MODEL_SUBDIR', 'cls/en/ch_ppocr_mobile_v2.0_cls_infer'))
        layout_model_dir = os.path.join(models_base_dir, os.getenv('LAYOUT_MODEL_SUBDIR', 'layout/en/picodet_lcnet_x1_0_fgd_layout_infer'))

        print(f"Constructed Detection model path: {det_model_dir}")
        print(f"Constructed Recognition model path: {rec_model_dir}")
        print(f"Constructed Classification model path: {cls_model_dir}")
        print(f"Constructed Layout model path: {layout_model_dir}")

        # Check if model paths and key files exist
        key_file = 'inference.pdmodel'
        for model_name, model_path in [
            ("Detection", det_model_dir),
            ("Recognition", rec_model_dir),
            ("Classification", cls_model_dir),
            ("Layout", layout_model_dir)
        ]:
            if not os.path.exists(os.path.join(model_path, key_file)):
                 print(f"WARNING: {model_name} model file '{key_file}' not found at {model_path}")
            else:
                 print(f"SUCCESS: Found {model_name} model file at {model_path}")


        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            layout=True,
            show_log=True, # Set to False in production if too verbose
            use_gpu=True,
            # Explicitly pass the model directory paths
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            layout_model_dir=layout_model_dir
        )
        print("PaddleOCR engine initialized successfully in OCRManager.")

    def _separate_columns(self, all_text_lines, image_width_px):
        if not all_text_lines:
            return "", ""
        column_divider_x = image_width_px / 2
        left_column_items, right_column_items = [], []
        for line_info in all_text_lines:
            points, (text, _) = line_info
            if not points: continue
            line_center_x = sum(p[0] for p in points) / len(points)
            y_coordinate = points[0][1] # Top-left y for sorting
            if line_center_x < column_divider_x:
                left_column_items.append((text, y_coordinate))
            else:
                right_column_items.append((text, y_coordinate))
        left_column_items.sort(key=lambda item: item[1])
        right_column_items.sort(key=lambda item: item[1])
        return "\n".join([i[0] for i in left_column_items]), "\n".join([i[0] for i in right_column_items])

    def ocr(self, image_bytes: bytes) -> str:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None: 
                print("OCRManager Error: Could not decode image from bytes.")
                return "Error: Could not decode image."
            
            image_height, image_width, _ = img_np.shape
            # print(f"OCRManager: Decoded image dimensions: {image_width}x{image_height}")

            ocr_result_batch = self.ocr_engine.ocr(img_np, cls=False) # Pass NumPy array
            if not ocr_result_batch or not ocr_result_batch[0]: 
                print("OCRManager: OCR returned no results.")
                return ""
            
            all_text_lines = ocr_result_batch[0]
            left_column_text, right_column_text = self._separate_columns(all_text_lines, image_width)
            
            # Combine columns into a single string
            full_text = f"--- LEFT COLUMN ---\n{left_column_text}\n\n--- RIGHT COLUMN ---\n{right_column_text}"
            return full_text

        except Exception as e:
            print(f"Error in OCRManager.ocr: {e}")
            # import traceback # Uncomment for detailed stack trace during debugging
            # traceback.print_exc()
            return f"Error during OCR processing: {str(e)}"