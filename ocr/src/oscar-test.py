from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import os
import pandas as pd

def create_labels_csv(image_folder, output_csv_path):
    """
    Scans an image folder for .jpg files and corresponding _text.txt files
    to create a labels.csv for OCR training.

    Args:
        image_folder (str): Path to the folder containing images and text files.
        output_csv_path (str): Path where the output labels.csv will be saved.
    """
    image_files = []
    text_contents = []
    found_pairs = 0
    missing_texts = 0
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif') # Add more if needed

    print(f"Scanning folder: {image_folder}")

    for filename in os.listdir(image_folder):
        file_basename, file_extension = os.path.splitext(filename)

        if file_extension.lower() in image_extensions:
            # Construct the expected text file name
            # Assumes text file is like 'sample_NUMBER_text.txt' for 'sample_NUMBER.jpg'
            # If your image name itself is 'sample_NUMBER_text.jpg', adjust this logic
            if file_basename.endswith("_text"): # Handles cases if image itself has _text
                 # This is unlikely for your jpgs but good to consider
                 pass # Or some other logic if image names are complex
            
            # Expected text file name based on your example like sample_996.jpg -> sample_996_text.txt
            # If sample_1573.jpg -> sample_1573_text.txt
            expected_text_filename = f"{file_basename}_text.txt" 
            
            # But your list also shows sample_1572_text.txt (without a clear sample_1572.jpg next to it)
            # Let's refine the logic: if we find an image, we look for its corresponding text file.
            # For an image like "sample_123.jpg", we look for "sample_123_text.txt"

            text_file_path = os.path.join(image_folder, expected_text_filename)

            if os.path.exists(text_file_path):
                try:
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text: # Only add if text file is not empty
                        image_full_path = os.path.join(image_folder, filename)
                        image_files.append(image_full_path)
                        text_contents.append(text)
                        found_pairs += 1
                    else:
                        print(f"Warning: Text file '{expected_text_filename}' is empty for image '{filename}'. Skipping.")
                        missing_texts +=1 # Count as missing if empty
                        
                except Exception as e:
                    print(f"Error reading text file '{expected_text_filename}' for image '{filename}': {e}")
                    missing_texts += 1
            else:
                # Check if the pattern is sample_NUMBER.jpg -> sample_NUMBER.txt (without _text)
                # This is less likely given your example but good to be flexible
                alternative_text_filename = f"{file_basename}.txt"
                alternative_text_file_path = os.path.join(image_folder, alternative_text_filename)
                if os.path.exists(alternative_text_file_path):
                    try:
                        with open(alternative_text_file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        if text:
                            image_full_path = os.path.join(image_folder, filename)
                            image_files.append(image_full_path)
                            text_contents.append(text)
                            found_pairs += 1
                        else:
                            print(f"Warning: Text file '{alternative_text_filename}' is empty for image '{filename}'. Skipping.")
                            missing_texts +=1
                    except Exception as e:
                        print(f"Error reading text file '{alternative_text_filename}' for image '{filename}': {e}")
                        missing_texts += 1
                else:
                    print(f"Warning: No corresponding text file ('{expected_text_filename}' or '{alternative_text_filename}') found for image '{filename}'. Skipping.")
                    missing_texts += 1

    if not image_files:
        print("No image-text pairs found. Please check your file naming convention and folder content.")
        return

    df = pd.DataFrame({
        'image_path': image_files,
        'text': text_contents
    })

    df.to_csv(output_csv_path, index=False)
    print(f"\nSuccessfully created '{output_csv_path}' with {found_pairs} image-text pairs.")
    if missing_texts > 0:
        print(f"Could not find or process text for {missing_texts} images.")



if __name__ == "__main__":
    
#     image_folder = "/home/jupyter/advanced/ocr/"

#     image = Image.open("/home/jupyter/advanced/ocr/sample_1577.jpg").convert("RGB")

#     processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed', use_fast=False)
#     model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values

#     generated_ids = model.generate(pixel_values)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # --- Configuration ---
    image_folder_path = "/home/jupyter/advanced/ocr/"
    # Save the CSV in the same folder, or specify a different path
    output_csv_file = os.path.join(image_folder_path, "labels.csv")

    # --- Run the script ---
    create_labels_csv(image_folder_path, output_csv_file)
