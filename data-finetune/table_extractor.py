import os
import uuid
from pathlib import Path
import glob

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

from transformers import AutoModelForObjectDetection
from pdf2image import convert_from_path
from collections import Counter
from loguru import logger

# --- Configuration ---
# Directory to save cropped table images
CROPPED_TABLES_OUTPUT_DIR = "C:/Users/PC/CODE/WDM-AI-TEMIS/output_img/0aed309e29e45111f67fb85aea1fcb5e"
# DPI for PDF to image conversion
PDF_DPI = 200 # Increased DPI for better quality if needed, original notebook used 144 for get_images_from_pdf
# Detection model parameters
DETECTION_CLASS_THRESHOLDS = {
    "table": 0.7, # Adjusted based on typical needs, notebook had 0.5
    "table rotated": 0.7,
    "no object": 10, # Effectively ignore "no object"
}
CROP_PADDING = 20 # Notebook used 10 then 25, 20 is a compromise

# --- Helper Functions (from notebook, slightly adapted) ---

def get_specific_pages_as_images(pdf_path, page_numbers_to_extract, dpi_resolution=PDF_DPI):
    """
    Convert specific PDF pages to PIL Images.
    page_numbers_to_extract is a list of 1-based page numbers.
    Returns a dictionary: {page_number: PIL.Image}
    """
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f'PDF file {pdf_path} does not exist'

    images_dict = {}
    
    # pdf2image convert_from_path uses 0-based indexing for pages if we were to get all,
    # but it's easier to convert one by one if page numbers are not contiguous.
    # However, convert_from_path is more efficient if it converts a range.
    # For simplicity and direct mapping to user's 1-based page_numbers:
    
    # Get all pages first, then select. This might be memory intensive for huge PDFs.
    # Alternative: iterate and call convert_from_path for each page if memory is an issue.
    try:
        all_pil_images = convert_from_path(pdf_path, dpi=dpi_resolution)
    except Exception as e:
        logger.error(f"Could not convert PDF {pdf_path} to images: {e}")
        return {}

    for page_num in page_numbers_to_extract:
        if 1 <= page_num <= len(all_pil_images):
            # User provides 1-based, list is 0-based
            img = all_pil_images[page_num - 1].convert('RGB') 
            
            # Optional: Resize to a common size if consistency is needed across pages
            # For table detection, it's often better to work with original/MaxResized aspect ratio
            # img_size_counter = Counter()
            # img_size_counter[img.size] +=1
            # common_img_size, _ = img_size_counter.most_common(1)[0]
            # if img.size != common_img_size:
            #     logger.info(f'Resizing page {page_num} image to {common_img_size}')
            #     img = img.resize(common_img_size)
            
            images_dict[page_num] = img
            logger.info(f"Successfully converted page {page_num} from {pdf_path.name} to image.")
        else:
            logger.warning(f"Page number {page_num} is out of range for PDF {pdf_path.name} (total pages: {len(all_pil_images)}).")
            
    return images_dict


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        if current_max_size == 0: # Avoid division by zero for empty images
             return image
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

# Global: Define detection_transform once
detection_transform = transforms.Compose([
    MaxResize(800), # Notebook used 800
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device) # Ensure tensor is on same device
    return b

def outputs_to_objects(outputs, img_size, id2label_map):
    logits = outputs.logits.cpu() # Move to CPU
    pred_boxes = outputs.pred_boxes.cpu() # Move to CPU

    m = logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().numpy())[0]
    pred_scores = list(m.values.detach().numpy())[0]
    pred_bboxes_scaled = rescale_bboxes(pred_boxes[0], img_size)
    pred_bboxes_list = [elem.tolist() for elem in pred_bboxes_scaled]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes_list):
        class_label = id2label_map.get(int(label), "unknown") # Use .get for safety
        if class_label != 'no object' and class_label != "unknown": # Filter out "no object"
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})
    return objects

def objects_to_crops(img, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes from detection into cropped table images.
    'tokens' argument removed as it's not used for this specific task.
    """
    table_crops_pil = [] # Store PIL images directly
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        bbox = obj['bbox']
        # Apply padding, ensuring coordinates are within image bounds
        padded_bbox = [
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(img.width, bbox[2] + padding),
            min(img.height, bbox[3] + padding)
        ]
        
        # Ensure the bounding box is valid (e.g. x2 > x1)
        if padded_bbox[0] >= padded_bbox[2] or padded_bbox[1] >= padded_bbox[3]:
            logger.warning(f"Skipping invalid bbox after padding: {padded_bbox} for object {obj}")
            continue

        cropped_img = img.crop(padded_bbox)

        # # If table is predicted to be rotated, rotate cropped image
        # if obj['label'] == 'table rotated':
        #     cropped_img = cropped_img.rotate(270, expand=True)
        
        table_crops_pil.append(cropped_img)
    return table_crops_pil

# --- Main Processing Function ---
def extract_and_save_tables_from_pdf(pdf_file_path: str, list_of_page_numbers: list, output_dir: str = CROPPED_TABLES_OUTPUT_DIR):
    """
    Extracts tables from specified pages of a PDF, saves them as images,
    and returns a list of paths to these images.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load detection model
    logger.info("Loading table detection model...")
    detection_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detection_model.to(device)
    logger.info(f"Table detection model loaded on {device}.")

    # Prepare id2label mapping for detection model
    id2label_detection = detection_model.config.id2label
    id2label_detection[len(id2label_detection)] = "no object" # As done in notebook

    pdf_path_obj = Path(pdf_file_path)
    pdf_name_stem = pdf_path_obj.stem

    page_images = get_specific_pages_as_images(pdf_file_path, list_of_page_numbers, dpi_resolution=PDF_DPI)
    
    if not page_images:
        logger.error(f"No images could be extracted from {pdf_file_path} for pages {list_of_page_numbers}.")
        return []

    saved_cropped_image_paths = []

    for page_num, pil_image in page_images.items():
        logger.info(f"Processing page {page_num} from {pdf_path_obj.name}...")
        
        # Prepare image for model
        pixel_values = detection_transform(pil_image).unsqueeze(0).to(device)

        # Forward pass for table detection
        with torch.no_grad():
            outputs = detection_model(pixel_values)
        
        detected_objects = outputs_to_objects(outputs, pil_image.size, id2label_detection)
        
        if not detected_objects:
            logger.info(f"No tables detected on page {page_num} of {pdf_path_obj.name}.")
            continue
        
        logger.info(f"Detected {len(detected_objects)} potential table(s) on page {page_num}.")

        # Filter objects by score before cropping
        filtered_objects = [obj for obj in detected_objects if obj['score'] >= DETECTION_CLASS_THRESHOLDS.get(obj['label'], 1.0)]

        if not filtered_objects:
            logger.info(f"No tables passed threshold on page {page_num} of {pdf_path_obj.name}.")
            continue
            
        cropped_pil_tables = objects_to_crops(pil_image, filtered_objects, DETECTION_CLASS_THRESHOLDS, padding=CROP_PADDING)
        
        for i, cropped_table_img in enumerate(cropped_pil_tables):
            try:
                # ---- THAY ĐỔI CÁCH ĐẶT TÊN FILE Ở ĐÂY ----
                new_uuid_str = str(uuid.uuid4())
                crop_filename = f"{new_uuid_str}.png"
                # -----------------------------------------
                
                crop_filepath = Path(output_dir) / crop_filename # output_dir được truyền vào hàm
                
                cropped_table_img.convert("RGB").save(crop_filepath)
                logger.info(f"Saved cropped table to: {crop_filepath}")
                saved_cropped_image_paths.append(str(crop_filepath))
            except Exception as e:
                logger.error(f"Error saving cropped table from page {page_num}: {e}")

    return saved_cropped_image_paths

# --- Script execution / Example Usage ---
if __name__ == "__main__":
    # Make sure poppler is installed and in PATH
    # On Linux: sudo apt-get install poppler-utils
    # On macOS: brew install poppler
    # On Windows: Download Poppler, extract, and add bin/ to PATH.
    #             Or specify poppler_path in convert_from_path if not in PATH.

    # Example: Create a dummy PDF for testing if you don't have one
    # This part requires 'reportlab' - pip install reportlab
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        PDF_PATH = "C:/Users/PC/CODE/WDM-AI-TEMIS/data-finetune/pdf4tabel/0aed309e29e45111f67fb85aea1fcb5e.pdf"

        # def create_dummy_pdf(filename=PDF_PATH, num_pages=5):
        #     if os.path.exists(filename):
        #         logger.info(f"Dummy PDF {filename} already exists. Skipping creation.")
        #         return
        #     c = canvas.Canvas(filename, pagesize=letter)
        #     for i in range(num_pages):
        #         c.drawString(100, 750, f"This is page {i+1} of the dummy PDF.")
        #         # Simulate a table-like structure
        #         if i % 2 == 0 : # Add "table" on even pages
        #             c.rect(100, 400, 400, 200, stroke=1, fill=0) # A simple rectangle
        #             c.drawString(110, 580, "A simulated table area")
        #             c.line(100, 500, 500, 500)
        #             c.line(300, 400, 300, 600)
        #         c.showPage()
        #     c.save()
        #     logger.info(f"Created dummy PDF: {filename} with {num_pages} pages.")
        
        # create_dummy_pdf()
        pdf_to_process = PDF_PATH
        pages_to_process = [1, 2, 3] # Process pages 1, 2, and 3 (1-based)

    except ImportError:
        logger.warning("ReportLab not installed. Skipping dummy PDF creation.")
        logger.info("Please provide a PDF_FILE_PATH and PAGES_TO_PROCESS manually for testing.")
        # MANUALLY SET THESE IF REPORTLAB IS NOT AVAILABLE:
        # pdf_to_process = "path/to/your/actual.pdf" 
        # pages_to_process = [1, 5, 10] 
        pdf_to_process = None # Ensure it's defined

    if pdf_to_process and os.path.exists(pdf_to_process):
        logger.info(f"Starting table extraction for PDF: {pdf_to_process}, pages: {pages_to_process}")
        
        # Ensure the output directory for this specific PDF run is clean or unique
        # For simplicity, we'll use the global CROPPED_TABLES_OUTPUT_DIR.
        # If running multiple times, you might want to clear it or use subdirectories.
        
        # Example of clearing directory before run (optional):
        # output_path = Path(CROPPED_TABLES_OUTPUT_DIR)
        # if output_path.exists():
        #     logger.info(f"Cleaning output directory: {output_path}")
        #     for f_path in glob.glob(str(output_path / '*')):
        #         try:
        #             os.remove(f_path)
        #         except OSError as e:
        #             logger.error(f"Error removing file {f_path}: {e}")
        # else:
        #    output_path.mkdir(parents=True, exist_ok=True)


        cropped_image_files = extract_and_save_tables_from_pdf(
            pdf_file_path=pdf_to_process,
            list_of_page_numbers=pages_to_process,
            output_dir=CROPPED_TABLES_OUTPUT_DIR 
        )

        if cropped_image_files:
            logger.info("\n--- Successfully Extracted Cropped Tables ---")
            for file_path in cropped_image_files:
                logger.info(file_path)
            logger.info(f"All cropped tables saved in: {Path(CROPPED_TABLES_OUTPUT_DIR).resolve()}")
        else:
            logger.info("No tables were extracted or saved.")
    elif pdf_to_process:
        logger.error(f"PDF file not found: {pdf_to_process}")
    else:
        logger.info("No PDF file specified for processing in the example.")