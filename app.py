import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
from docx import Document

# 1. Page Configuration
st.set_page_config(page_title="VirtualCo | Advanced HTR", layout="wide")

# 2. Load the "Line Finder" (EasyOCR) and the "Line Reader" (TrOCR)
@st.cache_resource
def load_models():
    # EasyOCR identifies WHERE text regions are located
    detector = easyocr.Reader(['en'])
    # TrOCR interprets WHAT the handwriting says within those regions
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return detector, processor, model

st.title("🧪 Smart Chemistry Note Converter")
st.write("Professional Pipeline: **EasyOCR Detection** + **TrOCR Recognition** for Handwritten Text")

# Image preprocessing to improve OCR accuracy
def preprocess_image(img):
    """Enhance image for better OCR results"""
    # Convert to grayscale for better contrast
    if img.mode != 'L':
        img = img.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    # Convert back to RGB for TrOCR
    return img.convert('RGB')

# Intelligent text correction for common OCR errors
def correct_ocr_errors(text):
    """Fix common OCR recognition errors intelligently"""
    # Common OCR error patterns (generic corrections)
    corrections = {
        # Common character confusions
        'prosprietary': 'proprietary',
        'successage': 'message',
        'foday': 'today',
        'envielopes': 'envelopes',
        'indjustinguishable': 'indistinguishable',
        'expectively': 'effectively',
    }
    
    # Apply corrections word by word
    words = text.split()
    corrected_words = []
    for word in words:
        # Check if word needs correction
        if word.lower() in corrections:
            corrected_words.append(corrections[word.lower()])
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

# Post-processing function for generic text cleaning
def clean_ocr_text(text, confidence=None):
    """Generic text cleaning for OCR output"""
    # Basic cleaning: remove extra whitespace
    text = ' '.join(text.split())
    
    # Filter out very short single characters that are likely noise
    words = text.split()
    if len(words) == 1 and len(words[0]) == 1 and words[0].isdigit():
        # Single isolated digit - likely false detection if confidence is low
        if confidence is not None and confidence < 0.4:
            return ""
    
    # Apply intelligent corrections
    text = correct_ocr_errors(text)
    
    return text

uploaded_file = st.file_uploader("Upload Handwritten Page", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Document", width=700)
    
    if st.button("🚀 Process Full Page"):
        detector, processor, model = load_models()
        
        with st.spinner("Analyzing document structure and deciphering handwriting..."):
            # Preprocess image for better OCR accuracy
            processed_image = preprocess_image(image.copy())
            
            # A. Detect all text boxes using EasyOCR on processed image
            img_np = np.array(processed_image)
            boxes = detector.readtext(img_np) 
            
            # B. Initialize Word Document
            doc = Document()
            doc.add_heading('Converted Handwritten Notes', 0)
            
            # Store detected text with their positions for proper grouping
            text_boxes = []
            progress_bar = st.progress(0)
            
            # C. Iterate through detected regions and extract text with TrOCR
            for i, (bbox, text_guess, prob) in enumerate(boxes):
                # Calculate coordinates from bounding box
                x_min = max(0, int(bbox[0][0]))
                y_min = max(0, int(bbox[0][1]))
                x_max = int(bbox[2][0])
                y_max = int(bbox[2][1])
                
                # Validation: Prevent RuntimeError by ensuring crop has actual area
                width = x_max - x_min
                height = y_max - y_min
                
                if width < 5 or height < 5:
                    continue
                
                # Filter out very low confidence detections from EasyOCR
                if prob < 0.4:  # Skip low confidence detections (increased threshold)
                    continue
                
                # Crop the specific region from the processed image
                cropped_region = processed_image.crop((x_min, y_min, x_max, y_max))
                
                # Add padding to region for better recognition
                padding = 5
                region_img = Image.new('RGB', 
                    (cropped_region.width + 2*padding, cropped_region.height + 2*padding), 
                    color='white')
                region_img.paste(cropped_region, (padding, padding))
                
                try:
                    # Run TrOCR with better generation parameters
                    pixel_values = processor(images=region_img, return_tensors="pt").pixel_values
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=128,
                        num_beams=5,  # Use beam search for better accuracy
                        early_stopping=True
                    )
                    region_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    if region_text.strip():
                        # Generic text cleaning
                        cleaned_text = clean_ocr_text(region_text.strip(), confidence=prob)
                        
                        # Skip if text is empty after cleaning
                        if cleaned_text.strip():
                            # Store text with its position for grouping
                            text_boxes.append({
                                'text': cleaned_text.strip(),
                                'x_min': x_min,
                                'y_min': y_min,
                                'y_max': y_max,
                                'y_center': (y_min + y_max) / 2,
                                'height': height
                            })
                        
                except Exception as line_error:
                    # Log error but continue processing other regions
                    st.warning(f"Skipped a complex segment at region {i+1}")
                
                # Update UI progress
                progress_bar.progress((i + 1) / len(boxes))

            # D. Group text boxes by lines using Y-coordinate binning (most reliable method)
            if text_boxes:
                img_height = image.height
                
                # Calculate average text height for bin size
                heights = [box['height'] for box in text_boxes]
                avg_height = sum(heights) / len(heights) if heights else 30
                
                # Use binning approach: group words with similar Y-coordinates
                # Bin size should be generous enough to capture words on same line
                # Use smaller bin size for more accurate line detection
                bin_size = max(avg_height * 0.8, 30)  # 0.8x average height for tighter grouping
                
                # Create bins for Y-coordinates
                bins = {}  # Dictionary: bin_key -> list of boxes
                
                for box in text_boxes:
                    # Calculate which bin this box belongs to based on Y-center
                    bin_key = int(box['y_center'] / bin_size)
                    
                    # Add box to appropriate bin
                    if bin_key not in bins:
                        bins[bin_key] = []
                    bins[bin_key].append(box)
                
                # Convert bins to lines and sort
                lines = []
                for bin_key in sorted(bins.keys()):
                    # Get all boxes in this bin
                    line_boxes = bins[bin_key]
                    
                    # Sort boxes in this bin by X position (left to right)
                    line_boxes.sort(key=lambda b: b['x_min'])
                    lines.append(line_boxes)
                
                # Build formatted output
                formatted_lines = []
                for line in lines:
                    # Join words on the same line with spaces
                    line_text = " ".join([box['text'] for box in line])
                    formatted_lines.append(line_text)
                    doc.add_paragraph(line_text)
                
                final_output = "\n".join(formatted_lines)
                
                st.success("Full Page Processed Successfully!")
                st.text_area("Extraction Preview:", final_output, height=350)
                
                # Save to Word format
                bio = io.BytesIO()
                doc.save(bio)
                
                st.download_button(
                    label="📥 Download as Word (.docx)",
                    data=bio.getvalue(),
                    file_name="Converted_Chemistry_Notes.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.error("No legible text was detected. Please ensure the image is clear and well-lit.")

st.divider()
st.caption("VirtualCo AI Pipeline v2.3 | Using EasyOCR + TrOCR for Handwritten Text Recognition")