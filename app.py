import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from PIL import Image
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

uploaded_file = st.file_uploader("Upload Handwritten Page", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Document", width=700)
    
    if st.button("🚀 Process Full Page"):
        detector, processor, model = load_models()
        
        with st.spinner("Analyzing document structure and deciphering handwriting..."):
            # A. Detect all text boxes using EasyOCR
            img_np = np.array(image)
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
                
                # Crop the specific region from the original image
                region_img = image.crop((x_min, y_min, x_max, y_max))
                
                try:
                    # Run TrOCR on the individual region
                    pixel_values = processor(images=region_img, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    region_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    if region_text.strip():
                        # Store text with its position for grouping
                        text_boxes.append({
                            'text': region_text.strip(),
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

            # D. Group text boxes by lines using accurate clustering algorithm
            if text_boxes:
                img_height = image.height
                
                # Sort all boxes by Y position (top to bottom), then by X (left to right)
                text_boxes_sorted = sorted(text_boxes, key=lambda b: (b['y_center'], b['x_min']))
                
                # Calculate adaptive threshold based on actual spacing between words
                if len(text_boxes_sorted) > 1:
                    # Calculate Y differences between consecutive boxes
                    y_diffs = []
                    for i in range(len(text_boxes_sorted) - 1):
                        y_diff = abs(text_boxes_sorted[i+1]['y_center'] - text_boxes_sorted[i]['y_center'])
                        y_diffs.append(y_diff)
                    
                    # Separate small gaps (words on same line) from large gaps (line breaks)
                    # Use median of small differences as threshold
                    import statistics
                    small_diffs = [d for d in y_diffs if d < img_height * 0.1]  # Ignore large gaps
                    if small_diffs:
                        median_small_diff = statistics.median(small_diffs)
                        # Use 0.7x median for conservative grouping (only group very close words)
                        line_threshold = max(median_small_diff * 0.7, 15)
                    else:
                        # Fallback: use average height
                        heights = [box['height'] for box in text_boxes]
                        avg_height = sum(heights) / len(heights)
                        line_threshold = max(avg_height * 0.6, 20)
                else:
                    line_threshold = 25
                
                # Group boxes into lines using conservative threshold
                lines = []
                current_line = [text_boxes_sorted[0]]
                current_line_y = text_boxes_sorted[0]['y_center']
                
                for i in range(1, len(text_boxes_sorted)):
                    box = text_boxes_sorted[i]
                    y_diff = abs(box['y_center'] - current_line_y)
                    
                    # Also check if boxes overlap vertically (more reliable indicator of same line)
                    y_overlap = not (box['y_min'] > current_line[-1]['y_max'] or box['y_max'] < current_line[-1]['y_min'])
                    
                    if y_diff <= line_threshold and y_overlap:
                        # Same line - add to current line
                        current_line.append(box)
                        # Update average Y position of the line
                        current_line_y = sum(b['y_center'] for b in current_line) / len(current_line)
                    else:
                        # New line detected - save current line and start new one
                        # Sort boxes in line by X position (left to right) for correct reading order
                        current_line.sort(key=lambda b: b['x_min'])
                        lines.append(current_line)
                        current_line = [box]
                        current_line_y = box['y_center']
                
                # Don't forget the last line
                if current_line:
                    current_line.sort(key=lambda b: b['x_min'])
                    lines.append(current_line)
                
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