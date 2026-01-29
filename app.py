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
st.write("Professional Pipeline: **EasyOCR Detection** + **Transformer Recognition**")

uploaded_file = st.file_uploader("Upload Handwritten Page", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Document", width=700)
    
    if st.button("🚀 Process Full Page"):
        detector, processor, model = load_models()
        
        with st.spinner("Analyzing document structure and deciphering handwriting..."):
            # A. Detect all text boxes
            img_np = np.array(image)
            boxes = detector.readtext(img_np) 
            
            # B. Initialize Word Document
            doc = Document()
            doc.add_heading('Converted Handwritten Notes', 0)
            
            full_text_list = []
            progress_bar = st.progress(0)
            
            # C. Iterate through detected regions
            for i, (bbox, text_guess, prob) in enumerate(boxes):
                # Calculate coordinates from bounding box
                # bbox format: [[top_left], [top_right], [bottom_right], [bottom_left]]
                x_min = max(0, int(bbox[0][0]))
                y_min = max(0, int(bbox[0][1]))
                x_max = int(bbox[2][0])
                y_max = int(bbox[2][1])
                
                # 🛡️ VALIDATION: Prevent RuntimeError by ensuring crop has actual area
                width = x_max - x_min
                height = y_max - y_min
                
                if width < 5 or height < 5:
                    continue
                
                # Crop the specific line from the original high-res image
                line_img = image.crop((x_min, y_min, x_max, y_max))
                
                try:
                    # Run TrOCR on the individual line
                    pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values)
                    line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    if line_text.strip():
                        full_text_list.append(line_text)
                        doc.add_paragraph(line_text)
                        
                except Exception as line_error:
                    # Log error but continue processing other lines
                    st.warning(f"Skipped a complex segment at line {i+1}")
                
                # Update UI progress
                progress_bar.progress((i + 1) / len(boxes))

            # D. Finalize Results
            if full_text_list:
                st.success("Full Page Processed Successfully!")
                
                final_output = "\n".join(full_text_list)
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
st.caption("VirtualCo AI Pipeline v2.1 | Using Microsoft TrOCR & EasyOCR")