import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from PIL import Image
import numpy as np
import io
from docx import Document

st.set_page_config(page_title="VirtualCo | Advanced HTR", layout="wide")

# 1. Load the "Line Finder" (EasyOCR) and the "Line Reader" (TrOCR)
@st.cache_resource
def load_models():
    # EasyOCR is great at finding WHERE the text is
    detector = easyocr.Reader(['en'])
    # TrOCR is great at reading WHAT the handwriting says
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return detector, processor, model

st.title("🧪 Smart Chemistry Note Converter")
st.write("This version uses a **Detection + Recognition Pipeline** to handle full pages.")

uploaded_file = st.file_uploader("Upload Handwritten Page", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Document", width=600)
    
    if st.button("🚀 Process Full Page"):
        detector, processor, model = load_models()
        
        with st.spinner("Detecting layout and reading handwriting..."):
            # A. Detect all text boxes (lines)
            img_np = np.array(image)
            # detect returns bounding boxes
            boxes = detector.readtext(img_np) 
            
            # B. Prepare Word Doc
            doc = Document()
            doc.add_heading('Converted Handwritten Notes', 0)
            
            full_text = ""
            
            # C. Loop through each detected line and run TrOCR
            progress_bar = st.progress(0)
            for i, (bbox, text_guess, prob) in enumerate(boxes):
                # Extract coordinates for the crop
                top_left = bbox[0]
                bottom_right = bbox[2]
                
                # Crop the original image to just this one line
                line_img = image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
                
                # Run TrOCR on the small crop
                pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Add to results
                full_text += line_text + "\n"
                doc.add_paragraph(line_text)
                progress_bar.progress((i + 1) / len(boxes))

            st.success("Full Page Processed!")
            st.text_area("Final Result:", full_text, height=300)
            
            # Download
            bio = io.BytesIO()
            doc.save(bio)
            st.download_button("📥 Download .docx", bio.getvalue(), "Chemistry_Notes.docx")