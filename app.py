import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
from docx import Document
from docx.shared import Pt
import io

# Professional Page Setup
st.set_page_config(page_title="VirtualCo OCR MVP", page_icon="📝", layout="centered")

# Custom CSS to make it look like a corporate tool
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📑 Image-to-Word Converter")
st.info("Phase 1 MVP: Preserving text and basic structure.")

# Sidebar for Project Info
with st.sidebar:
    st.header("Project Credits")
    st.write("**Group:** 3 Students")
    st.write("**Deadline:** Jan 28, 2026")
    st.divider()
    st.write("Engine: Tesseract OCR")

# File Uploader
uploaded_file = st.file_uploader("Upload Scanned Document (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Document Preview", use_container_width=True)
    
    if st.button("🚀 Convert to Editable Word"):
        with st.spinner("Processing OCR & Formatting..."):
            try:
                # 1. Image Preprocessing for better OCR accuracy
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # 2. Extract Data with Layout information
                # 'image_to_data' gives us block numbers and paragraph structures
                ocr_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
                
                # 3. Build the Word Document
                doc = Document()
                doc.add_heading('Converted Document', 0)
                
                full_text = []
                current_para = doc.add_paragraph()
                last_block = -1
                
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    if text:
                        # Detect new blocks/paragraphs
                        if ocr_data['block_num'][i] != last_block and last_block != -1:
                            current_para = doc.add_paragraph()
                        
                        run = current_para.add_run(text + " ")
                        run.font.size = Pt(11)
                        last_block = ocr_data['block_num'][i]

                # 4. Save to Memory Buffer
                doc_download = io.BytesIO()
                doc.save(doc_download)
                doc_download.seek(0)
                
                st.success("✅ Conversion Complete!")
                st.download_button(
                    label="📥 Download Word Document",
                    data=doc_download,
                    file_name="Converted_Document.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")