import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
from docx import Document

# Set Page Config for 2026 Standards
st.set_page_config(page_title="VirtualCo | AI Handwritten OCR", page_icon="✍️")

# 1. Load the specialized Handwritten Model
@st.cache_resource
def load_handwriting_model():
    # We use the 'handwritten' fine-tuned version of TrOCR
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

st.title("✍️ AI Handwritten-to-Word")
st.markdown("### Phase 1: Virtual Company MVP")
st.write("This model uses Microsoft's TrOCR to interpret handwritten chemical notes.")

uploaded_file = st.file_uploader("Upload your handwritten notes...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Note", width='stretch')
    
    if st.button("🚀 Run AI Extraction"):
        with st.spinner("Analyzing handwriting style and extracting text..."):
            try:
                processor, model = load_handwriting_model()
                
                # Prepare image
                pixel_values = processor(image, return_tensors="pt").pixel_values
                
                # Generate text using the Transformer
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 2. Format into Word
                doc = Document()
                doc.add_heading('Extracted Handwritten Notes', 0)
                doc.add_paragraph(generated_text)
                
                # Buffer for download
                bio = io.BytesIO()
                doc.save(bio)
                
                st.success("Analysis Complete!")
                st.text_area("Preview Extracted Text:", generated_text, height=200)
                
                st.download_button(
                    label="📥 Download as Word (.docx)",
                    data=bio.getvalue(),
                    file_name="AI_Handwritten_Notes.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"Hardware Limit Reached: {e}")
                st.info("Tip: If the image is too large, try cropping it to a specific paragraph.")

st.divider()
st.caption("Group Submission - Jan 2026 | Built with Hugging Face Transformers")