import streamlit as st
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
from docx import Document

# 1. Setup Model (Using Donut for Document Parsing)
@st.cache_resource
def load_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    return processor, model

st.title("🧪 Advanced AI Image-to-Word")
st.write("Using Hugging Face Transformers for Handwriting Recognition.")

uploaded_file = st.file_uploader("Upload Handwritten Notes", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Document", use_container_width=True)
    
    if st.button("Magic Convert"):
        with st.spinner("AI is reading your handwriting..."):
            processor, model = load_model()
            
            # Prepare image for the model
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # Generate Text
            task_prompt = "<s_docvqa><s_question>Extract all text from the page</s_question><s_answer>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.config.decoder.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            
            # Decode the output
            sequence = processor.batch_decode(outputs.sequences)[0]
            clean_text = sequence.replace(processor.tokenizer.eos_token, "").replace(task_prompt, "")
            
            # Create Word Doc
            doc = Document()
            doc.add_paragraph(clean_text)
            
            bio = io.BytesIO()
            doc.save(bio)
            
            st.success("Converted!")
            st.download_button("Download Docx", bio.getvalue(), "AI_Notes.docx")