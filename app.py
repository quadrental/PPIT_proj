import streamlit as st
import pytesseract
from PIL import Image
import io
from docx import Document

# 1. Page Configuration
st.set_page_config(page_title="VirtualCo | Advanced HTR", layout="wide")

st.title("🧪 Smart Chemistry Note Converter")
st.write("Professional Pipeline: **Tesseract OCR** for Clear Text Extraction")

uploaded_file = st.file_uploader("Upload Handwritten Page", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Document", width=700)
    
    if st.button("🚀 Process Full Page"):
        with st.spinner("Extracting text using Tesseract OCR..."):
            # Initialize Word Document
            doc = Document()
            doc.add_heading('Converted Handwritten Notes', 0)
            
            # Extract text using Tesseract OCR with word-level positioning
            try:
                # Use Tesseract OCR with configuration for better text extraction
                custom_config = r'--oem 3 --psm 11'
                
                # Get detailed data including word positions
                data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
                
                # Extract words with their positions
                words_with_positions = []
                n_boxes = len(data['text'])
                
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    if text and int(data['conf'][i]) > 0:  # Only include words with confidence > 0
                        words_with_positions.append({
                            'text': text,
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'y_center': data['top'][i] + data['height'][i] / 2
                        })
                
                if not words_with_positions:
                    st.error("No text was detected. Please ensure the image is clear and well-lit.")
                else:
                    # Group words by their vertical position (Y-coordinate) to form lines
                    img_height = image.height
                
                    # Calculate line threshold based on average word height
                    # Use a more generous threshold to ensure words on same line are grouped
                    if len(words_with_positions) > 1:
                        avg_height = sum(w['height'] for w in words_with_positions) / len(words_with_positions)
                        # Use 1.5x average height for more generous grouping
                        line_threshold = max(avg_height * 1.5, 40)  # 150% of average height, minimum 40px
                    else:
                        line_threshold = 50
                
                    # Sort words by Y position (top to bottom)
                    words_sorted = sorted(words_with_positions, key=lambda w: w['y_center'])
                    
                    # Group words into lines
                    lines = []
                    current_line = [words_sorted[0]]
                    current_line_y = words_sorted[0]['y_center']
                    
                    for i in range(1, len(words_sorted)):
                        word = words_sorted[i]
                        y_diff = abs(word['y_center'] - current_line_y)
                        
                        if y_diff <= line_threshold:
                            # Same line - add to current line
                            current_line.append(word)
                            # Update average Y position of the line
                            current_line_y = sum(w['y_center'] for w in current_line) / len(current_line)
                        else:
                            # New line detected
                            # Sort words in current line by X position (left to right)
                            current_line.sort(key=lambda w: w['left'])
                            lines.append(current_line)
                            current_line = [word]
                            current_line_y = word['y_center']
                    
                    # Don't forget the last line
                    if current_line:
                        current_line.sort(key=lambda w: w['left'])
                        lines.append(current_line)
                    
                    # Build formatted output
                    formatted_lines = []
                    for line in lines:
                        # Join words on the same line with spaces
                        line_text = ' '.join([word['text'] for word in line])
                        formatted_lines.append(line_text)
                        doc.add_paragraph(line_text)
                    
                    final_output = '\n'.join(formatted_lines)
                    
                    if final_output.strip():
                        st.success("Text Extracted Successfully!")
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
                        st.error("No text was detected. Please ensure the image is clear and well-lit.")
                    
            except Exception as e:
                st.error(f"Error during OCR processing: {str(e)}")
                st.info("Make sure Tesseract OCR is installed on your system. For Windows, download from: https://github.com/UB-Mannheim/tesseract/wiki")

st.divider()
st.caption("VirtualCo AI Pipeline v2.2 | Using Tesseract OCR for Clear Text Extraction")