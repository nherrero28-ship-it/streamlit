import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(page_title="Notes Transcriber", layout="wide")

st.title("Notes Transcriber")

if os.name == "posix":
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Extract text"):
        with st.spinner("Processing..."):
            img = np.array(image.convert("L"))
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                result = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 3")
            except Exception:
                result = pytesseract.image_to_string(img, lang="eng", config="--psm 3")

            if result.strip():
                st.text_area("Extracted text", value=result.strip(), height=400)
                st.download_button("⬇ Download as TXT", result.strip(), "notes.txt")
            else:
                st.warning("No text found. Try with a clearer image.")
