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

file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    if st.button("Extraer texto"):
        with st.spinner("Procesando..."):
            img = np.array(image.convert("L"))
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                resultado = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 3")
            except Exception:
                resultado = pytesseract.image_to_string(img, lang="eng", config="--psm 3")

            if resultado.strip():
                st.text_area("Texto extraído", value=resultado.strip(), height=400)
                st.download_button("⬇ Descargar como TXT", resultado.strip(), "notas.txt")
            else:
                st.warning("No se encontró texto. Prueba con otra imagen.")
