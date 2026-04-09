import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
from deep_translator import GoogleTranslator
import os

# ------------------------
# Page config + styling
# ------------------------
st.set_page_config(page_title="Notes Transcriber", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    textarea, input {
        background-color: #1e293b !important;
        color: white !important;
    }
    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Notes Transcriber")

# ------------------------
# Tesseract config (local + cloud)
# ------------------------
if os.name == "posix":
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ------------------------
# Session state
# ------------------------
if "text" not in st.session_state:
    st.session_state.text = ""

if "docs" not in st.session_state:
    st.session_state.docs = {}

# ------------------------
# Image preprocessing
# ------------------------
def preprocess(img):
    img = np.array(img.convert("L"))
    img = cv2.resize(img, None, fx=2, fy=2)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return img

# ------------------------
# Simple classification
# ------------------------
def classify(text):
    text = text.lower()

    if any(w in text for w in ["theorem", "derivative", "integral"]):
        return "Mathematics"
    elif any(w in text for w in ["energy", "force", "physics"]):
        return "Physics"
    elif any(w in text for w in ["history", "war", "century"]):
        return "History"
    elif any(w in text for w in ["cell", "biology", "organism"]):
        return "Biology"
    else:
        return "General"

# ------------------------
# Upload image
# ------------------------
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded image")

    if st.button("Extract text"):
        processed = preprocess(image)

        text = pytesseract.image_to_string(
            processed,
            lang="spa",
            config="--psm 6"
        )

        st.session_state.text = text

# ------------------------
# Editable text area
# ------------------------
st.session_state.text = st.text_area(
    "Text",
    st.session_state.text,
    height=300
)

# ------------------------
# Actions
# ------------------------
col1, col2, col3 = st.columns(3)

# Save
with col1:
    name = st.text_input("Document name")
    if st.button("Save"):
        if name:
            category = classify(st.session_state.text)
            st.session_state.docs[name] = {
                "content": st.session_state.text,
                "category": category
            }
            st.success(f"Saved under {category}")

# Translate
with col2:
    lang = st.selectbox("Translate to", ["en", "es", "fr", "de", "it"])
    if st.button("Translate"):
        try:
            translated = GoogleTranslator(
                source="auto",
                target=lang
            ).translate(st.session_state.text)

            st.session_state.text = translated
        except:
            st.error("Translation error")

# Download
with col3:
    if st.session_state.text:
        st.download_button(
            "Download as TXT",
            st.session_state.text,
            "notes.txt"
        )

# ------------------------
# Search
# ------------------------
st.subheader("Search documents")
search = st.text_input("Type to search")

# ------------------------
# Stored documents
# ------------------------
st.subheader("Saved documents")

for name, data in st.session_state.docs.items():
    content = data["content"]
    category = data["category"]

    if search.lower() in name.lower() or search.lower() in content.lower():
        if st.button(f"{name} ({category})"):
            st.session_state.text = content

# ------------------------
# Clear
# ------------------------
if st.button("Clear text"):
    st.session_state.text = ""

