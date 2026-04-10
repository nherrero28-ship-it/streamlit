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
# Image preprocessing  (FIX 1: better preprocessing)
# ------------------------
def preprocess(img):
    img_array = np.array(img.convert("L"))  # grayscale
    img_array = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Use adaptive thresholding instead of fixed — handles uneven lighting & varied notes
    img_array = cv2.adaptiveThreshold(
        img_array, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    # Mild denoise
    img_array = cv2.medianBlur(img_array, 3)
    return img_array

# ------------------------
# Simple classification
# ------------------------
def classify(text):
    text = text.lower()
    if any(w in text for w in ["theorem", "derivative", "integral", "equation", "algebra"]):
        return "Mathematics"
    elif any(w in text for w in ["energy", "force", "physics", "velocity", "momentum"]):
        return "Physics"
    elif any(w in text for w in ["history", "war", "century", "revolution", "empire"]):
        return "History"
    elif any(w in text for w in ["cell", "biology", "organism", "dna", "protein"]):
        return "Biology"
    else:
        return "General"

# ------------------------
# Upload image
# ------------------------
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="uploader")

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded image")

    if st.button("Extract text", key="extract_btn"):
        with st.spinner("Extracting text..."):
            processed = preprocess(image)
            # FIX 2: use --psm 3 (auto) and oem 3 for best results;
            # keep spa+eng so mixed content works
            lang = "spa+eng"
            try:
                text = pytesseract.image_to_string(
                    processed,
                    lang=lang,
                    config="--psm 3 --oem 3"
                )
            except pytesseract.TesseractError:
                # fallback to eng only if spa lang pack missing
                text = pytesseract.image_to_string(
                    processed,
                    lang="eng",
                    config="--psm 3 --oem 3"
                )
            st.session_state.text = text.strip()

# ------------------------
# FIX 3: Editable text area — use on_change callback to avoid overwrite loop
# ------------------------
def on_text_change():
    st.session_state.text = st.session_state._text_area

st.text_area(
    "Extracted / Editable Text",
    value=st.session_state.text,
    height=300,
    key="_text_area",
    on_change=on_text_change
)

# ------------------------
# Actions
# ------------------------
col1, col2, col3 = st.columns(3)

# Save
with col1:
    st.markdown("**Save document**")
    name = st.text_input("Document name", key="doc_name_input")  # FIX 4: explicit key
    if st.button("Save", key="save_btn"):
        if name and st.session_state.text:
            category = classify(st.session_state.text)
            st.session_state.docs[name] = {
                "content": st.session_state.text,
                "category": category
            }
            st.success(f"Saved under **{category}**")
        elif not name:
            st.warning("Enter a document name first.")
        else:
            st.warning("No text to save.")

# Translate
with col2:
    st.markdown("**Translate text**")
    lang = st.selectbox("Translate to", ["en", "es", "fr", "de", "it"], key="lang_select")
    if st.button("Translate", key="translate_btn"):
        if st.session_state.text:
            with st.spinner("Translating..."):
                try:
                    translated = GoogleTranslator(
                        source="auto",
                        target=lang
                    ).translate(st.session_state.text)
                    st.session_state.text = translated
                    st.rerun()  # FIX 5: force UI refresh after state change
                except Exception as e:
                    st.error(f"Translation error: {e}")  # FIX 6: show real error
        else:
            st.warning("No text to translate.")

# Download
with col3:
    st.markdown("**Export**")
    if st.session_state.text:
        st.download_button(
            "⬇ Download as TXT",
            st.session_state.text,
            "notes.txt",
            key="download_btn"
        )

# ------------------------
# Search — FIX 7: unique key so it doesn't conflict with doc_name_input
# ------------------------
st.subheader("Search documents")
search = st.text_input("Type to search", key="search_input")

# ------------------------
# Saved documents — FIX 8: use radio/selectbox instead of buttons in loop
# ------------------------
st.subheader("Saved documents")

if st.session_state.docs:
    filtered = {
        n: d for n, d in st.session_state.docs.items()
        if not search
        or search.lower() in n.lower()
        or search.lower() in d["content"].lower()
    }

    if filtered:
        for name, data in filtered.items():
            with st.expander(f"📄 {name}  —  *{data['category']}*"):
                st.text(data["content"][:500] + ("..." if len(data["content"]) > 500 else ""))
                if st.button(f"Load '{name}'", key=f"load_{name}"):  # FIX 9: unique key per button
                    st.session_state.text = data["content"]
                    st.rerun()
    else:
        st.info("No documents match your search.")
else:
    st.info("No saved documents yet.")

# ------------------------
# Clear
# ------------------------
if st.button("🗑 Clear text", key="clear_btn"):
    st.session_state.text = ""
    st.rerun()
