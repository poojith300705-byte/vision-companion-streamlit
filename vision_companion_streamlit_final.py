import streamlit as st
import cv2
import tempfile
import numpy as np
import pytesseract
from gtts import gTTS
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox

st.set_page_config(page_title="Vision Companion", layout="wide")

st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode", ["Object Detection", "Text Recognition (OCR)"])

st.title("👁️ Vision Companion")
st.markdown("Real-time Object & Text Detection with Voice Output (powered by gTTS)")

uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img = cv2.imread(tfile.name)

    if mode == "Object Detection":
        st.subheader("🔍 Object Detection Mode")
        bbox, label, conf = cv.detect_common_objects(img)
        output_image = draw_bbox(img, bbox, label, conf)

        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)
        detected_objects = ", ".join(label)
        st.write(f"**Detected:** {detected_objects}")

        if st.button("🔊 Speak Objects"):
            if detected_objects:
                tts = gTTS(text=f"I can see {detected_objects}", lang='en')
                tts.save("detected.mp3")
                st.audio("detected.mp3", format="audio/mp3")
            else:
                st.warning("No objects detected.")

    elif mode == "Text Recognition (OCR)":
        st.subheader("📝 Text Recognition Mode")
        text = pytesseract.image_to_string(img)
        st.text_area("Detected Text", text, height=200)

        if st.button("🔊 Read Text"):
            if text.strip():
                tts = gTTS(text=text, lang='en')
                tts.save("text.mp3")
                st.audio("text.mp3", format="audio/mp3")
            else:
                st.warning("No text detected.")
else:
    st.info("👆 Upload an image to begin analysis.")
