import streamlit as st
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
from ultralytics import YOLO
import tempfile
import os

# Load YOLOv8 model (downloads automatically)
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="Vision Companion", layout="wide")
st.title("🦾 Vision Companion")
st.write("Helping visually impaired people detect objects and read text aloud.")

# Sidebar controls
mode = st.sidebar.radio("Choose Mode:", ["🔍 Object Detection", "🔤 Text Reader"])
voice_speed = st.sidebar.radio("Voice Speed", ["Normal", "Slow"])
st.sidebar.markdown("---")
st.sidebar.info("Upload an image or use the webcam to get started!")

# File upload or webcam
source = st.radio("Choose input source:", ["📁 Upload Image", "📷 Webcam"])

def speak_text(text, slow=False):
    if not text.strip():
        text = "No readable text detected."
    tts = gTTS(text=text, lang='en', slow=slow)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_path.name)
    st.audio(temp_path.name)
    os.remove(temp_path.name)

if source == "📁 Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if mode == "🔍 Object Detection":
            results = model(img)
            annotated = results[0].plot()
            st.image(annotated, channels="BGR", use_container_width=True)

            detected_objects = [model.names[int(c)] for c in results[0].boxes.cls]
            if detected_objects:
                st.success(f"Detected: {', '.join(set(detected_objects))}")
                speak_text(f"I detected {', '.join(set(detected_objects))}.",
                           slow=(voice_speed == "Slow"))
            else:
                st.warning("No objects detected.")

        elif mode == "🔤 Text Reader":
            text = pytesseract.image_to_string(img)
            st.text_area("Detected Text:", text, height=200)
            speak_text(text, slow=(voice_speed == "Slow"))

elif source == "📷 Webcam":
    st.warning("Webcam input is not supported on Streamlit Cloud. Try uploading an image instead.")
