import streamlit as st
import cv2
import pytesseract
from gtts import gTTS
import tempfile
import os
from PIL import Image
import numpy as np

# App title
st.set_page_config(page_title="Vision Companion", layout="wide")
st.title("👁️ Vision Companion")
st.markdown("Real-time Object & Text Detection with Voice Output (powered by gTTS)")

# Sidebar options
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode", ["Object Detection", "Text Recognition (OCR)"])

# Load OpenCV’s DNN model for object detection
prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(prototxt) or not os.path.exists(model):
    st.error("❌ Model files missing! Please ensure 'MobileNetSSD_deploy.prototxt.txt' and 'MobileNetSSD_deploy.caffemodel' exist in the same folder.")
    st.stop()

net = cv2.dnn.readNetFromCaffe(prototxt, model)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Helper: speak text with gTTS
def speak_text(text):
    if not text:
        return
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Speech generation failed: {e}")

# Helper: detect objects
def detect_objects(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    detected_objects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            detected_objects.append(label)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, list(set(detected_objects))

# Helper: read text
def read_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Upload image
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if mode == "Object Detection":
        st.subheader("🧠 Object Detection Results")
        processed_image, objects = detect_objects(image.copy())
        st.image(processed_image, caption="Detected Objects", use_container_width=True)

        if objects:
            object_text = ", ".join(objects)
            st.success(f"Detected: {object_text}")
            speak_text(f"I found {object_text}")
        else:
            st.info("No objects detected.")
            speak_text("I didn't detect any objects.")

    elif mode == "Text Recognition (OCR)":
        st.subheader("🔤 Text Recognition Results")
        text = read_text(image)
        if text:
            st.success(f"Detected text:\n{text}")
            speak_text(f"The text says: {text}")
        else:
            st.info("No readable text detected.")
            speak_text("No readable text found.")
else:
    st.info("👆 Please upload an image to begin.")
