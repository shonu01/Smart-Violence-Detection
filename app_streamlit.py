import cv2
import streamlit as st
from preprocess import resize_normalize
from fake_model import FakeViolenceDetector

st.set_page_config(page_title="Violence Detection CCTV", layout="wide")
st.title("🔴 Smart Violence Detection – CCTV")

source = st.selectbox("Video Source", ["Webcam (0)", "Sample Video"])

if source == "Webcam (0)":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("sample.mp4")

frame_window = st.image([])
detector = FakeViolenceDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = resize_normalize(frame)

    # Fake prediction
    prob, label = detector.predict([img])

    # Convert for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show with prediction overlay
    frame_window.image(frame_rgb, caption=f"Prediction: {label} ({prob:.2f})")
