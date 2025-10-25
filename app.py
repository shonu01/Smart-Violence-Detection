import cv2
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from torchvision.models.video import R2Plus1D_18_Weights
import numpy as np
import tempfile
import os
import datetime
import pandas as pd
from pymongo import MongoClient

# ---------------------- STREAMLIT CONFIG ----------------------
st.set_page_config(page_title="Violence Detection CCTV", layout="wide")

# ---------------------- LOGIN SYSTEM ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Admin Login")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":  # change for deployment
            st.session_state.logged_in = True
            st.success("✅ Login successful! Loading system...")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.stop()

# ---------------------- MONGODB CONNECTION ----------------------
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["violence_detection"]
    collection = db["detections"]
    st.sidebar.success("🟢 MongoDB Connected")
except Exception as e:
    st.sidebar.error(f"⚠️ MongoDB connection failed: {e}")
    st.stop()

# ---------------------- SIDEBAR NAVIGATION ----------------------
menu = st.sidebar.radio("📋 Menu", ["Live Detection", "View Dashboard", "Logout"])

# ---------------------- LOGOUT OPTION ----------------------
if menu == "Logout":
    st.session_state.logged_in = False
    st.success("👋 Logged out successfully!")
    st.rerun()

# =============================================================
# 🧠 MODEL LOADING
# =============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "violence_model.pth"

@st.cache_resource
def load_model():
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = video_models.r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        st.sidebar.warning("successfully logged in.")

    model = model.to(device)
    model.eval()
    return model

model = load_model()

# =============================================================
# 🎥 FRAME PREPROCESSING FUNCTION
# =============================================================
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (112, 112))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    return frame_tensor

# =============================================================
# 📊 DASHBOARD VIEW
# =============================================================
if menu == "View Dashboard":
    st.title("📊 Violence Detection Dashboard")

    data = list(collection.find().sort("timestamp", -1))
    if not data:
        st.info("No detections found yet.")
        st.stop()

    # Convert MongoDB data to DataFrame
    df = pd.DataFrame(data)
    df["_id"] = df["_id"].astype(str)  # convert ObjectId
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter by date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df["timestamp"].min().date())
    with col2:
        end_date = st.date_input("End Date", df["timestamp"].max().date())

    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    filtered = df.loc[mask]

    st.dataframe(filtered[["timestamp", "source"]])

    st.bar_chart(filtered["source"].value_counts())

    # Download CSV
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Detections as CSV",
        data=csv,
        file_name="violence_detections.csv",
        mime="text/csv"
    )

    st.stop()

# =============================================================
# 🔴 LIVE DETECTION SECTION
# =============================================================
if menu == "Live Detection":
    st.title("🔴 Real-Time Violence Detection – CCTV")

    st.sidebar.header("🎥 Video Source Options")
    source = st.sidebar.selectbox(
        "Choose input source:",
        ["Upload Video", "Webcam (0)", "Sample Video"]
    )

    temp_video_path = None
    if source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("📂 Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded_file.read())
            temp_video.close()
            temp_video_path = temp_video.name
            cap = cv2.VideoCapture(temp_video_path)
        else:
            st.warning("Please upload a video to continue.")
            st.stop()
    elif source == "Webcam (0)":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("sample.mp4")

    frame_window = st.image([])
    st.sidebar.info("🎬 Press STOP button in the toolbar to end stream.")

    frame_buffer = []
    BUFFER_SIZE = 16

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = preprocess_frame(frame)
        frame_buffer.append(frame_tensor)

        if len(frame_buffer) == BUFFER_SIZE:
            clip = torch.stack(frame_buffer)
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                violence_prob = probs[0][1].item()

            label = "Violence" if pred_idx == 1 else "Normal"
            color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
            cv2.putText(frame, f"{label} ({violence_prob:.2f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # ✅ Log detection to MongoDB
            if label == "Violence":
                record = {
                    "timestamp": datetime.datetime.now(),
                    "source": source,
                    "probability": float(violence_prob)
                }
                collection.insert_one(record)

            frame_buffer = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()
    if temp_video_path:
        os.remove(temp_video_path)
