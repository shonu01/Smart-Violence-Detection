import os
import time
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from torchvision.models.video import R2Plus1D_18_Weights
from torchvision import transforms
from PIL import Image

# =====================
# Settings
# =====================
CHECKPOINT_PATH = "checkpoints/best_model.pth"
SEQ_LEN = 16                # number of frames per clip
SMOOTH_WINDOW = 8           # average this many clip predictions
UPPER_THRESHOLD = 0.55      # detect violence above this
LOWER_THRESHOLD = 0.5      # return to non-violence below this (prevents flicker)
BEEP_COOLDOWN = 2.0         # seconds between beeps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# =====================
# Safe beep (cross-platform)
# =====================
try:
    import winsound
    def beep():
        winsound.Beep(1000, 500)
except Exception:
    def beep():
        print("[ALERT] Violence detected (no sound support)")

# =====================
# Load Model
# =====================
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"❌ Model checkpoint not found: {CHECKPOINT_PATH}")

weights = R2Plus1D_18_Weights.KINETICS400_V1
model = video_models.r2plus1d_18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =====================
# Frame Transform
# =====================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645],
        std=[0.22803, 0.22145, 0.216989]
    )
])

def preprocess_frames(frames):
    """
    Converts a list of BGR frames to a tensor [1, C, T, H, W]
    """
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    frames = [transform(f) for f in pil_frames]
    frames = torch.stack(frames)           # [T, C, H, W]
    frames = frames.permute(1, 0, 2, 3)    # [C, T, H, W]
    return frames.unsqueeze(0)             # [1, C, T, H, W]

# =====================
# Inference Loop
# =====================
cap = cv2.VideoCapture(0)  # use webcam; replace with path for video file
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam. Check camera permissions.")

frame_buffer = []
pred_buffer = deque(maxlen=SMOOTH_WINDOW)
state = "Non-Violence"
last_beep_time = 0.0

print("✅ Real-time violence detection started. Press 'q' to quit.\n")

with torch.no_grad():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(frame)

            # once we have 16 frames, make a prediction
            if len(frame_buffer) >= SEQ_LEN:
                clip = frame_buffer[:SEQ_LEN]
                frame_buffer = frame_buffer[-SEQ_LEN//2:]  # sliding window overlap

                inputs = preprocess_frames(clip).to(DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

                pred_buffer.append(probs)

            # smooth predictions across last few clips
            if len(pred_buffer) > 0:
                avg_probs = np.mean(pred_buffer, axis=0)
                violence_prob = float(avg_probs[1])

                # hysteresis thresholding (avoids flicker)
                if state == "Non-Violence" and violence_prob > UPPER_THRESHOLD:
                    state = "Violence"
                elif state == "Violence" and violence_prob < LOWER_THRESHOLD:
                    state = "Non-Violence"

                # beep if violence detected
                if state == "Violence" and (time.time() - last_beep_time) > BEEP_COOLDOWN:
                    beep()
                    last_beep_time = time.time()

                # display results
                if state == "Violence":
                    label = f"VIOLENCE ({violence_prob:.2f})"
                    color = (0, 0, 255)
                else:
                    label = f"Non-Violence ({1 - violence_prob:.2f})"
                    color = (0, 255, 0)

                cv2.putText(frame, label, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            cv2.imshow("Real-Time Violence Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

cap.release()
cv2.destroyAllWindows()
print("✅ Program terminated safely.")
