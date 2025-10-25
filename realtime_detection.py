import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
import torchvision.transforms as transforms
import numpy as np
from collections import deque
from PIL import Image

# =====================
# Load model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = video_models.r2plus1d_18(weights=None)   # no pretrained weights
model.fc = nn.Linear(model.fc.in_features, 2)    # Fight / NonFight
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# =====================
# Video transforms
# =====================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor()
])

# Keep last 16 frames
frames_queue = deque(maxlen=16)

# =====================
# Realtime webcam
# =====================
cap = cv2.VideoCapture(0)  # 0 = webcam, or replace with "video.mp4"

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor_img = transform(pil_img)
        frames_queue.append(tensor_img)

        # Only predict when we have enough frames
        if len(frames_queue) == 16:
            inputs = torch.stack(list(frames_queue))  # [T, C, H, W]
            inputs = inputs.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
            inputs = inputs.to(device)

            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1).item()

            label = "Fight" if pred == 1 else "NonFight"
            color = (0, 0, 255) if pred == 1 else (0, 255, 0)

            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, color, 3)

        cv2.imshow("Real-Time Violence Detection", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
