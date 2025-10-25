# evaluate.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import ViolenceDataset
from torchvision import transforms
import torchvision.models.video as video_models
from torchvision.models.video import R2Plus1D_18_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# Paths
# =====================
DATASET_PATH = r"E:\RWF-2000\RWF-2000\val"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# =====================
# Dataset (Validation only)
# =====================
frame_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor()
])

val_dataset = ViolenceDataset(DATASET_PATH, transform=frame_transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

# =====================
# Model
# =====================
weights = R2Plus1D_18_Weights.KINETICS400_V1
model = video_models.r2plus1d_18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)  # Fight / NonFight
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()

# =====================
# Evaluation
# =====================
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        # [B, T, C, H, W] -> [B, C, T, H, W]
        inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =====================
# Metrics
# =====================
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Non-Violence", "Violence"]))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Violence", "Violence"],
            yticklabels=["Non-Violence", "Violence"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
os.makedirs("checkpoints/curves", exist_ok=True)
plt.savefig("checkpoints/curves/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to checkpoints/curves/confusion_matrix.png")
