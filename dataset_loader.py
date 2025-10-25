import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ------------------------------
# Custom Dataset
# ------------------------------
class ViolenceDataset(Dataset):
    def __init__(self, root_dir, num_frames=32, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []

        # Expect folders: root_dir/Fight/*  and root_dir/NonFight/*
        for label, cls in enumerate(["NonFight", "Fight"]):
            cls_folder = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_folder):
                continue
            for video in os.listdir(cls_folder):
                self.samples.append((os.path.join(cls_folder, video), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)

        if self.transform:
            frames = torch.stack([
                self.transform(Image.fromarray(frame))  # Convert NumPy → PIL → Tensor
                for frame in frames
            ])  # shape: [T, C, H, W]

        return frames, torch.tensor(label, dtype=torch.long)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // self.num_frames)

        for i in range(0, total_frames, step):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # NumPy array
            frames.append(frame)
            if len(frames) == self.num_frames:
                break

        cap.release()

        # If video is too short → pad with last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames


# ------------------------------
# Loader function for Train & Val
# ------------------------------
def get_loaders(root_dir, batch_size=4, num_frames=16, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ViolenceDataset(os.path.join(root_dir, "train"), num_frames, transform)
    val_dataset = ViolenceDataset(os.path.join(root_dir, "val"), num_frames, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
