import torch
from dataset_loader import get_loaders
from model import CNNLSTM  # use CNNLSTM instead of ViolenceModel

# Correct dataset path (update if your dataset path is different)
DATASET_PATH = "E:/RWF-2000/RWF-2000"

def main():
    # Load data
    train_loader, val_loader = get_loaders(DATASET_PATH, batch_size=2)

    # Load model
    model = CNNLSTM(num_classes=2)

    # Quick check on one batch
    for images, labels in train_loader:
        print("Batch shape:", images.shape)   # (batch, seq_len, 3, H, W)
        print("Labels:", labels)
        break

if __name__ == "__main__":
    main()