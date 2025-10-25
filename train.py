import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import ViolenceDataset
from torchvision import transforms
import torchvision.models.video as video_models
from torchvision.models.video import R2Plus1D_18_Weights
from sklearn.metrics import precision_score, recall_score, f1_score

# =====================
# Reproducibility & Speed
# =====================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # speed up for fixed-size inputs

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# Training settings
# =====================
epochs = 30
batch_size = 2     # keep small (video models are heavy)
learning_rate = 1e-4
accumulation_steps = 1  # set >1 if you want gradient accumulation

# =====================
# Dataset + transforms
# (frames arrive as NumPy, dataset converts to PIL; these ops are per-frame)
# =====================
frame_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

train_dataset = ViolenceDataset(r"E:\RWF-2000\RWF-2000\train", transform=frame_transform)
val_dataset   = ViolenceDataset(r"E:\RWF-2000\RWF-2000\val",   transform=frame_transform)

# Tune workers for your machine. On Windows, start with 0–2. On Linux, try 4–8.
NUM_WORKERS = 0 if os.name == "nt" else 4

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), drop_last=False
)

# =====================
# Model: r2plus1d_18
# =====================
weights = R2Plus1D_18_Weights.KINETICS400_V1
model = video_models.r2plus1d_18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)  # Fight / NonFight
model = model.to(device)

# =====================
# Loss, Optimizer, Scheduler, AMP scaler
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ✅ Updated GradScaler (new API)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# =====================
# Logging containers
# =====================
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0

# =====================
# Training Loop
# =====================
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for step, (inputs, labels) in enumerate(train_loader, start=1):
        # inputs: [B, T, C, H, W] -> [B, C, T, H, W]
        inputs = inputs.permute(0, 2, 1, 3, 4)
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ✅ Updated autocast (new API)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item() * inputs.size(0)  # sum over batch
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_epoch_loss = running_loss / max(total, 1)
    train_acc = correct / max(total, 1)
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_acc)
    print(f"Epoch {epoch}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_acc:.4f}")

    # =====================
    # Validation
    # =====================
    model.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []
    val_loss_sum = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.permute(0, 2, 1, 3, 4)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(inputs)
                vloss = criterion(outputs, labels)

            val_loss_sum += vloss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    val_epoch_loss = val_loss_sum / max(val_total, 1)
    val_acc = val_correct / max(val_total, 1)

    # Handle edge cases where a class might be missing in a small batch/epoch
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1        = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_acc)

    print(f"Validation | Loss: {val_epoch_loss:.4f} | Acc: {val_acc:.4f} | "
          f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    # =====================
    # Save best model
    # =====================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("checkpoints", exist_ok=True)
        best_ckpt = "checkpoints/best_model.pth"
        torch.save(model.state_dict(), best_ckpt)
        print(f"Best model saved at: {best_ckpt}")

    scheduler.step()

# =====================
# Save training curves/log
# =====================
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}, "checkpoints/training_log.pth")
print("Training log saved to checkpoints/training_log.pth")
