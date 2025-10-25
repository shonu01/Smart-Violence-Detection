import torch
import matplotlib.pyplot as plt
import os

# Path to your training log
log_path = "checkpoints/training_log.pth"
save_dir = "checkpoints/curves"
os.makedirs(save_dir, exist_ok=True)

# Load logs
log = torch.load(log_path, weights_only=True)

train_losses = log["train_losses"]
val_losses = log["val_losses"]
train_accuracies = log["train_accuracies"]
val_accuracies = log["val_accuracies"]

epochs = range(1, len(train_losses) + 1)

# ===== Plot Loss =====
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Train Loss", marker="o")
plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.close()

# ===== Plot Accuracy =====
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
plt.close()

print(f"Plots saved in {save_dir}")
