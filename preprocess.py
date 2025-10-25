# preprocess.py
import cv2
import numpy as np

def resize_normalize(frame, size: int = 224):
    """Resize and normalize a video frame."""
    img = cv2.resize(frame, (size, size))      # Resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR → RGB
    img = img.astype(np.float32) / 255.0       # Normalize [0,1]
    return img
