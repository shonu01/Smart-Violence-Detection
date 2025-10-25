# fake_model.py
import numpy as np

class FakeViolenceDetector:
    """Stub that generates random violence probabilities."""
    def predict(self, frames):
        # Pretend we analyze a batch of frames
        prob = np.random.rand()  # random number [0,1]
        return prob, "Violent" if prob > 0.6 else "Non-Violent"
