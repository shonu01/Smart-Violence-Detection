# db.py
from pymongo import MongoClient
import datetime

# Connect to local MongoDB (default)
client = MongoClient("mongodb://localhost:27017/")

# Database & collection
db = client["violence_detection"]
detections = db["detections"]

def log_detection(source, label, confidence):
    """Insert a detection event into MongoDB."""
    detections.insert_one({
        "timestamp": datetime.datetime.now(),
        "source": source,
        "label": label,
        "confidence": confidence
    })

def get_recent_detections(limit=20):
    """Fetch recent detection events."""
    return list(detections.find().sort("timestamp", -1).limit(limit))
