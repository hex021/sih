import os

# Pothole Model Settings
POTHOLE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "pothole"))
POTHOLE_MODEL_PATH = os.path.join(POTHOLE_MODEL_DIR, "pothole_model.pt")
POTHOLE_CONFIDENCE_THRESHOLD = 0.25
POTHOLE_IOU_THRESHOLD = 0.45

# Deduplication Settings
POTHOLE_DEDUPLICATION_IOU = 0.15          # Bounding box IoU threshold for matching across frames
POTHOLE_DEDUPLICATION_COOLDOWN_FRAMES = 15 # Frames before considering a tracked pothole "inactive"

# Storage Locations
EVENTS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "processed", "events", "pothole"))
