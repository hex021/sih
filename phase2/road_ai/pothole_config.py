import os

# Pothole Model Settings
POTHOLE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "pothole"))
POTHOLE_MODEL_PATH = os.path.join(POTHOLE_MODEL_DIR, "pothole_model.pt")
POTHOLE_CONFIDENCE_THRESHOLD = 0.5
POTHOLE_IOU_THRESHOLD = 0.45

# Deduplication & Validation Settings
POTHOLE_DEDUPLICATION_IOU = 0.15          # Bounding box IoU threshold for matching across frames
POTHOLE_DEDUPLICATION_COOLDOWN_FRAMES = 15 # Frames before considering a tracked pothole "inactive"
POTHOLE_MIN_RELATIVE_Y = 0.35             # Flexible lower-frame / driveable road region threshold (relative)
POTHOLE_MIN_CONFIRMATIONS = 2             # Minimum observations required within temporal window to validate event
POTHOLE_TEMPORAL_WINDOW_FRAMES = 10       # Temporal frame window to accumulate candidate observations

# Storage Locations
EVENTS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "processed", "events", "pothole"))

