import os

# Configuration for Phase 2.2 Still Photo / Image Analysis Module

# Allowed Image File Extensions
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Image Processing Directories
IMAGE_UPLOAD_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
)
IMAGE_PROCESSED_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "processed", "images")
)
PLATE_EVIDENCE_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "processed", "events", "number_plate")
)

os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PLATE_EVIDENCE_FOLDER, exist_ok=True)

# Detection Thresholds
IMAGE_VEHICLE_CONF_THRESHOLD = 0.25
IMAGE_PLATE_CONF_THRESHOLD = 0.25
IMAGE_NMS_IOU_THRESHOLD = 0.45
