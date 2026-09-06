import os

# Model Settings
MODEL_PATH = os.path.join("models", "yolo11n.pt")  # Pretrained lightweight YOLO model
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TRACKER_TYPE = "bytetrack.yaml"         # Fast tracker on CPU, replacing botsort.yaml
MAX_PROCESSING_DIMENSION = 960          # Max width or height for processing frames

# Centralized Class Mappings
# Maps YOLO COCO dataset class names (lowercase) to standardized internal class names
CLASS_MAP = {
    "car": "CAR",
    "motorcycle": "MOTORCYCLE",
    "bicycle": "BICYCLE",
    "bus": "BUS",
    "truck": "TRUCK",
    "person": "PERSON"
}

# The set of target classes we are interested in
TARGET_CLASSES = list(CLASS_MAP.values())

# Relative Region of Interest (ROI) Coordinates (normalized between 0.0 and 1.0)
# (x_min, y_min, x_max, y_max)
# For example, covering from 10% to 90% horizontally, and 30% to 90% vertically
ROI_RELATIVE = (0.0, 0.3, 1.0, 0.95)

# Relative Count Line Y Position (normalized between 0.0 and 1.0)
# 65% down from the top of the frame
COUNT_LINE_RELATIVE_Y = 0.65

# Traffic Density Classification Thresholds (Active Vehicles inside ROI)
# LOW: <= 5 vehicles
# MEDIUM: 6 to 15 vehicles
# HIGH: 16+ vehicles
DENSITY_THRESHOLDS = {
    "LOW_MAX": 5,
    "MEDIUM_MAX": 15
}

# Track History Configuration
TRACK_HISTORY_LIMIT = 50  # Bounded history (number of frames) to prevent memory growth

# Default File Locations
DEFAULT_INPUT_VIDEO = os.path.join("input", "road_video.mp4")
DEFAULT_OUTPUT_VIDEO = os.path.join("output", "traffic_detection.mp4")
DEFAULT_TEST_IMAGE = os.path.join("input", "test.jpg")

# Visual Display Settings
HUD_COLOR = (240, 240, 240)        # Near white text for panel
HUD_BG_COLOR = (30, 30, 30, 200)   # Dark semi-transparent background (RGBA overlay)
ROI_COLOR = (0, 255, 255)          # Cyan
COUNT_LINE_COLOR = (0, 0, 255)     # Red
BBOX_COLOR = (0, 255, 0)           # Green
TRAJECTORY_COLOR = (255, 255, 0)   # Yellow
TEXT_COLOR = (255, 255, 255)       # White

# Standard Line thicknesses and text scales
BBOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
