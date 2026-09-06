import os

# Centralized Settings for Phase 2.2 — ANPR & Vehicle Identification

# Plate Detection Thresholds
PLATE_CONF_THRESHOLD = 0.25
PLATE_MIN_AREA_RATIO = 0.0005  # Min relative area of plate inside vehicle box
PLATE_MAX_AREA_RATIO = 0.35    # Max relative area of plate inside vehicle box

# Spatial Association Thresholds
MIN_PLATE_VEHICLE_IOU = 0.05
STRICT_CONTAINMENT_REQUIRED = True

# OCR Engine Settings
OCR_CONF_THRESHOLD = 0.40
MAX_OCR_CANDIDATES_PER_TRACK = 20

# RegEx Patterns for Indian Vehicle Registration Numbers
# Format e.g., GJ01AB1234, MH12CD5678, DL01A1234, KA05M9999
INDIAN_PLATE_PATTERN = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$"

# Preprocessing Settings
PREPROC_RESIZE_WIDTH = 320
PREPROC_RESIZE_HEIGHT = 96
USE_CLAHE = True
USE_SHARPENING = True

# Output Evidence Storage Path
ANPR_EVENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "processed", "events", "vehicle")
)
os.makedirs(ANPR_EVENTS_DIR, exist_ok=True)
