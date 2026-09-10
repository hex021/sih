import os

# Centralized Settings for Phase 2.2 — ANPR & Vehicle Identification

# Plate Detection Thresholds
PLATE_CONF_THRESHOLD = 0.25
PLATE_MIN_AREA_RATIO = 0.0005  # Min relative area of plate inside vehicle box
PLATE_MAX_AREA_RATIO = 0.35    # Max relative area of plate inside vehicle box

# Spatial Association Thresholds
MIN_PLATE_VEHICLE_IOU = 0.05
STRICT_CONTAINMENT_REQUIRED = True

# Phase B & C Parameters
DEFAULT_TOP_N_VEHICLES = 3     # Only attempt OCR on the largest N vehicle boxes per frame
PLATE_MIN_ASPECT_RATIO = 2.0   # Min aspect ratio (width:height) for Indian plates
PLATE_MAX_ASPECT_RATIO = 5.0   # Max aspect ratio (width:height) for Indian plates
PREPROC_UPSCALE_FACTOR = 4     # 4x upscaling with INTER_CUBIC

# OCR Engine Settings
OCR_CONF_THRESHOLD = 0.40      # Minimum confidence for validated plate (C3)
MAX_OCR_CANDIDATES_PER_TRACK = 20
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# RegEx Patterns for Indian Vehicle Registration Numbers (C2)
# Standard: e.g. GJ01AB1234, MH12A5678
INDIAN_STANDARD_PATTERN = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
# Bharat (BH) series: e.g. 22BH1234AA
INDIAN_BHARAT_PATTERN = r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"
# Backward-compatible alias
INDIAN_PLATE_PATTERN = INDIAN_STANDARD_PATTERN

# Preprocessing Settings
PREPROC_RESIZE_WIDTH = 320
PREPROC_RESIZE_HEIGHT = 96
USE_CLAHE = True
USE_SHARPENING = False

# Output Evidence Storage Path
ANPR_EVENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "processed", "events", "vehicle")
)
os.makedirs(ANPR_EVENTS_DIR, exist_ok=True)

# Debug Crops Directory (Phase B: save both raw and preprocessed crops)
OCR_DEBUG_CROPS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "processed", "crops")
)
os.makedirs(OCR_DEBUG_CROPS_DIR, exist_ok=True)

