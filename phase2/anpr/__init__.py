# Phase 2.2 — ANPR, OCR & Vehicle Identification Module
from .plate_models import PlateDetection, OCRResult, VehicleRecord
from .plate_detector import PlateDetector
from .ocr_engine import OCREngine
from .anpr_aggregator import ANPRAggregator
from .anpr_pipeline import run_anpr_pipeline
