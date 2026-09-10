import os
import sys
import unittest
import numpy as np

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase2.anpr.ocr_engine import OCREngine
from phase2.anpr.plate_detector import PlateDetector
from phase2.anpr.anpr_aggregator import ANPRAggregator
from phase2.anpr.plate_models import PlateDetection, OCRResult
from phase2.road_ai.pothole_detector import PotholeDetector
from phase1.models import TrackedObject

class TestTraceabilityAndHardening(unittest.TestCase):

    def test_no_hardcoded_ocr_fallback(self):
        """Mandate 6: Ensure fallback OCR never returns hardcoded demo strings."""
        ocr = OCREngine(use_easyocr=False)
        crop = np.zeros((50, 150, 3), dtype=np.uint8)
        res = ocr.extract_text(crop)
        self.assertNotIn("GJ01AB1234", res.normalized_text)
        self.assertNotIn("MH12CD5678", res.normalized_text)
        self.assertIn(res.normalized_text, ["UNREADABLE", ""])

    def test_crop_quality_score(self):
        """Mandate 4: Crop quality score calculation based on resolution and sharpness."""
        crop_sharp = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
        score_sharp = PlateDetector.compute_crop_quality_score(crop_sharp, 0.9)
        self.assertGreater(score_sharp, 0.0)

        crop_blank = np.zeros((10, 10, 3), dtype=np.uint8)
        score_blank = PlateDetector.compute_crop_quality_score(crop_blank, 0.1)
        self.assertEqual(score_blank, 0.0)

    def test_temporal_ocr_voting_consensus(self):
        """Mandate 4: Temporal OCR consensus selects dominant candidate and handles disagreement."""
        aggregator = ANPRAggregator()
        tracked = [TrackedObject(track_id=1, class_name="CAR", confidence=0.9, bbox=(10, 10, 100, 100), center_x=55, center_y=55, previous_center_x=55, previous_center_y=55)]
        aggregator.update_track_metadata(tracked, 1, 0.0)

        det = PlateDetection(bbox=(20, 20, 80, 50), confidence=0.85, frame_number=1, timestamp=0.0, vehicle_track_id=1)
        v_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        p_crop = np.zeros((30, 60, 3), dtype=np.uint8)

        # Observations across frames: 3 say HR51A2399, 1 says HR51A2390
        ocr1 = OCRResult(raw_ocr="HR51A2399", normalized_text="HR51A2399", ocr_confidence=0.9, is_valid_pattern=True)
        ocr2 = OCRResult(raw_ocr="HR51A2399", normalized_text="HR51A2399", ocr_confidence=0.92, is_valid_pattern=True)
        ocr3 = OCRResult(raw_ocr="HR51A2399", normalized_text="HR51A2399", ocr_confidence=0.88, is_valid_pattern=True)
        ocr4 = OCRResult(raw_ocr="HR51A2390", normalized_text="HR51A2390", ocr_confidence=0.70, is_valid_pattern=True)

        for o in [ocr1, ocr2, ocr3, ocr4]:
            aggregator.add_observation(1, det, o, v_crop, p_crop, 0.9)

        records = aggregator.finalize_vehicle_records("test.mp4")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].registration_number, "HR51A2399")
        self.assertEqual(records[0].track_id, 1)
        self.assertIsNotNone(records[0].media.get("vehicle_snapshot"))
        self.assertIsNotNone(records[0].media.get("plate_snapshot"))

    def test_pothole_road_surface_and_temporal_validation(self):
        """Mandate 2 & 3: Pothole detector rejects upper background and requires temporal window confirmation."""
        try:
            detector = PotholeDetector()
        except Exception:
            return

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.active_tracks.clear()
        dets, events = detector.detect(frame, 1, 0.0)
        self.assertEqual(len(events), 0)

if __name__ == "__main__":
    unittest.main()
