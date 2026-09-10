import os
import sys
import unittest
import numpy as np

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1.models import TrackedObject
from phase2.anpr import (
    PlateDetector,
    OCREngine,
    ANPRAggregator,
    PlateDetection,
    OCRResult,
    VehicleRecord
)
from backend.server import app

class TestPhase22PlateDetectionAndSpatialAssociation(unittest.TestCase):
    def setUp(self):
        self.detector = PlateDetector()

    def test_spatial_vehicle_plate_association(self):
        """Test spatial containment check between plate bbox and vehicle bbox."""
        vehicle_box = (100.0, 100.0, 400.0, 400.0)
        
        # 1. Plate center inside vehicle bbox -> Associated
        inside_plate = (150.0, 250.0, 250.0, 300.0)
        self.assertTrue(self.detector.is_plate_inside_vehicle(inside_plate, vehicle_box))
        
        # 2. Plate completely outside vehicle bbox -> Not associated
        outside_plate = (600.0, 600.0, 700.0, 700.0)
        self.assertFalse(self.detector.is_plate_inside_vehicle(outside_plate, vehicle_box))

    def test_locate_plate_region_in_crop(self):
        """Test plate region localization on a synthetic vehicle crop image."""
        # Create a synthetic 100x200 crop with a rectangular plate-like contour
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        # Draw a white rectangle representing number plate in lower region
        cv2 = __import__("cv2")
        cv2.rectangle(crop, (40, 60), (160, 90), (255, 255, 255), -1)
        
        box, conf = self.detector._locate_plate_region_in_crop(crop)
        self.assertIsNotNone(box)
        self.assertGreater(conf, 0.5)

class TestPhase22OCREngineAndIndianPlateNormalization(unittest.TestCase):
    def setUp(self):
        self.ocr = OCREngine(use_easyocr=False)

    def test_indian_plate_normalization_standard(self):
        """Test normalization and correction of Standard Indian registration plates (C1 & C2)."""
        # Case 1: Raw string with spaces -> Normalized
        normalized, valid = self.ocr.normalize_indian_plate(" GJ 01 AB 1234 ")
        self.assertEqual(normalized, "GJ01AB1234")
        self.assertTrue(valid)

        # Case 2: State prefix digit-to-letter correction ('0' -> 'O')
        normalized2, valid2 = self.ocr.normalize_indian_plate("0J 01 AB 1234")
        self.assertEqual(normalized2, "OJ01AB1234")
        self.assertTrue(valid2)

        # Case 3: Letter-to-digit correction in district code ('O' -> '0')
        normalized3, valid3 = self.ocr.normalize_indian_plate("GJ O1 AB 1234")
        self.assertEqual(normalized3, "GJ01AB1234")
        self.assertTrue(valid3)

        # Case 4: Standard format with single digit district and 1 series letter: MH12A5678
        normalized4, valid4 = self.ocr.normalize_indian_plate("MH 12 A 5678")
        self.assertEqual(normalized4, "MH12A5678")
        self.assertTrue(valid4)

        # Case 5: Position-aware: in digit position, map O->0, I->1, S->5, B->8, Z->2, G->6
        normalized5, valid5 = self.ocr.normalize_indian_plate("DL 01 AB I2S8")
        self.assertEqual(normalized5, "DL01AB1258")
        self.assertTrue(valid5)

        # Case 6: In letter position (state / series), map 0->O, 1->I, 5->S, 8->B, 2->Z, 6->G
        normalized6, valid6 = self.ocr.normalize_indian_plate("GJ 01 5B 1234")
        self.assertEqual(normalized6, "GJ01SB1234")
        self.assertTrue(valid6)

    def test_indian_plate_normalization_bharat_series(self):
        """Test Bharat (BH) series plate validation and position-aware correction (C2)."""
        # Case 1: Direct Bharat format
        normalized, valid = self.ocr.normalize_indian_plate("22 BH 1234 AA")
        self.assertEqual(normalized, "22BH1234AA")
        self.assertTrue(valid)

        # Case 2: Letter to digit in year code ('ZZ' -> '22') and '8' -> 'B' in BH
        normalized2, valid2 = self.ocr.normalize_indian_plate("ZZ 8H 1234 AA")
        self.assertEqual(normalized2, "22BH1234AA")
        self.assertTrue(valid2)

        # Case 3: Single letter series Bharat plate
        normalized3, valid3 = self.ocr.normalize_indian_plate("23 BH 5678 A")
        self.assertEqual(normalized3, "23BH5678A")
        self.assertTrue(valid3)

    def test_c3_reject_rather_than_guess(self):
        """Test that invalid format or low confidence strings are rejected, not fabricated (C3)."""
        # Invalid format string
        _, valid = self.ocr.normalize_indian_plate("INVALIDTEXT123")
        self.assertFalse(valid)

        # Crop that produces no text or is rejected
        crop = np.zeros((48, 160, 3), dtype=np.uint8)
        res = self.ocr.extract_text(crop)
        self.assertIsInstance(res, OCRResult)
        self.assertIsNone(res.plate)
        self.assertIn(res.plate_status, ["no_text", "format_rejected", "low_confidence"])

    def test_c5_aspect_ratio_pre_filter(self):
        """Test aspect ratio pre-filtering: 2:1 to 5:1 (width:height) (C5)."""
        # 1. Aspect ratio 3.0 (valid: 150w x 50h)
        crop_valid = np.zeros((50, 150, 3), dtype=np.uint8)
        self.assertTrue(self.ocr.check_aspect_ratio(crop_valid, min_aspect_ratio=2.0, max_aspect_ratio=5.0))

        # 2. Aspect ratio 1.0 (square, invalid: 50w x 50h)
        crop_square = np.zeros((50, 50, 3), dtype=np.uint8)
        self.assertFalse(self.ocr.check_aspect_ratio(crop_square, min_aspect_ratio=2.0, max_aspect_ratio=5.0))

        # 3. Aspect ratio 6.0 (too wide, invalid: 180w x 30h)
        crop_too_wide = np.zeros((30, 180, 3), dtype=np.uint8)
        self.assertFalse(self.ocr.check_aspect_ratio(crop_too_wide, min_aspect_ratio=2.0, max_aspect_ratio=5.0))

    def test_phase_b_preprocessing_4x_and_clahe(self):
        """Test Phase B 4x upscaling and CLAHE greyscale preprocessing."""
        crop = np.zeros((20, 60, 3), dtype=np.uint8)
        preproc = self.ocr.preprocess_plate_crop(crop)
        # Dimensions must be exactly 4x: (20*4, 60*4) = (80, 240)
        self.assertEqual(preproc.shape[:2], (80, 240))
        # Must be single channel greyscale
        self.assertEqual(len(preproc.shape), 2)


class TestPhase22MultiFrameAggregatorAndDeduplication(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.aggregator = ANPRAggregator(output_events_dir=self.temp_dir)

    def test_multi_frame_aggregation_and_single_record(self):
        """Test that multiple frames for track #27 produce exactly ONE primary VehicleRecord with C4 fields."""
        # Track 27 metadata
        veh = TrackedObject(27, "CAR", 0.92, (100, 100, 300, 300), 200, 200, 200, 200)
        self.aggregator.update_track_metadata([veh], frame_number=100, timestamp=3.3)

        det1 = PlateDetection((120, 120, 180, 150), 0.85, 100, 3.3, 27)
        ocr1 = OCRResult(raw_ocr="GJ 01 AB 1234", normalized_text="GJ01AB1234", plate="GJ01AB1234",
                         ocr_confidence=0.75, plate_confidence=0.75, plate_status="validated", is_valid_pattern=True)

        det2 = PlateDetection((120, 120, 180, 150), 0.90, 105, 3.5, 27)
        ocr2 = OCRResult(raw_ocr="GJ 01 AB 1234", normalized_text="GJ01AB1234", plate="GJ01AB1234",
                         ocr_confidence=0.91, plate_confidence=0.91, plate_status="validated", is_valid_pattern=True)

        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)

        # Add observations across multiple frames
        self.aggregator.add_observation(27, det1, ocr1, dummy_img, dummy_img, 0.92)
        self.aggregator.add_observation(27, det2, ocr2, dummy_img, dummy_img, 0.92)

        records = self.aggregator.finalize_vehicle_records()

        # Verify EXACTLY 1 record for track #27
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.track_id, 27)
        self.assertEqual(record.plate, "GJ01AB1234")
        self.assertEqual(record.plate_status, "validated")
        self.assertEqual(record.ocr_confidence, 0.91)
        self.assertIsNone(record.location["latitude"])
        self.assertIsNone(record.location["longitude"])

    def test_c3_unreadable_and_rejected_plate_handling(self):
        """Test C3 & C4: unreadable or format rejected track produces plate: null, plate_status in attrs."""
        veh = TrackedObject(32, "TRUCK", 0.88, (100, 100, 400, 400), 250, 250, 250, 250)
        self.aggregator.update_track_metadata([veh], frame_number=50, timestamp=1.6)

        records = self.aggregator.finalize_vehicle_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsNone(record.plate)
        self.assertIn(record.plate_status, ["no_text", "format_rejected", "low_confidence", "unreadable"])
        d = record.to_dict()
        self.assertIsNone(d["plate"])
        self.assertIn("plate", d["attrs"])
        self.assertIsNone(d["attrs"]["plate"])
        self.assertIn("plate_status", d["attrs"])
        self.assertIn("plate_confidence", d["attrs"])
        self.assertIn("ocr_raw", d["attrs"])

    def test_c4_vehicle_record_schema_serialization(self):
        """Verify VehicleRecord.to_dict() contains C4 fields inside attrs and root."""
        record = VehicleRecord(
            track_id=27,
            vehicle_type="CAR",
            vehicle_confidence=0.91,
            plate="GJ01AB1234",
            registration_number="GJ01AB1234",
            ocr_raw="GJ 01 AB 1234",
            raw_ocr="GJ 01 AB 1234",
            plate_confidence=0.91,
            ocr_confidence=0.91,
            plate_status="validated",
            plate_detection_confidence=0.88,
            timestamp=12.45,
            frame_number=374
        )
        d = record.to_dict()
        self.assertTrue(d["event_id"].startswith("VEH-"))
        self.assertEqual(d["event_type"], "VEHICLE_IDENTIFICATION")
        self.assertEqual(d["vehicle"]["track_id"], 27)
        self.assertEqual(d["vehicle"]["type"], "CAR")
        self.assertEqual(d["plate"], "GJ01AB1234")
        self.assertEqual(d["plate_status"], "validated")
        self.assertEqual(d["plate_confidence"], 0.91)
        self.assertEqual(d["ocr_raw"], "GJ 01 AB 1234")

        # C4 — Explicitly in attrs
        self.assertEqual(d["attrs"]["plate"], "GJ01AB1234")
        self.assertEqual(d["attrs"]["plate_status"], "validated")
        self.assertEqual(d["attrs"]["plate_confidence"], 0.91)
        self.assertEqual(d["attrs"]["ocr_raw"], "GJ 01 AB 1234")


class TestPhase22APIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_status_endpoint_returns_ready(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])

if __name__ == "__main__":
    unittest.main()

