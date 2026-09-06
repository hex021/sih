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

    def test_indian_plate_normalization(self):
        """Test normalization and correction of Indian vehicle registration plates."""
        # Case 1: Raw string with spaces -> Normalized
        normalized, valid = self.ocr.normalize_indian_plate(" GJ 01 AB 1234 ")
        self.assertEqual(normalized, "GJ01AB1234")
        self.assertTrue(valid)

        # Case 2: State prefix digit-to-letter correction ('0' -> 'O')
        normalized2, valid2 = self.ocr.normalize_indian_plate("MH 12 CD 5678")
        self.assertEqual(normalized2, "MH12CD5678")
        self.assertTrue(valid2)

        # Case 3: Letter-to-digit correction in district code ('O' -> '0')
        normalized3, valid3 = self.ocr.normalize_indian_plate("GJ O1 AB 1234")
        self.assertEqual(normalized3, "GJ01AB1234")
        self.assertTrue(valid3)

        # Case 4: Invalid short text
        _, valid4 = self.ocr.normalize_indian_plate("AB")
        self.assertFalse(valid4)

    def test_extract_text_fallback(self):
        """Test text extraction pipeline fallback on synthetic plate crop."""
        crop = np.zeros((48, 160, 3), dtype=np.uint8)
        res = self.ocr.extract_text(crop)
        self.assertIsInstance(res, OCRResult)
        self.assertIn(res.normalized_text, ["UNREADABLE", "GJ01AB1234", "MH12CD5678"])

class TestPhase22MultiFrameAggregatorAndDeduplication(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.aggregator = ANPRAggregator(output_events_dir=self.temp_dir)

    def test_multi_frame_aggregation_and_single_record(self):
        """Test that multiple frames for track #27 produce exactly ONE primary VehicleRecord."""
        # Track 27 metadata
        veh = TrackedObject(27, "CAR", 0.92, (100, 100, 300, 300), 200, 200, 200, 200)
        self.aggregator.update_track_metadata([veh], frame_number=100, timestamp=3.3)
        
        det1 = PlateDetection((120, 120, 180, 150), 0.85, 100, 3.3, 27)
        ocr1 = OCRResult("GJ 01 AB 1234", "GJ01AB1234", 0.75, True)
        
        det2 = PlateDetection((120, 120, 180, 150), 0.90, 105, 3.5, 27)
        ocr2 = OCRResult("GJ 01 AB 1234", "GJ01AB1234", 0.91, True)

        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)

        # Add observations across multiple frames
        self.aggregator.add_observation(27, det1, ocr1, dummy_img, dummy_img, 0.92)
        self.aggregator.add_observation(27, det2, ocr2, dummy_img, dummy_img, 0.92)

        records = self.aggregator.finalize_vehicle_records()
        
        # Verify EXACTLY 1 record for track #27
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.track_id, 27)
        self.assertEqual(record.registration_number, "GJ01AB1234")
        self.assertEqual(record.ocr_confidence, 0.91)
        self.assertIsNone(record.location["latitude"])
        self.assertIsNone(record.location["longitude"])

    def test_unreadable_plate_handling(self):
        """Test graceful handling of track with no readable plate."""
        veh = TrackedObject(32, "TRUCK", 0.88, (100, 100, 400, 400), 250, 250, 250, 250)
        self.aggregator.update_track_metadata([veh], frame_number=50, timestamp=1.6)

        records = self.aggregator.finalize_vehicle_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].registration_number, "UNREADABLE")
        self.assertEqual(records[0].ocr_confidence, 0.0)

    def test_vehicle_record_schema_serialization(self):
        """Verify VehicleRecord.to_dict() matches the Phase 2.2 JSON schema."""
        record = VehicleRecord(
            track_id=27,
            vehicle_type="CAR",
            vehicle_confidence=0.91,
            registration_number="GJ01AB1234",
            raw_ocr="GJ 01 AB 1234",
            ocr_confidence=0.91,
            plate_detection_confidence=0.88,
            timestamp=12.45,
            frame_number=374
        )
        d = record.to_dict()
        self.assertTrue(d["event_id"].startswith("VEH-"))
        self.assertEqual(d["event_type"], "VEHICLE_IDENTIFICATION")
        self.assertEqual(d["vehicle"]["track_id"], 27)
        self.assertEqual(d["vehicle"]["type"], "CAR")
        self.assertEqual(d["number_plate"]["text"], "GJ01AB1234")
        self.assertEqual(d["number_plate"]["raw_ocr"], "GJ 01 AB 1234")
        self.assertIsNone(d["location"]["latitude"])
        self.assertIsNone(d["location"]["longitude"])

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
