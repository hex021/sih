import os
import sys
# Programmatically add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from phase1 import config
from phase1 import roi
from phase1 import density
from phase1.models import Detection, TrackedObject
from phase1.counter import Counter
from phase1.tracker import Tracker
from phase1.detector import Detector

class TestUnitModels(unittest.TestCase):
    def test_detection_dataclass(self):
        det = Detection(class_name="CAR", confidence=0.95, bbox=(10.0, 20.0, 100.0, 150.0))
        self.assertEqual(det.class_name, "CAR")
        self.assertEqual(det.confidence, 0.95)
        self.assertEqual(det.bbox, (10.0, 20.0, 100.0, 150.0))

    def test_tracked_object_dataclass(self):
        obj = TrackedObject(
            track_id=5,
            class_name="BUS",
            confidence=0.88,
            bbox=(50.0, 60.0, 200.0, 250.0),
            center_x=125.0,
            center_y=155.0,
            previous_center_x=120.0,
            previous_center_y=150.0
        )
        self.assertEqual(obj.track_id, 5)
        self.assertEqual(obj.class_name, "BUS")
        self.assertEqual(obj.center_x, 125.0)
        self.assertEqual(obj.previous_center_y, 150.0)

class TestUnitConfig(unittest.TestCase):
    def test_class_mapping(self):
        self.assertEqual(config.CLASS_MAP["car"], "CAR")
        self.assertEqual(config.CLASS_MAP["person"], "PERSON")
        self.assertIn("CAR", config.TARGET_CLASSES)
        self.assertIn("PERSON", config.TARGET_CLASSES)

class TestUnitROI(unittest.TestCase):
    def test_roi_conversion(self):
        # Frame size 1000x1000
        # config.ROI_RELATIVE is (0.0, 0.3, 1.0, 0.95)
        x_min, y_min, x_max, y_max = roi.get_pixel_roi(1000, 1000)
        self.assertEqual(x_min, int(config.ROI_RELATIVE[0] * 1000))
        self.assertEqual(y_min, int(config.ROI_RELATIVE[1] * 1000))
        self.assertEqual(x_max, int(config.ROI_RELATIVE[2] * 1000))
        self.assertEqual(y_max, int(config.ROI_RELATIVE[3] * 1000))

    def test_roi_inclusion(self):
        # Inside point (relative ROI covers 30% to 95% vertical space by default)
        self.assertTrue(roi.is_inside((500, 500), 1000, 1000))
        # Outside point (above ROI)
        self.assertFalse(roi.is_inside((500, 100), 1000, 1000))
        # Outside point (below ROI)
        self.assertFalse(roi.is_inside((500, 980), 1000, 1000))

class TestUnitCounting(unittest.TestCase):
    def test_line_crossing_and_duplicate_prevention(self):
        # Let's say frame height is 1000
        # config.COUNT_LINE_RELATIVE_Y is 0.65, so absolute line y is 650
        counter = Counter()
        
        # 1. Vehicle approach: previous y = 600, current y = 630. No crossing.
        obj = TrackedObject(
            track_id=1, class_name="CAR", confidence=0.9, bbox=(0,0,0,0),
            center_x=100, center_y=630, previous_center_x=100, previous_center_y=600
        )
        counter.update([obj], 1000, 1000)
        self.assertEqual(counter.counts["incoming"]["CAR"], 0)
        
        # 2. Vehicle crosses: previous y = 630, current y = 660. Crosses line 650 downwards.
        obj = TrackedObject(
            track_id=1, class_name="CAR", confidence=0.9, bbox=(0,0,0,0),
            center_x=100, center_y=660, previous_center_x=100, previous_center_y=630
        )
        counter.update([obj], 1000, 1000)
        self.assertEqual(counter.counts["incoming"]["CAR"], 1)
        
        # 3. Vehicle continues below line: previous y = 660, current y = 690.
        # Should not count again because track_id=1 is already counted as incoming.
        obj = TrackedObject(
            track_id=1, class_name="CAR", confidence=0.9, bbox=(0,0,0,0),
            center_x=100, center_y=690, previous_center_x=100, previous_center_y=660
        )
        counter.update([obj], 1000, 1000)
        self.assertEqual(counter.counts["incoming"]["CAR"], 1)
        
        # 4. Outgoing crossing check: ID 2 moves upwards from 700 to 600.
        # Crosses line 650 upwards.
        obj2 = TrackedObject(
            track_id=2, class_name="MOTORCYCLE", confidence=0.9, bbox=(0,0,0,0),
            center_x=200, center_y=600, previous_center_x=200, previous_center_y=700
        )
        counter.update([obj2], 1000, 1000)
        self.assertEqual(counter.counts["outgoing"]["MOTORCYCLE"], 1)
        
        # 5. Verify totals separation
        # total vehicles: CAR (incoming 1) + MOTORCYCLE (outgoing 1) = 2
        # total persons: 0
        total_veh, total_pers = counter.get_totals()
        self.assertEqual(total_veh, 2)
        self.assertEqual(total_pers, 0)
        
        # 6. Verify Person count is separate
        obj3 = TrackedObject(
            track_id=3, class_name="PERSON", confidence=0.9, bbox=(0,0,0,0),
            center_x=300, center_y=660, previous_center_x=300, previous_center_y=600
        )
        counter.update([obj3], 1000, 1000)
        total_veh, total_pers = counter.get_totals()
        self.assertEqual(total_veh, 2)
        self.assertEqual(total_pers, 1)

class TestUnitDensity(unittest.TestCase):
    def test_density_classification(self):
        # We will create mock TrackedObjects inside and outside the ROI
        # ROI relative is (0.0, 0.3, 1.0, 0.95), which is y=300..950 in 1000x1000 space
        # Thresholds: LOW_MAX=5, MEDIUM_MAX=15
        
        # Case A: 3 active vehicles in ROI (LOW)
        objs = [
            TrackedObject(1, "CAR", 0.9, (0,0,0,0), 500, 500, 500, 500), # In ROI
            TrackedObject(2, "BUS", 0.9, (0,0,0,0), 500, 600, 500, 600), # In ROI
            TrackedObject(3, "TRUCK", 0.9, (0,0,0,0), 500, 800, 500, 800), # In ROI
            TrackedObject(4, "PERSON", 0.9, (0,0,0,0), 500, 500, 500, 500), # Person (excluded)
            TrackedObject(5, "CAR", 0.9, (0,0,0,0), 500, 100, 500, 100), # Out of ROI (above)
        ]
        density_cls, active_count = density.classify_density(objs, 1000, 1000)
        self.assertEqual(active_count, 3)
        self.assertEqual(density_cls, "LOW")
        
        # Case B: 6 active vehicles in ROI (MEDIUM)
        objs_med = [TrackedObject(i, "CAR", 0.9, (0,0,0,0), 500, 500, 500, 500) for i in range(6)]
        density_cls, active_count = density.classify_density(objs_med, 1000, 1000)
        self.assertEqual(active_count, 6)
        self.assertEqual(density_cls, "MEDIUM")
        
        # Case C: 16 active vehicles in ROI (HIGH)
        objs_high = [TrackedObject(i, "CAR", 0.9, (0,0,0,0), 500, 500, 500, 500) for i in range(16)]
        density_cls, active_count = density.classify_density(objs_high, 1000, 1000)
        self.assertEqual(active_count, 16)
        self.assertEqual(density_cls, "HIGH")

class TestIntegrationYOLO(unittest.TestCase):
    def setUp(self):
        # Check if weight file exists
        self.weights_exist = os.path.exists(config.MODEL_PATH)
        
    def test_yolo_integration_if_weights_available(self):
        if not self.weights_exist:
            self.skipTest(f"YOLO model weights not found at {config.MODEL_PATH}. Skipping integration tests.")
            
        # 1. Test Tracker initialization
        tracker = Tracker()
        self.assertIsNotNone(tracker.model)
        
        # 2. Test Detector initialization
        detector = Detector()
        self.assertIsNotNone(detector.model)
        
        # 3. Run Tracker update with a dummy blank image
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracked_objects = tracker.update(dummy_frame)
        self.assertIsInstance(tracked_objects, list)
        
        # 4. Run Detector detect with dummy blank image
        detections = detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)

from phase1.main import filter_pedestrians

class TestUnitPedestrianSuppression(unittest.TestCase):
    def test_rider_suppression(self):
        # Create a motorcycle and a person that is riding it (overlapping/contained within box)
        motorcycle = TrackedObject(
            track_id=10, class_name="MOTORCYCLE", confidence=0.9, bbox=(100, 200, 200, 300),
            center_x=150, center_y=250, previous_center_x=150, previous_center_y=250
        )
        # Person center (150, 220) is inside motorcycle bbox (100, 200, 200, 300)
        rider = TrackedObject(
            track_id=11, class_name="PERSON", confidence=0.85, bbox=(120, 180, 180, 240),
            center_x=150, center_y=210, previous_center_x=150, previous_center_y=210
        )
        
        # Normal independent pedestrian (far away)
        pedestrian = TrackedObject(
            track_id=12, class_name="PERSON", confidence=0.9, bbox=(500, 500, 540, 580),
            center_x=520, center_y=540, previous_center_x=520, previous_center_y=540
        )
        
        objs = [motorcycle, rider, pedestrian]
        filtered = filter_pedestrians(objs)
        
        # Verify rider is suppressed (renamed to RIDER) and pedestrian is kept as PERSON
        self.assertEqual(rider.class_name, "RIDER")
        self.assertEqual(pedestrian.class_name, "PERSON")
        self.assertEqual(motorcycle.class_name, "MOTORCYCLE")

class TestUnitCountingPortraitOverride(unittest.TestCase):
    def test_portrait_line_crossing(self):
        # In a 1000x1000 frame, with line override rel_y = 0.75, absolute line y is 750.
        counter = Counter()
        
        # Moving downwards from 720 to 760 -> Crosses Y=750 downward (INCOMING)
        obj1 = TrackedObject(
            track_id=1, class_name="CAR", confidence=0.9, bbox=(0,0,0,0),
            center_x=100, center_y=760, previous_center_x=100, previous_center_y=720
        )
        counter.update([obj1], 1000, 1000, line_y_rel=0.75)
        self.assertEqual(counter.counts["incoming"]["CAR"], 1)
        
        # Moving upwards from 780 to 730 -> Crosses Y=750 upward (OUTGOING)
        obj2 = TrackedObject(
            track_id=2, class_name="CAR", confidence=0.9, bbox=(0,0,0,0),
            center_x=200, center_y=730, previous_center_x=200, previous_center_y=780
        )
        counter.update([obj2], 1000, 1000, line_y_rel=0.75)
        self.assertEqual(counter.counts["outgoing"]["CAR"], 1)

if __name__ == "__main__":
    unittest.main()
