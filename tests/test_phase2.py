import os
import sys
import unittest
import numpy as np
import tempfile
import cv2

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase2.road_ai import pothole_config
from phase2.road_ai.pothole_models import PotholeEvent, PotholeDetection
from phase2.road_ai.pothole_detector import PotholeDetector
from backend.server import app

class TestPhase2Models(unittest.TestCase):
    def test_pothole_event_creation_and_serialization(self):
        """Test PotholeEvent instantiation and nullable GPS/IMU structure."""
        event = PotholeEvent(
            event_id="pothole_abc123",
            timestamp=5.25,
            confidence=0.89,
            location=None, # Nullable GPS
            source={"camera": "uploaded_video", "vehicle_id": None},
            media={"frame": 150, "snapshot": "/processed/events/pothole/event_abc123.jpg", "video": None},
            sensor_data={"gps": None, "imu": None} # Nullable IMU
        )
        
        # Verify nullable parameters
        self.assertIsNone(event.location)
        self.assertIsNone(event.sensor_data["gps"])
        self.assertIsNone(event.sensor_data["imu"])
        
        # Verify dict conversion
        d = event.to_dict()
        self.assertEqual(d["event_id"], "pothole_abc123")
        self.assertEqual(d["timestamp"], 5.25)
        self.assertEqual(d["confidence"], 0.89)
        self.assertIsNone(d["location"])
        self.assertIsNone(d["sensor_data"]["gps"])

class TestPotholeDeduplication(unittest.TestCase):
    def setUp(self):
        # We instantiate PotholeDetector without loading the YOLO model for unit-testing IoU/deduplication
        # We will mock the YOLO model and override active_tracks for spatial/temporal tests
        self.detector = PotholeDetector.__new__(PotholeDetector)
        self.detector.active_tracks = []
        
    def test_iou_calculation(self):
        """Test bounding box Intersection over Union (IoU) helper."""
        boxA = (100, 100, 200, 200)
        boxB = (150, 150, 250, 250) # Overlap size: 50x50 = 2500
        
        iou = self.detector._compute_iou(boxA, boxB)
        # Intersection = 50 * 50 = 2500
        # Union = (100*100) + (100*100) - 2500 = 17500
        # IoU = 2500 / 17500 = 0.1428
        self.assertAlmostEqual(iou, 0.142857, places=5)
        
        # No overlap
        boxC = (300, 300, 400, 400)
        self.assertEqual(self.detector._compute_iou(boxA, boxC), 0.0)

    def test_track_association_and_deduplication(self):
        """Test that same physical pothole across consecutive frames does not spawn new events."""
        # Frame 100: New Pothole Event Created
        event_id = "pothole_test1"
        event = PotholeEvent(event_id=event_id, timestamp=3.3, confidence=0.85)
        self.detector.active_tracks.append({
            "event_id": event_id,
            "bbox": (100, 100, 200, 200),
            "last_frame": 100,
            "max_conf": 0.85,
            "event": event
        })
        
        # Frame 101: Same Pothole with heavy overlap (IoU = 1.0)
        matched = self.detector._find_matching_track((100, 100, 200, 200))
        self.assertIsNotNone(matched)
        self.assertEqual(matched["event_id"], event_id)
        
        # Frame 101: Different Pothole location (IoU = 0) -> No match
        no_match = self.detector._find_matching_track((400, 400, 500, 500))
        self.assertIsNone(no_match)

    def test_stale_track_pruning(self):
        """Test that tracks not seen within cooldown frames are correctly pruned from active tracks list."""
        self.detector.active_tracks = [
            {"event_id": "p1", "last_frame": 100},
            {"event_id": "p2", "last_frame": 80} # 25 frames inactive if current_frame = 105
        ]
        
        # Run pruning at frame 105 (with cooldown_frames = 15)
        self.detector._prune_stale_tracks(105)
        
        # p1 remains (105-100 = 5 <= 15), p2 is pruned (105-80 = 25 > 15)
        self.assertEqual(len(self.detector.active_tracks), 1)
        self.assertEqual(self.detector.active_tracks[0]["event_id"], "p1")

class TestPotholeDetectorLoading(unittest.TestCase):
    def test_missing_model_exception(self):
        """Verify that initializing PotholeDetector with an invalid path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            PotholeDetector(model_path="nonexistent_folder/best_weights.pt")

class TestAPIAnalysisRoutes(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_status_endpoint(self):
        """Verify that the status API route works successfully."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("device", data)

class TestTrafficAnalyticsCameraModes(unittest.TestCase):
    def test_counter_directional_counting_and_hover_prevention(self):
        """Test that vehicle crossing counting distinguishes incoming/outgoing and prevents duplicate/hover count."""
        from phase1.counter import Counter
        from phase1.models import TrackedObject
        
        counter = Counter()
        # Incoming Crossing (prev_y < line_y <= curr_y)
        # Rel Y = 0.5, Height = 300 -> Line Y = 150
        obj = TrackedObject(
            track_id=1,
            class_name="CAR",
            confidence=0.9,
            bbox=(50, 100, 150, 200),
            center_x=100.0,
            center_y=200.0,
            previous_center_x=100.0,
            previous_center_y=100.0
        )
        
        counter.update([obj], frame_width=300, frame_height=300, line_y_rel=0.5)
        self.assertEqual(counter.counts["incoming"]["CAR"], 1)
        self.assertEqual(counter.counts["outgoing"]["CAR"], 0)
        
        # Hovering around line (Y=200 -> Y=180) -> should not count again
        obj.previous_center_y = 200.0
        obj.center_y = 180.0
        counter.update([obj], frame_width=300, frame_height=300, line_y_rel=0.5)
        self.assertEqual(counter.counts["incoming"]["CAR"], 1) # Still 1

    def test_rider_filtering(self):
        """Test rider classification spatial heuristic."""
        from phase1.main import filter_pedestrians
        from phase1.models import TrackedObject
        
        # Overlapping rider
        motorcycle = TrackedObject(
            track_id=1,
            class_name="MOTORCYCLE",
            confidence=0.85,
            bbox=(100, 100, 200, 200),
            center_x=150.0,
            center_y=150.0,
            previous_center_x=150.0,
            previous_center_y=150.0
        )
        person = TrackedObject(
            track_id=2,
            class_name="PERSON",
            confidence=0.9,
            bbox=(120, 110, 180, 190),
            center_x=150.0,
            center_y=150.0,
            previous_center_x=150.0,
            previous_center_y=150.0
        )
        
        filtered = filter_pedestrians([motorcycle, person])
        self.assertEqual(filtered[1].class_name, "RIDER")
        
        # Separate pedestrian
        separate_person = TrackedObject(
            track_id=3,
            class_name="PERSON",
            confidence=0.9,
            bbox=(400, 400, 500, 500),
            center_x=450.0,
            center_y=450.0,
            previous_center_x=450.0,
            previous_center_y=450.0
        )
        filtered_sep = filter_pedestrians([motorcycle, separate_person])
        self.assertEqual(filtered_sep[1].class_name, "PERSON")

    def test_density_evaluation_by_active_roi_vehicles(self):
        """Verify that traffic density is determined solely by active ROI vehicles, not historic seen list."""
        from phase1.density import classify_density
        from phase1.models import TrackedObject
        
        # 3 active vehicles in ROI
        v1 = TrackedObject(
            track_id=1,
            class_name="CAR",
            confidence=0.9,
            bbox=(50, 150, 150, 250),
            center_x=100.0,
            center_y=200.0,
            previous_center_x=100.0,
            previous_center_y=200.0
        )
        v2 = TrackedObject(
            track_id=2,
            class_name="MOTORCYCLE",
            confidence=0.8,
            bbox=(80, 150, 120, 250),
            center_x=100.0,
            center_y=200.0,
            previous_center_x=100.0,
            previous_center_y=200.0
        )
        v3 = TrackedObject(
            track_id=3,
            class_name="BUS",
            confidence=0.9,
            bbox=(200, 150, 280, 250),
            center_x=240.0,
            center_y=200.0,
            previous_center_x=240.0,
            previous_center_y=200.0
        )
        
        density_class, count = classify_density([v1, v2, v3], frame_width=300, frame_height=300, roi_relative=(0.0, 0.3, 1.0, 0.95))
        self.assertEqual(count, 3)
        self.assertEqual(density_class, "LOW")
        
        # 6 active vehicles in ROI -> MEDIUM
        vehicles = [TrackedObject(
            track_id=i,
            class_name="CAR",
            confidence=0.9,
            bbox=(50, 150, 150, 250),
            center_x=100.0,
            center_y=200.0,
            previous_center_x=100.0,
            previous_center_y=200.0
        ) for i in range(6)]
        density_class_med, count_med = classify_density(vehicles, frame_width=300, frame_height=300, roi_relative=(0.0, 0.3, 1.0, 0.95))
        self.assertEqual(count_med, 6)
        self.assertEqual(density_class_med, "MEDIUM")

    def test_camera_mode_roi_selection(self):
        """Test pixel calculation of moving camera custom ROI (excl. sky & dashboard)."""
        from phase1.roi import get_pixel_roi
        
        moving_roi = (0.1, 0.4, 0.9, 0.85)
        x1, y1, x2, y2 = get_pixel_roi(1000, 1000, moving_roi)
        self.assertEqual(x1, 100)
        self.assertEqual(y1, 400)
        self.assertEqual(x2, 900)
        self.assertEqual(y2, 850)

class TestVisualizerRegressionAndPothole(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_visualizer_draw_signature_regression(self):
        """Verify Visualizer.draw() works cleanly with counter=counter parameter without unexpected keyword argument errors."""
        from phase1.visualizer import Visualizer
        from phase1.counter import Counter
        from phase1.models import TrackedObject
        import numpy as np

        vis = Visualizer(debug=False)
        counter = Counter()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracked = [
            TrackedObject(track_id=1, class_name="car", confidence=0.9, bbox=(10, 10, 100, 100), center_x=55, center_y=55, previous_center_x=55, previous_center_y=55)
        ]

        annotated = vis.draw(
            frame=frame,
            tracked_objects=tracked,
            counter=counter,
            density_class="LOW",
            active_roi_count=1,
            line_y_rel=0.5,
            roi_relative=(0.0, 0.0, 1.0, 1.0),
            camera_mode="stationary",
            unique_counts={"car": 1}
        )
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (480, 640, 3))

    def test_pothole_detection_on_test_image(self):
        """Verify PotholeDetector runs inference and detects potholes on the sample pothole test image."""
        test_img_path = "uploads/10116dc1_image_input.jpg"
        if os.path.exists(test_img_path):
            detector = PotholeDetector()
            frame = cv2.imread(test_img_path)
            dets, events = detector.detect(frame, frame_number=1, timestamp=0.0)
            self.assertGreater(len(dets), 0, "PotholeDetector should detect potholes on test image")
            self.assertGreater(len(events), 0, "PotholeDetector should generate pothole events on test image")
            for d in dets:
                self.assertEqual(d.class_name, "POTHOLE")
                self.assertGreater(d.confidence, 0.1)

if __name__ == "__main__":
    unittest.main()

