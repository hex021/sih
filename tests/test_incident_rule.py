import os
import sys
import shutil
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1.models import TrackedObject
from phase2.road_ai.rules import RashDrivingRule, rule_registry
from phase2.road_ai.incident_models import IncidentReport
from phase2.road_ai.incident_generator import get_sample_track27_incident_report
from backend import db

def make_veh(track_id: int, class_name: str, confidence: float, bbox: tuple) -> TrackedObject:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        center_x=cx,
        center_y=cy,
        previous_center_x=cx,
        previous_center_y=cy
    )

class TestRashDrivingRule(unittest.TestCase):
    def setUp(self):
        self.test_crops_dir = "data/crops/test_incidents"
        os.makedirs(self.test_crops_dir, exist_ok=True)
        self.rule = RashDrivingRule(
            lateral_threshold=0.15,
            approach_threshold=0.12,
            min_box_height=60.0,
            cooldown_seconds=5.0,
            crops_dir=self.test_crops_dir
        )
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self):
        if os.path.exists(self.test_crops_dir):
            shutil.rmtree(self.test_crops_dir, ignore_errors=True)

    def test_minimum_box_height_filter(self):
        """Vehicles with box height < 60px should be ignored."""
        small_veh = make_veh(
            track_id=1,
            class_name="car",
            confidence=0.90,
            bbox=(100.0, 100.0, 140.0, 140.0) # height = 40px < 60px
        )
        events = self.rule.evaluate([small_veh], self.frame, frame_idx=1, timestamp=0.1)
        self.assertEqual(len(events), 0)
        self.assertEqual(self.rule.total_evaluated, 0)

    def test_lateral_swerve_detection_after_3_frames(self):
        """Vehicle swerving horizontally should fire after 3 consecutive frames."""
        # Box height = 100px. Threshold = 0.15 -> dx > 15px
        track_id = 10
        # Frame 0: Baseline
        v0 = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(100.0, 100.0, 200.0, 200.0))
        self.rule.evaluate([v0], self.frame, frame_idx=1, timestamp=0.1)

        # Frame 1: Shift right by 20px (lat_vel = 20/100 = 0.20 > 0.15) -> hit 1
        v1 = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(120.0, 100.0, 220.0, 200.0))
        ev1 = self.rule.evaluate([v1], self.frame, frame_idx=2, timestamp=0.2)
        self.assertEqual(len(ev1), 0)

        # Frame 2: Shift right by 20px -> hit 2
        v2 = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(140.0, 100.0, 240.0, 200.0))
        ev2 = self.rule.evaluate([v2], self.frame, frame_idx=3, timestamp=0.3)
        self.assertEqual(len(ev2), 0)

        # Frame 3: Shift right by 20px -> hit 3 -> FIRES INCIDENT!
        v3 = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(160.0, 100.0, 260.0, 200.0))
        ev3 = self.rule.evaluate([v3], self.frame, frame_idx=4, timestamp=0.4)
        self.assertEqual(len(ev3), 1)

        event = ev3[0]
        self.assertEqual(event["event_type"], "INCIDENT")
        self.assertEqual(event["attrs"]["track_id"], track_id)
        self.assertEqual(event["attrs"]["reason"], "lateral_swerve")
        self.assertGreaterEqual(event["attrs"]["measured_value"], 0.15)
        self.assertTrue(os.path.exists(event["attrs"]["crop_path"]))

    def test_rapid_approach_detection_after_3_frames(self):
        """Vehicle rapidly growing in height should fire after 3 consecutive frames."""
        track_id = 20
        # Frame 0: height = 100
        v0 = make_veh(track_id=track_id, class_name="truck", confidence=0.92, bbox=(100.0, 100.0, 200.0, 200.0))
        self.rule.evaluate([v0], self.frame, frame_idx=1, timestamp=0.1)

        # Frame 1: height grows to 115 (app_rate = 15/115 ~ 0.13 > 0.12) -> hit 1
        v1 = make_veh(track_id=track_id, class_name="truck", confidence=0.92, bbox=(100.0, 100.0, 200.0, 215.0))
        self.rule.evaluate([v1], self.frame, frame_idx=2, timestamp=0.2)

        # Frame 2: height grows to 135 -> hit 2
        v2 = make_veh(track_id=track_id, class_name="truck", confidence=0.92, bbox=(100.0, 100.0, 200.0, 235.0))
        self.rule.evaluate([v2], self.frame, frame_idx=3, timestamp=0.3)

        # Frame 3: height grows to 160 -> hit 3 -> FIRES!
        v3 = make_veh(track_id=track_id, class_name="truck", confidence=0.92, bbox=(100.0, 100.0, 200.0, 260.0))
        ev3 = self.rule.evaluate([v3], self.frame, frame_idx=4, timestamp=0.4)
        self.assertEqual(len(ev3), 1)
        self.assertEqual(ev3[0]["attrs"]["reason"], "rapid_approach")

    def test_cooldown_prevents_duplicate_firing(self):
        """Rule must respect 5 second cooldown per track ID."""
        track_id = 30
        # Trigger an incident
        v = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(100.0, 100.0, 200.0, 200.0))
        self.rule.evaluate([v], self.frame, frame_idx=1, timestamp=1.0)
        v = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(130.0, 100.0, 230.0, 200.0))
        self.rule.evaluate([v], self.frame, frame_idx=2, timestamp=1.1)
        v = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(160.0, 100.0, 260.0, 200.0))
        self.rule.evaluate([v], self.frame, frame_idx=3, timestamp=1.2)
        v = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(190.0, 100.0, 290.0, 200.0))
        ev = self.rule.evaluate([v], self.frame, frame_idx=4, timestamp=1.3)
        self.assertEqual(len(ev), 1)

        # Continue swerving during cooldown (timestamp=2.0s < 1.3s + 5s)
        for i in range(5, 9):
            v = make_veh(track_id=track_id, class_name="car", confidence=0.95, bbox=(200.0 + i*25, 100.0, 300.0 + i*25, 200.0))
            ev_sub = self.rule.evaluate([v], self.frame, frame_idx=i, timestamp=1.3 + (i-4)*0.1)
            self.assertEqual(len(ev_sub), 0)

    def test_dedup_exemption_for_incidents(self):
        """Verify INCIDENT events are exempt from deduplication in db.py."""
        db.init_db()
        ev1 = {
            "event_id": "INC-TEST1",
            "event_type": "INCIDENT",
            "timestamp": 10.0,
            "confidence": 0.95,
            "location": {"latitude": 23.0225, "longitude": 72.5714},
            "attrs": {"track_id": 1, "reason": "lateral_swerve"}
        }
        ev2 = {
            "event_id": "INC-TEST2",
            "event_type": "INCIDENT",
            "timestamp": 12.0,
            "confidence": 0.91,
            "location": {"latitude": 23.0225, "longitude": 72.5714},
            "attrs": {"track_id": 2, "reason": "rapid_approach"}
        }
        res1 = db.insert_event(ev1, "BUS-01", "SESS-01")
        res2 = db.insert_event(ev2, "BUS-01", "SESS-01")
        self.assertEqual(res1, "inserted")
        self.assertEqual(res2, "inserted") # NOT merged/deduped

    def test_sample_report_badge_and_flag(self):
        """Verify sample report contains is_sample=True and HTML summary renders badge."""
        sample = get_sample_track27_incident_report()
        d = sample.to_dict()
        self.assertTrue(d.get("is_sample"))
        html = sample.generate_html_summary()
        self.assertIn("SAMPLE REPORT", html)

if __name__ == "__main__":
    unittest.main()
