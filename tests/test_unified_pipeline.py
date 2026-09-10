import os
import sys
import tempfile
import unittest
import cv2
import numpy as np

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase2.unified_pipeline import run_unified_pipeline
from backend.server import app

class TestUnifiedPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_video_path = os.path.join(self.temp_dir, "test_input.mp4")
        self.output_video_path = os.path.join(self.temp_dir, "test_output.mp4")

        # Create a synthetic 320x240 video clip (15 frames) with a drawn vehicle & plate rectangle
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.input_video_path, fourcc, 15.0, (320, 240))
        for i in range(15):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Vehicle box
            cv2.rectangle(frame, (50 + i * 2, 50), (200 + i * 2, 180), (100, 100, 100), -1)
            # Plate box
            cv2.rectangle(frame, (100 + i * 2, 130), (160 + i * 2, 160), (255, 255, 255), -1)
            out.write(frame)
        out.release()

        self.client = app.test_client()

    def test_missing_input_file_returns_error(self):
        """Verify that non-existent input video path returns graceful error dict."""
        res = run_unified_pipeline(
            input_path=os.path.join(self.temp_dir, "non_existent.mp4"),
            output_path=self.output_video_path
        )
        self.assertFalse(res["success"])
        self.assertIn("error", res)

    def test_run_unified_pipeline_auto_camera_mode(self):
        """Test unified pipeline execution with auto camera mode detection."""
        res = run_unified_pipeline(
            input_path=self.input_video_path,
            output_path=self.output_video_path,
            camera_mode="auto"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "unified")
        self.assertIn(res["camera_mode"], ["stationary", "moving"])
        self.assertIn("traffic", res)
        self.assertIn("vehicle_records", res)
        self.assertIn("events", res)
        self.assertIn("summary", res)
        self.assertTrue(os.path.exists(self.output_video_path))
        self.assertGreater(os.path.getsize(self.output_video_path), 0)

    def test_run_unified_pipeline_explicit_moving_mode(self):
        """Test unified pipeline execution with explicit camera_mode='moving'."""
        out_path = os.path.join(self.temp_dir, "test_moving_out.mp4")
        res = run_unified_pipeline(
            input_path=self.input_video_path,
            output_path=out_path,
            camera_mode="moving"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["camera_mode"], "moving")
        self.assertFalse(res["traffic"]["line_crossings"]["enabled"])
        self.assertTrue(os.path.exists(out_path))

    def test_run_unified_pipeline_explicit_stationary_mode(self):
        """Test unified pipeline execution with explicit camera_mode='stationary'."""
        out_path = os.path.join(self.temp_dir, "test_stat_out.mp4")
        res = run_unified_pipeline(
            input_path=self.input_video_path,
            output_path=out_path,
            camera_mode="stationary"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["camera_mode"], "stationary")
        self.assertTrue(res["traffic"]["line_crossings"]["enabled"])
        self.assertTrue(os.path.exists(out_path))

    def test_api_analyze_endpoint_default_unified(self):
        """Test POST /api/analyze Flask API endpoint defaulting to unified pipeline."""
        with open(self.input_video_path, "rb") as f:
            data = {"video": (f, "test_input.mp4")}
            response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json["success"])
        self.assertEqual(res_json["mode"], "unified")
        self.assertIn("summary", res_json)
        self.assertIn("video_url", res_json)
        self.assertIn("download_url", res_json)

if __name__ == "__main__":
    unittest.main()
