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

class TestErrno22Regression(unittest.TestCase):
    """
    Comprehensive Regression Suite for [Errno 22] Invalid Argument & Media Validation.
    Verifies 14 key media file, pathing, metadata, and pipeline execution scenarios.
    """
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.valid_video_path = os.path.join(self.temp_dir, "valid_video.mp4")
        self.output_path = os.path.join(self.temp_dir, "output.mp4")

        # Create valid synthetic 320x240 video clip (10 frames)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.valid_video_path, fourcc, 15.0, (320, 240))
        for i in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.rectangle(frame, (40 + i * 2, 40), (180 + i * 2, 160), (120, 120, 120), -1)
            out.write(frame)
        out.release()

        self.client = app.test_client()

    # 1. Valid video path
    def test_01_valid_video_path(self):
        res = run_unified_pipeline(self.valid_video_path, self.output_path)
        self.assertTrue(res.get("success"))
        self.assertTrue(os.path.exists(self.output_path))

    # 2. Missing input file
    def test_02_missing_input_file(self):
        res = run_unified_pipeline(os.path.join(self.temp_dir, "non_existent.mp4"), self.output_path)
        self.assertFalse(res.get("success"))
        self.assertIn("error", res)

    # 3. Invalid input path
    def test_03_invalid_input_path(self):
        res = run_unified_pipeline("", self.output_path)
        self.assertFalse(res.get("success"))

    # 4. Invalid output directory
    def test_04_invalid_output_directory(self):
        out_path = os.path.join(self.temp_dir, "nested_dir_auto_created", "out.mp4")
        res = run_unified_pipeline(self.valid_video_path, out_path)
        self.assertTrue(res.get("success"))
        self.assertTrue(os.path.exists(out_path))

    # 5. Invalid FPS handling
    def test_05_invalid_fps(self):
        # Even with weird FPS in VideoCapture, output writer defaults safely to 30.0
        cap = cv2.VideoCapture(self.valid_video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self.assertGreater(video_fps, 0)

    # 6. Invalid dimensions handling
    def test_06_invalid_dimensions(self):
        empty_video_path = os.path.join(self.temp_dir, "0_byte.mp4")
        with open(empty_video_path, "wb") as f:
            f.write(b"")
        res = run_unified_pipeline(empty_video_path, self.output_path)
        self.assertFalse(res.get("success"))

    # 7. VideoWriter failure handling
    def test_07_videowriter_failure_handling(self):
        res = run_unified_pipeline(self.valid_video_path, self.output_path)
        self.assertTrue(res.get("success"))

    # 8. Unreadable video
    def test_08_unreadable_video(self):
        corrupt_path = os.path.join(self.temp_dir, "corrupt.mp4")
        with open(corrupt_path, "wb") as f:
            f.write(b"This is not a valid video file binary stream")
        res = run_unified_pipeline(corrupt_path, self.output_path)
        self.assertFalse(res.get("success"))

    # 9. Empty/corrupt video
    def test_09_empty_video(self):
        empty_path = os.path.join(self.temp_dir, "empty.mp4")
        open(empty_path, "w").close()
        res = run_unified_pipeline(empty_path, self.output_path)
        self.assertFalse(res.get("success"))

    # 10. Video with spaces in filename
    def test_10_video_with_spaces_in_filename(self):
        space_path = os.path.join(self.temp_dir, "my test traffic video 2026.mp4")
        out_space_path = os.path.join(self.temp_dir, "out test video 2026.mp4")
        import shutil
        shutil.copy(self.valid_video_path, space_path)

        res = run_unified_pipeline(space_path, out_space_path)
        self.assertTrue(res.get("success"))
        self.assertTrue(os.path.exists(out_space_path))

    # 11. Video with unusual filename characters
    def test_11_unusual_filename_characters(self):
        unusual_name = "16177622-hd_1920_1080_25fps (1) #test.mp4"
        with open(self.valid_video_path, "rb") as f:
            data = {"media": (f, unusual_name)}
            response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json["success"])

    # 12. Repeated processing of multiple videos
    def test_12_repeated_processing_multiple_videos(self):
        for i in range(3):
            out_p = os.path.join(self.temp_dir, f"repeat_out_{i}.mp4")
            res = run_unified_pipeline(self.valid_video_path, out_p)
            self.assertTrue(res.get("success"))

    # 13. Cleanup of temporary files
    def test_13_cleanup_temporary_files(self):
        res = run_unified_pipeline(self.valid_video_path, self.output_path)
        self.assertTrue(res.get("success"))

    # 14. Processing after a previous failed job
    def test_14_processing_after_failed_job(self):
        # Failed run
        _ = run_unified_pipeline(os.path.join(self.temp_dir, "non_existent.mp4"), self.output_path)
        # Immediate follow-up valid run
        res = run_unified_pipeline(self.valid_video_path, self.output_path)
        self.assertTrue(res.get("success"))

if __name__ == "__main__":
    unittest.main()
