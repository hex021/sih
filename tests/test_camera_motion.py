import os
import sys
import tempfile
import unittest
import cv2
import numpy as np

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1.camera_motion import detect_camera_motion

class TestCameraMotionDetection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def _create_synthetic_video(self, filename: str, move_pixels_per_frame: float = 0.0, num_frames: int = 30) -> str:
        """Helper to create a synthetic video with textured background and optional motion."""
        file_path = os.path.join(self.temp_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        width, height = 320, 240
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        # Base image with rich texture (grid lines and circles for keypoint detection)
        base_img = np.zeros((height, width, 3), dtype=np.uint8)
        base_img[:, :] = (50, 50, 50)

        # Draw grid pattern
        for x in range(0, width, 20):
            cv2.line(base_img, (x, 0), (x, height), (200, 200, 200), 2)
        for y in range(0, height, 20):
            cv2.line(base_img, (0, y), (width, y), (200, 200, 200), 2)
        cv2.circle(base_img, (160, 120), 40, (0, 255, 0), -1)

        for i in range(num_frames):
            shift_x = int(i * move_pixels_per_frame)
            # Create shifted matrix for camera motion simulation
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            frame = cv2.warpAffine(base_img, M, (width, height))
            out.write(frame)

        out.release()
        return file_path

    def test_non_existent_file_returns_stationary(self):
        """Verify that non-existent video path gracefully defaults to 'stationary'."""
        mode = detect_camera_motion(os.path.join(self.temp_dir, "does_not_exist.mp4"))
        self.assertEqual(mode, "stationary")

    def test_stationary_camera_detection(self):
        """Verify that video with zero background movement is classified as 'stationary'."""
        video_path = self._create_synthetic_video("stationary_test.mp4", move_pixels_per_frame=0.0)
        mode = detect_camera_motion(video_path, max_frames=30)
        self.assertEqual(mode, "stationary")

    def test_moving_camera_detection(self):
        """Verify that video with horizontal background displacement is classified as 'moving'."""
        video_path = self._create_synthetic_video("moving_test.mp4", move_pixels_per_frame=4.0)
        mode = detect_camera_motion(video_path, max_frames=30, motion_threshold=1.2)
        self.assertEqual(mode, "moving")

    def test_custom_parameters_and_stride(self):
        """Verify custom max_frames, stride, and threshold parameters execute cleanly."""
        video_path = self._create_synthetic_video("custom_test.mp4", move_pixels_per_frame=0.0)
        mode = detect_camera_motion(video_path, max_frames=15, motion_threshold=2.0, sample_stride=1)
        self.assertEqual(mode, "stationary")

if __name__ == "__main__":
    unittest.main()
