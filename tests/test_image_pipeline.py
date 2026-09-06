import os
import sys
import tempfile
import unittest
import numpy as np
import cv2

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase2.image_ai import ImageProcessor, extract_image_exif_metadata
from backend.server import app

class TestPhotoAnalysisPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_img_path = os.path.join(self.temp_dir, "test_traffic.jpg")
        
        # Create a synthetic 480x640 traffic scene image with a vehicle shape and plate rectangle
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw vehicle body (gray box)
        cv2.rectangle(img, (100, 100), (500, 400), (120, 120, 120), -1)
        # Draw license plate rectangle (white box with black border)
        cv2.rectangle(img, (200, 320), (400, 370), (255, 255, 255), -1)
        cv2.rectangle(img, (200, 320), (400, 370), (0, 0, 0), 2)
        cv2.imwrite(self.test_img_path, img)

        self.processor = ImageProcessor()
        self.client = app.test_client()

    def test_exif_metadata_extraction_returns_none_when_missing(self):
        """Verify that missing EXIF timestamp/GPS returns None and NEVER fabricates fake coordinates."""
        meta = extract_image_exif_metadata(self.test_img_path)
        self.assertIsNone(meta["capture_timestamp"])
        self.assertIsNone(meta["gps_latitude"])
        self.assertIsNone(meta["gps_longitude"])
        self.assertFalse(meta["has_exif_gps"])
        self.assertFalse(meta["has_exif_timestamp"])

    def test_process_still_image_pipeline(self):
        """Test process_image on synthetic still photo."""
        out_img_path = os.path.join(self.temp_dir, "annotated_out.jpg")
        result = self.processor.process_image(self.test_img_path, out_img_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["source_type"], "image")
        self.assertEqual(result["timestamp"]["label"], "Analysis Time")
        self.assertFalse(result["location"]["available"])
        self.assertIsNone(result["location"]["latitude"])
        self.assertIsNone(result["sensor_data"]["imu"])
        self.assertTrue(os.path.exists(out_img_path))

    def test_api_analyze_image_endpoint(self):
        """Test POST /api/analyze-image Flask endpoint with synthetic image upload."""
        with open(self.test_img_path, "rb") as f:
            data = {"image": (f, "test_upload.jpg")}
            response = self.client.post("/api/analyze-image", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        res_json = response.get_json()
        self.assertTrue(res_json["success"])
        self.assertEqual(res_json["source_type"], "image")
        self.assertIn("detections", res_json)
        self.assertIn("output_url", res_json["image"])

    def test_api_analyze_image_unsupported_format(self):
        """Verify that unsupported file formats (.txt) return 400 Bad Request."""
        invalid_path = os.path.join(self.temp_dir, "test.txt")
        with open(invalid_path, "w") as f:
            f.write("not an image")

        with open(invalid_path, "rb") as f:
            data = {"image": (f, "test.txt")}
            response = self.client.post("/api/analyze-image", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        res_json = response.get_json()
        self.assertFalse(res_json["success"])

if __name__ == "__main__":
    unittest.main()
