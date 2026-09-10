import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from phase1.models import TrackedObject
from phase2.anpr import plate_config
from phase2.anpr.plate_models import PlateDetection

logger = logging.getLogger("SIH_Server")

class PlateDetector:
    """
    Number Plate Detector & Spatial Vehicle-Plate Association Engine.
    Locates number plate regions within vehicle bounding boxes and assigns
    each detected plate to its corresponding vehicle Track ID.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.yolo_model = None
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                logger.info(f"YOLO Plate Detector model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO plate model from {model_path}: {e}")

    def _compute_iou(self, boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

    def is_plate_inside_vehicle(
        self,
        plate_box: Tuple[float, float, float, float],
        vehicle_box: Tuple[float, float, float, float]
    ) -> bool:
        """
        Spatial Association Check:
        Verifies that plate bounding box center point lies inside or overlaps substantially with vehicle bbox.
        """
        px1, py1, px2, py2 = plate_box
        vx1, vy1, vx2, vy2 = vehicle_box

        # Plate center point
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0

        # Center point containment check with 10% margin
        margin_x = (vx2 - vx1) * 0.1
        margin_y = (vy2 - vy1) * 0.1

        inside_center = (vx1 - margin_x <= pcx <= vx2 + margin_x) and (vy1 - margin_y <= pcy <= vy2 + margin_y)
        iou = self._compute_iou(plate_box, vehicle_box)

        return inside_center or (iou >= plate_config.MIN_PLATE_VEHICLE_IOU)

    def detect_plates_in_vehicles(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[TrackedObject],
        frame_number: int,
        timestamp: float,
        top_n_vehicles: int = plate_config.DEFAULT_TOP_N_VEHICLES
    ) -> Dict[int, Tuple[PlateDetection, np.ndarray]]:
        """
        Detects number plates within tracked vehicle regions.
        PHASE B requirement: Only attempts plate detection / OCR on the largest N vehicle boxes
        per frame (nearest vehicles have the biggest plates). Parameterized with default N=3.
        Returns a mapping: track_id -> (PlateDetection, plate_crop_image).
        """
        results: Dict[int, Tuple[PlateDetection, np.ndarray]] = {}
        if frame is None or not tracked_vehicles:
            return results

        h_frame, w_frame = frame.shape[:2]

        # 1. Filter non-vehicle objects (e.g. PERSON, RIDER)
        valid_vehicles = [v for v in tracked_vehicles if v.class_name not in ["PERSON", "RIDER"]]
        if not valid_vehicles:
            return results

        # 2. Sort by vehicle bounding box area in descending order (largest vehicles first)
        def _veh_area(v: TrackedObject) -> float:
            return max(0.0, float((v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1])))

        sorted_vehicles = sorted(valid_vehicles, key=_veh_area, reverse=True)

        # 3. Restrict to top N vehicle boxes per frame
        target_vehicles = sorted_vehicles[:top_n_vehicles]

        for veh in target_vehicles:
            vx1, vy1, vx2, vy2 = [int(v) for v in veh.bbox]
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w_frame, vx2), min(h_frame, vy2)

            if (vx2 - vx1) < 20 or (vy2 - vy1) < 20:
                continue

            vehicle_crop = frame[vy1:vy2, vx1:vx2]

            plate_box_rel, confidence = self._locate_plate_region_in_crop(vehicle_crop)
            if plate_box_rel is None:
                continue

            rx1, ry1, rx2, ry2 = plate_box_rel

            # Convert relative crop coordinates to global frame coordinates
            px1 = vx1 + rx1
            py1 = vy1 + ry1
            px2 = vx1 + rx2
            py2 = vy1 + ry2

            global_plate_box = (float(px1), float(py1), float(px2), float(py2))

            # Spatial association check
            if not self.is_plate_inside_vehicle(global_plate_box, (float(vx1), float(vy1), float(vx2), float(vy2))):
                continue

            plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]
            if plate_crop.size == 0:
                continue

            detection = PlateDetection(
                bbox=global_plate_box,
                confidence=round(confidence, 2),
                frame_number=frame_number,
                timestamp=round(timestamp, 2),
                vehicle_track_id=veh.track_id
            )

            results[veh.track_id] = (detection, plate_crop)

        return results


    @staticmethod
    def compute_crop_quality_score(plate_crop: np.ndarray, detector_conf: float = 0.8) -> float:
        """
        Evaluates plate crop quality using resolution, Laplacian variance sharpness, contrast, and detector confidence.
        """
        if plate_crop is None or plate_crop.size == 0:
            return 0.0
        h, w = plate_crop.shape[:2]
        area = w * h
        if area < 150 or detector_conf <= 0.0:
            return 0.0


        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())

        size_factor = min(1.0, area / 15000.0)
        sharpness_factor = min(1.0, laplacian_var / 500.0)
        contrast_factor = min(1.0, contrast / 64.0)

        quality_score = (size_factor * 0.4) + (sharpness_factor * 0.3) + (contrast_factor * 0.1) + (detector_conf * 0.2)
        return round(float(quality_score), 3)


    def _locate_plate_region_in_crop(self, vehicle_crop: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Morphological & Rectangular Contour Analysis to detect license plate region inside vehicle crop.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, 0.0

        h_crop, w_crop = vehicle_crop.shape[:2]
        crop_area = float(h_crop * w_crop)

        # 1. Grayscale & Noise Reduction
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Sobel Horizontal Gradient (highlight vertical edges of plate letters)
        sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

        # 3. Otsu Binarization & Morphological Closing to connect characters
        _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 4. Find Contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_conf = 0.0

        # Number plate expected aspect ratio ~ 3.0 to 5.5
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = float(w * h)
            area_ratio = area / crop_area

            # Filter bounds
            if area_ratio < plate_config.PLATE_MIN_AREA_RATIO or area_ratio > plate_config.PLATE_MAX_AREA_RATIO:
                continue

            aspect_ratio = w / float(h + 1e-5)
            if (plate_config.PLATE_MIN_ASPECT_RATIO <= aspect_ratio <= 5.5) and h >= 6 and w >= 18:
                # Plate location preference: lower 60% of vehicle crop
                y_center_rel = (y + h / 2.0) / float(h_crop)
                conf = 0.70 + (0.20 if y_center_rel > 0.35 else 0.05)

                if conf > best_conf:
                    best_conf = conf
                    best_box = (x, y, x + w, y + h)


        # Fallback: Lower central rectangle of vehicle crop if morphological detection misses
        if best_box is None and h_crop > 40 and w_crop > 60:
            fx1 = int(w_crop * 0.25)
            fy1 = int(h_crop * 0.55)
            fx2 = int(w_crop * 0.75)
            fy2 = int(h_crop * 0.85)
            best_box = (fx1, fy1, fx2, fy2)
            best_conf = 0.60

        return best_box, best_conf
