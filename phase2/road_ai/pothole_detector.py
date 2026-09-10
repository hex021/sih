import os
import sys
import uuid
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from typing import List, Dict, Any, Tuple, Optional

from . import pothole_config
from .pothole_models import PotholeDetection, PotholeEvent

logger = logging.getLogger("SIH_Server")

class PotholeDetector:
    """
    Modular Road AI perception layer for pothole detection.
    Performs YOLO inference, frame coordinate mapping, spatial-temporal
    event deduplication, and evidence snapshot generation.
    """
    def __init__(
        self,
        model_path: str = None,
        gps_track=None,
        conf_threshold: float = 0.35,
        road_roi: Tuple[float, float, float, float] = (0.25, 0.60, 0.75, 0.95),
        green_max_fraction: float = 0.35,
        sat_max_mean: float = 220.0
    ):
        self.gps_track = gps_track
        self.model_path = model_path if model_path else pothole_config.POTHOLE_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.road_roi = road_roi
        self.green_max_fraction = green_max_fraction
        self.sat_max_mean = sat_max_mean
        
        # Verify model file exists before loading
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Pothole model weights not found at: {self.model_path}\n"
                "Please run: 'python phase2/road_ai/setup_pothole_model.py' to download the weights file, "
                "or place your custom trained weights file there."
            )
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_path)
        
        # Deduplication Tracker State
        # Each active track is: {"event_id": str, "bbox": Tuple, "last_frame": int, "max_conf": float, "event": PotholeEvent}
        self.active_tracks: List[Dict[str, Any]] = []
        
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 1,
        timestamp: float = 0.0,
        road_roi: Optional[Tuple[float, float, float, float]] = None,
        is_still_image: bool = False
    ) -> Tuple[List[PotholeDetection], List[PotholeEvent]]:
        """
        Runs inference on a single frame (or ROI crop), translates bounding boxes
        back to full-frame coordinates, applies geometric & foliage/saturation filters, and runs deduplication.
        """
        h_frame, w_frame = frame.shape[:2]

        if road_roi is not None:
            roi_coords = road_roi
        elif is_still_image:
            roi_coords = (0.0, 0.0, 1.0, 1.0)
        else:
            roi_coords = getattr(self, "road_roi", (0.25, 0.60, 0.75, 0.95))

        conf_thresh = getattr(self, "conf_threshold", 0.35)
        max_green_frac = getattr(self, "green_max_fraction", 0.35)
        max_sat_mean = getattr(self, "sat_max_mean", 220.0)

        # 0. Crop frame to road surface ROI region for inference
        crop_x1 = max(0, min(int(w_frame * roi_coords[0]), w_frame - 1))
        crop_y1 = max(0, min(int(h_frame * roi_coords[1]), h_frame - 1))
        crop_x2 = max(crop_x1 + 1, min(int(w_frame * roi_coords[2]), w_frame))
        crop_y2 = max(crop_y1 + 1, min(int(h_frame * roi_coords[3]), h_frame))

        roi_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # 1. Run YOLO inference on road ROI crop
        results = self.model(
            source=roi_crop,
            conf=conf_thresh,
            iou=pothole_config.POTHOLE_IOU_THRESHOLD,
            imgsz=1280,
            device=self.device,
            verbose=False
        )
        
        detections: List[PotholeDetection] = []
        if not results or len(results) == 0:
            self._prune_stale_tracks(frame_number)
            return detections, []
            
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self._prune_stale_tracks(frame_number)
            return detections, []
            
        xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        conf_list = boxes.conf.tolist() if boxes.conf is not None else []
        cls_list = boxes.cls.tolist() if boxes.cls is not None else []
        
        frame_area = float(w_frame * h_frame)

        # 2. Extract detections, translate to full-frame space, and apply geometric & color/saturation filters
        for idx in range(len(xyxy_list)):
            cls_id = int(cls_list[idx])
            class_name = result.names[cls_id].lower()

            # Verify if class is a pothole
            if "pothole" in class_name or class_name in ["0", "pothole"]:
                # Translate bounding box coordinates from crop space back to full-frame space
                fx1 = xyxy_list[idx][0] + crop_x1
                fy1 = xyxy_list[idx][1] + crop_y1
                fx2 = xyxy_list[idx][2] + crop_x1
                fy2 = xyxy_list[idx][3] + crop_y1
                bbox = (fx1, fy1, fx2, fy2)

                box_w = fx2 - fx1
                box_h = fy2 - fy1

                # Geometric Filter 1: Reject if invalid dimensions or abnormal aspect ratio in video
                if box_w <= 0 or box_h <= 0:
                    continue
                if not is_still_image:
                    if (box_h / box_w) > 1.3:
                        aspect_ratio = box_h / max(box_w, 1e-5)
                        logger.debug(f"Pothole box rejected: aspect ratio {aspect_ratio:.2f} > 1.3 (taller than wide)")
                        continue

                # Geometric Filter 2: Reject if box area exceeds maximum threshold
                box_area = box_w * box_h
                max_area_ratio = 0.90 if is_still_image else 0.35
                if box_area > max_area_ratio * frame_area:
                    area_pct = (box_area / frame_area) * 100.0
                    logger.debug(f"Pothole box rejected: area ratio {area_pct:.2f}% > {max_area_ratio*100:.0f}% of frame area")
                    continue

                # Geometric Filter 3: Reject tiny noise
                if box_area < 0.0003 * frame_area:
                    continue

                # Color/Foliage Filter 1: Vegetation Green Range
                bx1, by1, bx2, by2 = max(0, int(fx1)), max(0, int(fy1)), min(w_frame, int(fx2)), min(h_frame, int(fy2))
                box_crop = frame[by1:by2, bx1:bx2]
                if box_crop is None or box_crop.size == 0 or box_crop.shape[0] == 0 or box_crop.shape[1] == 0:
                    continue

                hsv = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 40, 40], dtype=np.uint8)
                upper_green = np.array([85, 255, 255], dtype=np.uint8)
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                green_pixels = cv2.countNonZero(green_mask)
                total_pixels = float(box_crop.shape[0] * box_crop.shape[1])
                green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0

                if green_ratio > max_green_frac:
                    logger.debug(f"Pothole box rejected: vegetation (green fraction {green_ratio*100.0:.2f}% > {max_green_frac*100.0:.0f}%)")
                    continue

                # Color/Foliage Filter 2: Extreme Artificial Saturation
                mean_sat = float(hsv[:, :, 1].mean()) if hsv.size > 0 else 0.0
                effective_sat_limit = 240.0 if is_still_image else max_sat_mean
                if mean_sat > effective_sat_limit:
                    logger.debug(f"Pothole box rejected: high saturation (mean sat {mean_sat:.1f} > {effective_sat_limit:.1f})")
                    continue

                center_x = (fx1 + fx2) / 2.0
                center_y = (fy1 + fy2) / 2.0

                detections.append(PotholeDetection(
                    class_name="POTHOLE",
                    confidence=conf_list[idx],
                    bbox=bbox,
                    center_x=center_x,
                    center_y=center_y,
                    frame_number=frame_number,
                    timestamp=timestamp
                ))

        # 3. Deduplicate events spatially and temporally with non-strict confirmation window
        new_events: List[PotholeEvent] = []
        min_confirmations = 1 if is_still_image else getattr(pothole_config, "POTHOLE_MIN_CONFIRMATIONS", 2)

        for det in detections:
            matched_track = self._find_matching_track(det.bbox)

            if matched_track:
                # Update existing candidate track
                matched_track["last_frame"] = frame_number
                matched_track["bbox"] = det.bbox
                matched_track["obs_count"] = matched_track.get("obs_count", 1) + 1
                if det.confidence > matched_track["max_conf"]:
                    matched_track["max_conf"] = det.confidence
                    if matched_track.get("event"):
                        matched_track["event"].confidence = det.confidence

                # Validate candidate event if confirmation threshold reached and not yet emitted
                if matched_track["obs_count"] >= min_confirmations and not matched_track.get("emitted", False):
                    if matched_track.get("event") is None:
                        event_id = matched_track["event_id"]
                        snapshot_path = self._save_snapshot(frame, det.bbox, event_id)
                        event = PotholeEvent(
                            event_id=event_id,
                            timestamp=round(timestamp, 2),
                            confidence=round(matched_track["max_conf"], 2),
                            location=(self.gps_track.get_location(frame_number) if self.gps_track else None),
                            source={"camera": "uploaded_video", "vehicle_id": None},
                            media={"frame": frame_number, "snapshot": snapshot_path, "video": None},
                            sensor_data={"gps": (self.gps_track.get_location(frame_number) if self.gps_track else None), "imu": None}
                        )
                        matched_track["event"] = event

                    matched_track["emitted"] = True
                    new_events.append(matched_track["event"])
            else:
                # Initialize new candidate track
                event_id = f"pothole_{str(uuid.uuid4())[:8]}"
                new_track = {
                    "event_id": event_id,
                    "bbox": det.bbox,
                    "last_frame": frame_number,
                    "max_conf": det.confidence,
                    "obs_count": 1,
                    "emitted": False,
                    "event": None,
                    "frame": frame
                }

                # Single-frame override for high-confidence detections if min_confirmations == 1
                if min_confirmations <= 1:
                    snapshot_path = self._save_snapshot(frame, det.bbox, event_id)
                    event = PotholeEvent(
                        event_id=event_id,
                        timestamp=round(timestamp, 2),
                        confidence=round(det.confidence, 2),
                        location=(self.gps_track.get_location(frame_number) if self.gps_track else None),
                        source={"camera": "uploaded_video", "vehicle_id": None},
                        media={"frame": frame_number, "snapshot": snapshot_path, "video": None},
                        sensor_data={"gps": (self.gps_track.get_location(frame_number) if self.gps_track else None), "imu": None}
                    )
                    new_track["event"] = event
                    new_track["emitted"] = True
                    new_events.append(event)

                self.active_tracks.append(new_track)

        # 4. Prune stale tracks
        self._prune_stale_tracks(frame_number)

        return detections, new_events

        
    def _find_matching_track(self, bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """
        Finds an active tracked pothole with spatial overlap above the IoU threshold.
        """
        best_iou = 0.0
        best_match = None
        
        for track in self.active_tracks:
            iou = self._compute_iou(bbox, track["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = track
                
        if best_iou >= pothole_config.POTHOLE_DEDUPLICATION_IOU:
            return best_match
        return None
        
    def _prune_stale_tracks(self, current_frame: int):
        """
        Prunes tracks that have not been seen for longer than the cooldown frame limit.
        """
        self.active_tracks = [
            t for t in self.active_tracks 
            if (current_frame - t["last_frame"]) <= pothole_config.POTHOLE_DEDUPLICATION_COOLDOWN_FRAMES
        ]
        
    def _compute_iou(self, boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """
        Computes Intersection over Union (IoU) of two bounding boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        unionArea = boxAArea + boxBArea - interArea
        if unionArea <= 0:
            return 0.0
        return interArea / unionArea
        
    def _save_snapshot(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], event_id: str) -> str:
        """
        Crops the pothole area from the frame with padding and saves it to disk.
        Returns the relative URL path for frontend rendering.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Crop dimensions with a 15% padding margin
        pad_w = int((x2 - x1) * 0.15)
        pad_h = int((y2 - y1) * 0.15)
        
        px1 = max(0, x1 - pad_w)
        py1 = max(0, y1 - pad_h)
        px2 = min(w, x2 + pad_w)
        py2 = min(h, y2 + pad_h)
        
        crop = frame[py1:py2, px1:px2]
        
        os.makedirs(pothole_config.EVENTS_FOLDER, exist_ok=True)
        filename = f"event_{event_id}.jpg"
        file_path = os.path.join(pothole_config.EVENTS_FOLDER, filename)
        
        # Save snapshot safely
        if crop is not None and crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
            cv2.imwrite(file_path, crop)
        else:
            logger.warning(f"Skipped saving pothole snapshot for event {event_id}: empty crop array.")
        
        # Return URL format served by static route
        return f"/processed/events/pothole/{filename}"
