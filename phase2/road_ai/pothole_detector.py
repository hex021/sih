import os
import sys
import uuid
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple

from . import pothole_config
from .pothole_models import PotholeDetection, PotholeEvent

class PotholeDetector:
    """
    Modular Road AI perception layer for pothole detection.
    Performs YOLO inference, frame coordinate mapping, spatial-temporal
    event deduplication, and evidence snapshot generation.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path if model_path else pothole_config.POTHOLE_MODEL_PATH
        
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
        
    def detect(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Tuple[List[PotholeDetection], List[PotholeEvent]]:
        """
        Runs inference on a single frame, filters by confidence, maps coordinates,
        and runs E2E spatial-temporal deduplication to generate unique road events.
        """
        # 1. Run YOLO inference
        results = self.model(
            source=frame,
            conf=pothole_config.POTHOLE_CONFIDENCE_THRESHOLD,
            iou=pothole_config.POTHOLE_IOU_THRESHOLD,
            imgsz=640,
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
        
        # 2. Extract standard detections
        for idx in range(len(xyxy_list)):
            cls_id = int(cls_list[idx])
            class_name = result.names[cls_id].lower()
            
            # Verify if class is a pothole
            if "pothole" in class_name:
                bbox = (xyxy_list[idx][0], xyxy_list[idx][1], xyxy_list[idx][2], xyxy_list[idx][3])
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                
                detections.append(PotholeDetection(
                    class_name="POTHOLE",
                    confidence=conf_list[idx],
                    bbox=bbox,
                    center_x=center_x,
                    center_y=center_y,
                    frame_number=frame_number,
                    timestamp=timestamp
                ))
                
        # 3. Deduplicate events spatially and temporally
        new_events: List[PotholeEvent] = []
        h, w = frame.shape[:2]
        
        for det in detections:
            matched_track = self._find_matching_track(det.bbox)
            
            if matched_track:
                # Update existing track
                matched_track["last_frame"] = frame_number
                matched_track["bbox"] = det.bbox
                if det.confidence > matched_track["max_conf"]:
                    matched_track["max_conf"] = det.confidence
                    matched_track["event"].confidence = det.confidence
            else:
                # Generate new unique event
                event_id = f"pothole_{str(uuid.uuid4())[:8]}"
                
                # Crop and save crop-snapshot as evidence
                snapshot_relative_path = self._save_snapshot(frame, det.bbox, event_id)
                
                # Initialize standard event structure
                event = PotholeEvent(
                    event_id=event_id,
                    timestamp=round(timestamp, 2),
                    confidence=round(det.confidence, 2),
                    location=None, # Nullable for future GPS integration
                    source={"camera": "uploaded_video", "vehicle_id": None},
                    media={
                        "frame": frame_number,
                        "snapshot": snapshot_relative_path,
                        "video": None
                    },
                    sensor_data={"gps": None, "imu": None}
                )
                
                new_track = {
                    "event_id": event_id,
                    "bbox": det.bbox,
                    "last_frame": frame_number,
                    "max_conf": det.confidence,
                    "event": event
                }
                
                self.active_tracks.append(new_track)
                new_events.append(event)
                
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
        
        # Save snapshot
        cv2.imwrite(file_path, crop)
        
        # Return URL format served by static route
        return f"/processed/events/pothole/{filename}"
