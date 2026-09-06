import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Set
from collections import deque
from phase1 import config
from phase1.models import TrackedObject

class Tracker:
    """
    YOLO Wrapper for video tracking.
    Manages persistent state of tracked vehicles and objects.
    Ensures single-inference execution per frame.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path if model_path else config.MODEL_PATH
        # Device auto-detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_path)
        
        # Bounded history of track centers: {track_id: deque([(cx, cy), ...])}
        self.track_history: Dict[int, deque] = {}
        # Inactive frame count for each track ID: {track_id: inactive_frame_count}
        self.inactive_counts: Dict[int, int] = {}
        
    def update(self, frame: np.ndarray, conf: float = None) -> List[TrackedObject]:
        """
        Processes a video frame, executes YOLO tracking, updates history,
        and returns a list of TrackedObject instances.
        """
        confidence = conf if conf is not None else config.CONFIDENCE_THRESHOLD
        
        # Run tracking on the frame.
        # persist=True enables tracking across frames.
        results = self.model.track(
            source=frame,
            persist=True,
            conf=confidence,
            iou=config.IOU_THRESHOLD,
            device=self.device,
            tracker=config.TRACKER_TYPE,
            verbose=False
        )
        
        tracked_objects = []
        if not results or len(results) == 0:
            return tracked_objects
            
        result = results[0]
        boxes = result.boxes
        
        # If no boxes are detected or tracking IDs are not assigned (e.g. tracking not initialized)
        if boxes is None or len(boxes) == 0:
            # All tracks in history become inactive in this frame
            self._cleanup_inactive_tracks(set())
            return tracked_objects
            
        # Get bounding box coordinates, class ids, confidences, and tracking ids
        xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        conf_list = boxes.conf.tolist() if boxes.conf is not None else []
        cls_list = boxes.cls.tolist() if boxes.cls is not None else []
        id_list = boxes.id.tolist() if boxes.id is not None else []
        
        # Track IDs seen in the current frame
        active_ids = set()
        
        for idx in range(len(xyxy_list)):
            cls_id = int(cls_list[idx])
            coco_cls_name = result.names[cls_id]
            
            # Map COCO class name to our standardized internal class
            if coco_cls_name in config.CLASS_MAP:
                class_name = config.CLASS_MAP[coco_cls_name]
                confidence_score = conf_list[idx]
                bbox = (xyxy_list[idx][0], xyxy_list[idx][1], xyxy_list[idx][2], xyxy_list[idx][3])
                
                # Calculate current center coordinates
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                
                # YOLO tracking might not assign an ID to some frames/detections if they are unconfirmed.
                # If no ID, we skip it since we can't reliably track/count it.
                if idx < len(id_list) and id_list[idx] is not None:
                    track_id = int(id_list[idx])
                    active_ids.add(track_id)
                else:
                    continue
                
                # Manage track center history
                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=config.TRACK_HISTORY_LIMIT)
                    previous_center_x = center_x
                    previous_center_y = center_y
                else:
                    # Get the last recorded center point
                    prev_point = self.track_history[track_id][-1]
                    previous_center_x = prev_point[0]
                    previous_center_y = prev_point[1]
                
                # Record the new center point and reset inactive counter
                self.track_history[track_id].append((center_x, center_y))
                self.inactive_counts[track_id] = 0
                
                tracked_objects.append(TrackedObject(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=confidence_score,
                    bbox=bbox,
                    center_x=center_x,
                    center_y=center_y,
                    previous_center_x=previous_center_x,
                    previous_center_y=previous_center_y
                ))
                
        # Clean up stale track histories
        self._cleanup_inactive_tracks(active_ids)
        
        return tracked_objects
        
    def _cleanup_inactive_tracks(self, active_ids: Set[int]):
        """
        Increments inactive counts for all stored tracks not present in active_ids.
        Deletes tracking history for IDs inactive for more than 30 frames.
        """
        # Collect IDs currently in history
        known_ids = list(self.track_history.keys())
        
        for track_id in known_ids:
            if track_id not in active_ids:
                # Increment inactive counter
                self.inactive_counts[track_id] = self.inactive_counts.get(track_id, 0) + 1
                
                # If inactive for too long, delete history to release memory
                if self.inactive_counts[track_id] > 30:
                    del self.track_history[track_id]
                    del self.inactive_counts[track_id]
            else:
                self.inactive_counts[track_id] = 0
