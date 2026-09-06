import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List
from phase1 import config
from phase1.models import Detection

class Detector:
    """
    YOLO Wrapper for still-image object detection.
    Intended for standalone image testing and independent verification.
    Not to be called inside the frame loop of the video tracking pipeline.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path if model_path else config.MODEL_PATH
        # Device auto-detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load YOLO model
        self.model = YOLO(self.model_path)
        
    def detect(self, image: np.ndarray, conf: float = None) -> List[Detection]:
        """
        Perform detection on a still image and return list of Detection.
        """
        confidence = conf if conf is not None else config.CONFIDENCE_THRESHOLD
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=config.IOU_THRESHOLD,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if not results or len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            coco_cls_name = result.names[cls_id]
            
            # Map COCO class to standardized class if it exists in target map
            if coco_cls_name in config.CLASS_MAP:
                class_name = config.CLASS_MAP[coco_cls_name]
                confidence_score = float(box.conf[0].item())
                
                # Get bounding box coordinates [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                
                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence_score,
                    bbox=bbox
                ))
                
        return detections
