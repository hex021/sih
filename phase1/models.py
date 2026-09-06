from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    """
    Standardized internal representation of an object detection.
    Decoupled from YOLO or any specific model API.
    """
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class TrackedObject:
    """
    Standardized internal representation of a tracked object across frames.
    Maintains center history for line crossing and direction detection.
    """
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center_x: float
    center_y: float
    previous_center_x: float
    previous_center_y: float
