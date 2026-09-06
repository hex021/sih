from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

@dataclass
class PotholeDetection:
    """
    Internal representation of a single pothole detection in a frame.
    """
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center_x: float
    center_y: float
    frame_number: int
    timestamp: float

@dataclass
class PotholeEvent:
    """
    Common event structure representing a confirmed unique physical pothole.
    Integrates media snapshots and prepares slots for future GPS and IMU data.
    """
    event_id: str
    event_type: str = "POTHOLE"
    timestamp: float = 0.0          # Time offset in video (seconds)
    confidence: float = 0.0
    location: Optional[Dict[str, float]] = None # Nullable for future GPS: {"latitude": None, "longitude": None}
    source: Dict[str, Any] = field(default_factory=lambda: {"camera": "uploaded_video", "vehicle_id": None})
    media: Dict[str, str] = field(default_factory=lambda: {"frame": 0, "snapshot": "", "video": None})
    sensor_data: Dict[str, Any] = field(default_factory=lambda: {"gps": None, "imu": None})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the event into a standard JSON-compatible dictionary.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "location": self.location,
            "source": self.source,
            "media": self.media,
            "sensor_data": self.sensor_data,
            "metadata": self.metadata
        }
