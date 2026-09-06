import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

@dataclass
class PlateDetection:
    """
    Internal representation of a detected number plate region inside a vehicle bounding box.
    """
    bbox: Tuple[float, float, float, float]  # (px1, py1, px2, py2) relative to full frame
    confidence: float
    frame_number: int
    timestamp: float
    vehicle_track_id: int

@dataclass
class OCRResult:
    """
    Internal representation of an OCR extraction on a cropped number plate image.
    """
    raw_ocr: str
    normalized_text: str
    ocr_confidence: float
    is_valid_pattern: bool = False

@dataclass
class VehicleRecord:
    """
    Formal Phase 2.2 Vehicle Identification Event Record.
    Associates vehicle track, detected number plate, OCR extraction,
    timestamp, evidence snapshots, and null GPS/IMU placeholders.
    """
    event_id: str = field(default_factory=lambda: f"VEH-{uuid.uuid4().hex[:6].upper()}")
    event_type: str = "VEHICLE_IDENTIFICATION"
    track_id: int = 0
    vehicle_type: str = "CAR"
    vehicle_confidence: float = 0.0
    registration_number: str = "UNREADABLE"
    raw_ocr: str = "UNREADABLE"
    ocr_confidence: float = 0.0
    plate_detection_confidence: float = 0.0
    timestamp: float = 0.0
    frame_number: int = 0
    location: Dict[str, Optional[float]] = field(default_factory=lambda: {"latitude": None, "longitude": None})
    sensor_data: Dict[str, Any] = field(default_factory=lambda: {"gps": None, "imu": None})
    media: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "vehicle_snapshot": None,
        "plate_snapshot": None,
        "video": None
    })
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "source": "video",
        "camera_id": None
    })

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes record into exact JSON schema specified for Phase 2.2.
        """
        # Calculate documented aggregate confidence score
        # Final identification confidence = (vehicle_conf * 0.2) + (plate_conf * 0.3) + (ocr_conf * 0.5) if readable else 0.0
        final_conf = round(
            (self.vehicle_confidence * 0.2) + (self.plate_detection_confidence * 0.3) + (self.ocr_confidence * 0.5),
            2
        ) if self.registration_number != "UNREADABLE" else 0.0

        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "vehicle": {
                "track_id": self.track_id,
                "type": self.vehicle_type,
                "detection_confidence": round(self.vehicle_confidence, 2)
            },
            "number_plate": {
                "text": self.registration_number,
                "raw_ocr": self.raw_ocr,
                "ocr_confidence": round(self.ocr_confidence, 2),
                "plate_detection_confidence": round(self.plate_detection_confidence, 2)
            },
            "final_identification_confidence": final_conf,
            "timestamp": round(self.timestamp, 2),
            "frame_number": self.frame_number,
            "location": self.location,
            "sensor_data": self.sensor_data,
            "media": self.media,
            "metadata": self.metadata
        }
