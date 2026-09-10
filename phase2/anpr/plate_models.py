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
    raw_ocr: str = ""
    normalized_text: Optional[str] = None
    plate: Optional[str] = None
    ocr_confidence: float = 0.0
    plate_confidence: float = 0.0
    plate_status: str = "no_text"  # 'validated' | 'format_rejected' | 'low_confidence' | 'no_text'
    is_valid_pattern: bool = False
    raw_crop: Optional[Any] = None
    preprocessed_crop: Optional[Any] = None

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
    plate: Optional[str] = None
    registration_number: Optional[str] = "UNREADABLE"
    ocr_raw: str = ""
    raw_ocr: str = "UNREADABLE"
    plate_confidence: float = 0.0
    ocr_confidence: float = 0.0
    plate_status: str = "no_text"  # 'validated' | 'format_rejected' | 'low_confidence' | 'no_text'
    plate_detection_confidence: float = 0.0
    timestamp: float = 0.0
    frame_number: int = 0
    location: Dict[str, Optional[float]] = field(default_factory=lambda: {"latitude": None, "longitude": None})
    sensor_data: Dict[str, Any] = field(default_factory=lambda: {"gps": None, "imu": None})
    media: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "vehicle_snapshot": None,
        "plate_snapshot": None,
        "raw_plate_snapshot": None,
        "preproc_plate_snapshot": None,
        "video": None
    })
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "source": "video",
        "camera_id": None
    })
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes record into exact JSON schema specified for Phase 2.2,
        with C4 fields inside attrs.
        """
        # Calculate documented aggregate confidence score
        # Final identification confidence = (vehicle_conf * 0.2) + (plate_conf * 0.3) + (ocr_conf * 0.5) if validated plate else 0.0
        active_conf = self.plate_confidence if self.plate_confidence > 0 else self.ocr_confidence
        final_conf = round(
            (self.vehicle_confidence * 0.2) + (self.plate_detection_confidence * 0.3) + (active_conf * 0.5),
            2
        ) if self.plate is not None else 0.0

        # C4 — Record outcome explicitly inside attrs
        attrs_dict = {
            "plate": self.plate,
            "plate_confidence": round(active_conf, 2),
            "plate_status": self.plate_status,
            "ocr_raw": self.ocr_raw if self.ocr_raw else (self.raw_ocr if self.raw_ocr != "UNREADABLE" else ""),
            **self.attrs
        }

        # Keep legacy string registration_number aligned
        display_reg = self.plate if self.plate is not None else (
            self.registration_number if self.registration_number != "UNREADABLE" else "UNREADABLE"
        )

        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "vehicle": {
                "track_id": self.track_id,
                "type": self.vehicle_type,
                "detection_confidence": round(self.vehicle_confidence, 2)
            },
            "number_plate": {
                "text": display_reg,
                "raw_ocr": attrs_dict["ocr_raw"],
                "ocr_confidence": round(active_conf, 2),
                "plate_detection_confidence": round(self.plate_detection_confidence, 2),
                "plate_status": self.plate_status
            },
            "plate": self.plate,
            "plate_confidence": round(active_conf, 2),
            "plate_status": self.plate_status,
            "ocr_raw": attrs_dict["ocr_raw"],
            "attrs": attrs_dict,
            "final_identification_confidence": final_conf,
            "timestamp": round(self.timestamp, 2),
            "frame_number": self.frame_number,
            "location": self.location,
            "sensor_data": self.sensor_data,
            "media": self.media,
            "metadata": self.metadata
        }

