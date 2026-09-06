import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class RashDrivingEvent:
    """
    Data model representing a single Rash Driving / Traffic Safety violation event.
    Integrates persistent vehicle tracking ID, ANPR license plate, timestamp,
    GPS telemetry, and visual evidence snapshots.
    """
    event_id: str = field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8].upper()}")
    event_type: str = "RASH DRIVING"
    track_id: int = 27
    license_plate: str = "GJ01AB1234"
    timestamp: float = 14.25  # Offset in seconds
    timestamp_iso: str = "2026-09-02T21:42:31+05:30"
    location: Dict[str, Any] = field(default_factory=lambda: {
        "latitude": 23.0225,
        "longitude": 72.5714,
        "address": "SG Highway Junction, Ahmedabad, Gujarat"
    })
    vehicle_metadata: Dict[str, Any] = field(default_factory=lambda: {
        "class_name": "CAR",
        "color": "Silver Metallic",
        "speed_kmh": 84.5,
        "speed_limit_kmh": 50.0,
        "severity": "CRITICAL"
    })
    evidence: Dict[str, Any] = field(default_factory=lambda: {
        "frame_number": 356,
        "snapshot_path": "/processed/events/rash_driving/track_27_snapshot.jpg",
        "plate_crop_path": "/processed/events/plates/plate_GJ01AB1234.jpg",
        "trajectory_points": [[120, 450], [180, 410], [290, 320], [410, 210]]
    })
    status: str = "VERIFIED"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes event object into standard JSON dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "track_id": self.track_id,
            "license_plate": self.license_plate,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "location": self.location,
            "vehicle_metadata": self.vehicle_metadata,
            "evidence": self.evidence,
            "status": self.status
        }


@dataclass
class IncidentReport:
    """
    Formal Incident Report generated from a Rash Driving event,
    ready for traffic enforcement export, printing, or API payload.
    """
    report_id: str
    event: RashDrivingEvent
    issued_by: str = "UrbanPulse AI Traffic Enforcement System"
    penalty_points: int = 4
    fine_amount_inr: int = 2000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "issued_by": self.issued_by,
            "penalty_points": self.penalty_points,
            "fine_amount_inr": self.fine_amount_inr,
            "event": self.event.to_dict()
        }

    def generate_html_summary(self) -> str:
        """Returns clean HTML markup snippet for UI modal and report viewing."""
        ev = self.event
        vm = ev.vehicle_metadata
        loc = ev.location
        return f"""
        <div class="incident-report-document">
            <div class="report-header">
                <h2>SAFETY INCIDENT VIOLATION REPORT</h2>
                <span class="report-badge badge-critical">{ev.event_type}</span>
            </div>
            <div class="report-meta-grid">
                <div><strong>Report ID:</strong> {self.report_id}</div>
                <div><strong>Vehicle Track ID:</strong> #{ev.track_id}</div>
                <div><strong>License Plate:</strong> <span class="plate-number">{ev.license_plate}</span></div>
                <div><strong>Timestamp:</strong> {ev.timestamp_iso} ({ev.timestamp}s)</div>
                <div><strong>GPS Coordinates:</strong> {loc['latitude']}° N, {loc['longitude']}° E</div>
                <div><strong>Location:</strong> {loc['address']}</div>
                <div><strong>Observed Speed:</strong> <span class="speed-warning">{vm['speed_kmh']} km/h</span> (Limit: {vm['speed_limit_kmh']} km/h)</div>
                <div><strong>Status:</strong> {ev.status}</div>
            </div>
            <div class="report-fine-summary">
                <span>Penalty Points: <strong>{self.penalty_points}</strong></span> | 
                <span>Proposed Fine: <strong>₹{self.fine_amount_inr}</strong></span>
            </div>
        </div>
        """
