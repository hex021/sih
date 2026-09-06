import os
from typing import Dict, Any, Optional
from phase2.road_ai.incident_models import RashDrivingEvent, IncidentReport

# Global store for active incident reports (in-memory cache for prototype)
INCIDENT_REPORTS_STORE: Dict[str, IncidentReport] = {}

def create_rash_driving_event(
    track_id: int = 27,
    license_plate: str = "GJ01AB1234",
    timestamp: float = 14.25,
    timestamp_iso: str = "2026-09-02T21:42:31+05:30",
    latitude: float = 23.0225,
    longitude: float = 72.5714,
    address: str = "SG Highway Junction, Ahmedabad, Gujarat",
    speed_kmh: float = 84.5,
    speed_limit_kmh: float = 50.0,
    frame_number: int = 356,
    snapshot_path: str = "/processed/events/rash_driving/track_27_snapshot.jpg",
    plate_crop_path: str = "/processed/events/plates/plate_GJ01AB1234.jpg"
) -> RashDrivingEvent:
    """
    Creates and returns a structured RashDrivingEvent object based on vehicle telemetry.
    """
    return RashDrivingEvent(
        track_id=track_id,
        license_plate=license_plate,
        timestamp=timestamp,
        timestamp_iso=timestamp_iso,
        location={
            "latitude": latitude,
            "longitude": longitude,
            "address": address
        },
        vehicle_metadata={
            "class_name": "CAR",
            "color": "Silver Metallic",
            "speed_kmh": speed_kmh,
            "speed_limit_kmh": speed_limit_kmh,
            "severity": "CRITICAL" if speed_kmh > speed_limit_kmh + 20 else "WARNING"
        },
        evidence={
            "frame_number": frame_number,
            "snapshot_path": snapshot_path,
            "plate_crop_path": plate_crop_path,
            "trajectory_points": [[120, 450], [180, 410], [290, 320], [410, 210]]
        },
        status="VERIFIED"
    )

def generate_incident_report(event: RashDrivingEvent) -> IncidentReport:
    """
    Generates a formal IncidentReport from a RashDrivingEvent and stores it.
    """
    report_id = f"REP-{event.event_id}"
    report = IncidentReport(
        report_id=report_id,
        event=event,
        issued_by="UrbanPulse AI Safety Enforcement Engine",
        penalty_points=4,
        fine_amount_inr=2000
    )
    INCIDENT_REPORTS_STORE[report_id] = report
    return report

def get_sample_track27_incident_report() -> IncidentReport:
    """
    Convenience function returning the default sample Incident Report for Track #27 / GJ01AB1234.
    """
    event = create_rash_driving_event(
        track_id=27,
        license_plate="GJ01AB1234",
        timestamp=14.25,
        timestamp_iso="2026-09-02T21:42:31+05:30",
        latitude=23.0225,
        longitude=72.5714,
        address="SG Highway Junction, Ahmedabad, Gujarat",
        speed_kmh=84.5,
        speed_limit_kmh=50.0
    )
    return generate_incident_report(event)

def get_incident_report_by_id(report_id: str) -> Optional[IncidentReport]:
    """Retrieves stored incident report by report ID."""
    if report_id not in INCIDENT_REPORTS_STORE:
        # Generate sample on-demand if requesting sample report
        if "27" in report_id or "GJ01AB1234" in report_id:
            return get_sample_track27_incident_report()
        return None
    return INCIDENT_REPORTS_STORE[report_id]
