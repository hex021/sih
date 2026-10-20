import os
from typing import Dict, Any, Optional
from phase2.road_ai.incident_models import RashDrivingEvent, IncidentReport

# Global store for active incident reports (in-memory cache for prototype)
INCIDENT_REPORTS_STORE: Dict[str, IncidentReport] = {}

def create_rash_driving_event(
    track_id: int,
    license_plate: str = "UNREADABLE",
    timestamp: float = 0.0,
    timestamp_iso: str = "",
    latitude: float = 0.0,
    longitude: float = 0.0,
    address: str = "Onboard Bus Route",
    speed_kmh: float = 0.0,
    speed_limit_kmh: float = 50.0,
    frame_number: int = 0,
    snapshot_path: str = "",
    plate_crop_path: str = ""
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
            "class_name": "VEHICLE",
            "color": "Standard",
            "speed_kmh": speed_kmh,
            "speed_limit_kmh": speed_limit_kmh,
            "severity": "CRITICAL" if speed_kmh > speed_limit_kmh + 20 else "WARNING"
        },
        evidence={
            "frame_number": frame_number,
            "snapshot_path": snapshot_path,
            "plate_crop_path": plate_crop_path,
            "trajectory_points": []
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
        issued_by="NagarNetra AI Safety Enforcement Engine",
        penalty_points=4,
        fine_amount_inr=2000
    )
    INCIDENT_REPORTS_STORE[report_id] = report
    return report

def get_sample_track27_incident_report() -> IncidentReport:
    """
    Returns hardcoded sample Incident Report for Track #27 / GJ01AB1234
    with is_sample=True flag for frontend badge rendering.
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
        speed_limit_kmh=50.0,
        frame_number=356,
        snapshot_path="/processed/events/rash_driving/track_27_snapshot.jpg",
        plate_crop_path="/processed/events/plates/plate_GJ01AB1234.jpg"
    )
    report = IncidentReport(
        report_id="REP-SAMPLE-TRACK27",
        event=event,
        issued_by="NagarNetra AI Safety Enforcement Engine",
        penalty_points=4,
        fine_amount_inr=2000,
        is_sample=True
    )
    INCIDENT_REPORTS_STORE[report.report_id] = report
    return report

def get_incident_report_by_id(report_id: str) -> Optional[IncidentReport]:
    """Retrieves stored incident report by report ID."""
    if report_id not in INCIDENT_REPORTS_STORE:
        if "sample" in report_id.lower() or "27" in report_id:
            return get_sample_track27_incident_report()
        return None
    return INCIDENT_REPORTS_STORE.get(report_id)

