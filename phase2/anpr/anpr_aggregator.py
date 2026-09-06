import os
import cv2
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from phase1.models import TrackedObject
from phase2.anpr import plate_config
from phase2.anpr.plate_models import VehicleRecord, OCRResult, PlateDetection

logger = logging.getLogger("SIH_Server")

class ANPRAggregator:
    """
    Multi-Frame OCR Aggregator & Duplicate Suppression Engine.
    Collects number plate OCR observations across frames for each tracked vehicle ID,
    selects the most reliable registration number via temporal voting, and produces
    exactly one primary VehicleRecord per tracked vehicle with evidence snapshots.
    """
    def __init__(self, output_events_dir: Optional[str] = None):
        self.events_dir = output_events_dir or plate_config.ANPR_EVENTS_DIR
        os.makedirs(self.events_dir, exist_ok=True)
        # track_id -> List of candidate dictionaries
        self.track_candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        # track_id -> Primary Vehicle metadata (vehicle_type, max_veh_confidence, first_seen_frame, first_seen_time)
        self.track_metadata: Dict[int, Dict[str, Any]] = {}

    def update_track_metadata(self, tracked_vehicles: List[TrackedObject], frame_number: int, timestamp: float):
        """Updates persistent vehicle metadata for active tracks."""
        for veh in tracked_vehicles:
            if veh.class_name in ["PERSON", "RIDER"]:
                continue

            if veh.track_id not in self.track_metadata:
                self.track_metadata[veh.track_id] = {
                    "track_id": veh.track_id,
                    "type": veh.class_name,
                    "max_confidence": veh.confidence,
                    "first_frame": frame_number,
                    "first_timestamp": timestamp,
                    "latest_frame": frame_number,
                    "latest_timestamp": timestamp,
                    "best_vehicle_crop": None
                }
            else:
                meta = self.track_metadata[veh.track_id]
                meta["latest_frame"] = frame_number
                meta["latest_timestamp"] = timestamp
                if veh.confidence > meta["max_confidence"]:
                    meta["max_confidence"] = veh.confidence

    def add_observation(
        self,
        track_id: int,
        plate_det: PlateDetection,
        ocr_res: OCRResult,
        vehicle_crop: np.ndarray,
        plate_crop: np.ndarray,
        vehicle_confidence: float
    ):
        """Adds an OCR observation for a vehicle track ID across frames."""
        # Store vehicle crop in metadata if not already stored or higher confidence
        if track_id in self.track_metadata:
            meta = self.track_metadata[track_id]
            if meta["best_vehicle_crop"] is None or vehicle_confidence > meta["max_confidence"]:
                meta["best_vehicle_crop"] = vehicle_crop.copy() if vehicle_crop is not None else None

        self.track_candidates[track_id].append({
            "raw_ocr": ocr_res.raw_ocr,
            "normalized_text": ocr_res.normalized_text,
            "ocr_confidence": ocr_res.ocr_confidence,
            "plate_confidence": plate_det.confidence,
            "vehicle_confidence": vehicle_confidence,
            "is_valid_pattern": ocr_res.is_valid_pattern,
            "frame_number": plate_det.frame_number,
            "timestamp": plate_det.timestamp,
            "vehicle_crop": vehicle_crop,
            "plate_crop": plate_crop
        })

    def finalize_vehicle_records(self, video_filename: Optional[str] = None) -> List[VehicleRecord]:
        """
        Processes all aggregated candidates per track ID and builds final deduplicated VehicleRecord list.
        """
        records: List[VehicleRecord] = []

        for track_id, meta in self.track_metadata.items():
            candidates = self.track_candidates.get(track_id, [])

            if not candidates:
                # UNREADABLE Plate record
                record = VehicleRecord(
                    track_id=track_id,
                    vehicle_type=meta["type"],
                    vehicle_confidence=meta["max_confidence"],
                    registration_number="UNREADABLE",
                    raw_ocr="UNREADABLE",
                    ocr_confidence=0.0,
                    plate_detection_confidence=0.0,
                    timestamp=meta["first_timestamp"],
                    frame_number=meta["first_frame"]
                )
                # Save vehicle crop if available
                if meta["best_vehicle_crop"] is not None:
                    veh_snap_url, _ = self._save_evidence_snapshots(record.event_id, meta["best_vehicle_crop"], None)
                    record.media["vehicle_snapshot"] = veh_snap_url
                records.append(record)
                continue

            # Select best registration candidate using temporal voting & confidence weights
            best_candidate = self._select_best_candidate(candidates)

            reg_text = best_candidate["normalized_text"] if best_candidate["normalized_text"] else "UNREADABLE"
            raw_text = best_candidate["raw_ocr"] if best_candidate["raw_ocr"] else "UNREADABLE"

            record = VehicleRecord(
                track_id=track_id,
                vehicle_type=meta["type"],
                vehicle_confidence=meta["max_confidence"],
                registration_number=reg_text,
                raw_ocr=raw_text,
                ocr_confidence=best_candidate["ocr_confidence"],
                plate_detection_confidence=best_candidate["plate_confidence"],
                timestamp=best_candidate["timestamp"],
                frame_number=best_candidate["frame_number"]
            )

            # Save evidence snapshots for best observation frame
            veh_snap_url, plate_snap_url = self._save_evidence_snapshots(
                record.event_id,
                best_candidate["vehicle_crop"] if best_candidate["vehicle_crop"] is not None else meta["best_vehicle_crop"],
                best_candidate["plate_crop"]
            )
            record.media["vehicle_snapshot"] = veh_snap_url
            record.media["plate_snapshot"] = plate_snap_url
            record.media["video"] = video_filename

            records.append(record)

        return records

    def _select_best_candidate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Temporal Voting Algorithm:
        Groups candidates by normalized_text, ranks by valid Indian regex pattern match,
        frequency count across frames, and combined confidence score.
        """
        valid_candidates = [c for c in candidates if c["normalized_text"] != "UNREADABLE" and len(c["normalized_text"]) >= 5]
        if not valid_candidates:
            return candidates[0]

        # Group by normalized_text
        text_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in valid_candidates:
            text_groups[c["normalized_text"]].append(c)

        scored_groups = []
        for text, items in text_groups.items():
            freq = len(items)
            max_item = max(items, key=lambda x: x["ocr_confidence"] + x["plate_confidence"])
            pattern_bonus = 2.0 if max_item["is_valid_pattern"] else 0.5
            total_score = (freq * 1.5) + (max_item["ocr_confidence"] * 2.0) + (max_item["plate_confidence"] * 1.0) + pattern_bonus

            scored_groups.append((total_score, max_item))

        scored_groups.sort(key=lambda x: x[0], reverse=True)
        return scored_groups[0][1]

    def _save_evidence_snapshots(
        self,
        event_id: str,
        vehicle_crop: Optional[np.ndarray],
        plate_crop: Optional[np.ndarray]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Saves vehicle snapshot and plate crop image to processed events directory."""
        event_folder = os.path.join(self.events_dir, event_id)
        os.makedirs(event_folder, exist_ok=True)

        veh_url = None
        plate_url = None

        if vehicle_crop is not None and vehicle_crop.size > 0:
            veh_path = os.path.join(event_folder, "vehicle.jpg")
            cv2.imwrite(veh_path, vehicle_crop)
            veh_url = f"/processed/events/vehicle/{event_id}/vehicle.jpg"

        if plate_crop is not None and plate_crop.size > 0:
            plate_path = os.path.join(event_folder, "plate.jpg")
            cv2.imwrite(plate_path, plate_crop)
            plate_url = f"/processed/events/vehicle/{event_id}/plate.jpg"

        return veh_url, plate_url
