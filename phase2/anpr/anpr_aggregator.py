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
        from phase2.anpr.plate_detector import PlateDetector
        quality_score = PlateDetector.compute_crop_quality_score(plate_crop, plate_det.confidence)

        # Store vehicle crop in metadata if not already stored or higher crop quality score
        if track_id in self.track_metadata:
            meta = self.track_metadata[track_id]
            if meta["best_vehicle_crop"] is None or quality_score > meta.get("best_quality_score", 0.0):
                meta["best_vehicle_crop"] = vehicle_crop.copy() if vehicle_crop is not None else None
                meta["best_quality_score"] = quality_score

        self.track_candidates[track_id].append({
            "raw_ocr": ocr_res.raw_ocr,
            "normalized_text": ocr_res.normalized_text,
            "plate": ocr_res.plate if ocr_res.plate else (ocr_res.normalized_text if ocr_res.is_valid_pattern else None),
            "plate_status": ocr_res.plate_status if ocr_res.plate_status != "no_text" else ("validated" if ocr_res.is_valid_pattern else "no_text"),
            "ocr_confidence": ocr_res.ocr_confidence,
            "plate_confidence": ocr_res.plate_confidence if ocr_res.plate_confidence > 0 else ocr_res.ocr_confidence,
            "vehicle_confidence": vehicle_confidence,
            "quality_score": quality_score,
            "is_valid_pattern": ocr_res.is_valid_pattern,
            "frame_number": plate_det.frame_number,
            "timestamp": plate_det.timestamp,
            "vehicle_crop": vehicle_crop,
            "plate_crop": plate_crop,
            "preprocessed_crop": ocr_res.preprocessed_crop
        })

    def finalize_vehicle_records(self, video_filename: Optional[str] = None) -> List[VehicleRecord]:
        """
        Processes all aggregated candidates per track ID and builds final deduplicated VehicleRecord list.
        Applies C3 & C4: records outcome explicitly inside attrs (plate, plate_confidence, plate_status, ocr_raw).
        """
        records: List[VehicleRecord] = []

        for track_id, meta in self.track_metadata.items():
            candidates = self.track_candidates.get(track_id, [])

            if not candidates:
                record = VehicleRecord(
                    track_id=track_id,
                    vehicle_type=meta["type"],
                    vehicle_confidence=meta["max_confidence"],
                    plate=None,
                    registration_number="UNREADABLE",
                    ocr_raw="",
                    raw_ocr="UNREADABLE",
                    plate_confidence=0.0,
                    ocr_confidence=0.0,
                    plate_status="no_text",
                    plate_detection_confidence=0.0,
                    timestamp=meta["first_timestamp"],
                    frame_number=meta["first_frame"]
                )
                if meta["best_vehicle_crop"] is not None:
                    veh_snap_url, _, _, _ = self._save_evidence_snapshots(record.event_id, meta["best_vehicle_crop"], None, None)
                    record.media["vehicle_snapshot"] = veh_snap_url
                records.append(record)
                continue

            best_candidate = self._select_best_candidate(candidates)

            # Strict C3 rejection: if not validated pattern or confidence < 0.4, plate is None
            if best_candidate.get("plate_status") == "validated" and best_candidate.get("plate"):
                reg_plate = best_candidate["plate"]
                status = "validated"
            else:
                reg_plate = None
                status = best_candidate.get("plate_status", "unreadable")
                if status not in ["format_rejected", "low_confidence", "no_text"]:
                    status = "format_rejected" if best_candidate.get("raw_ocr") else "no_text"

            raw_text = best_candidate.get("raw_ocr", "")
            active_conf = best_candidate.get("plate_confidence", best_candidate.get("ocr_confidence", 0.0))

            record = VehicleRecord(
                track_id=track_id,
                vehicle_type=meta["type"],
                vehicle_confidence=meta["max_confidence"],
                plate=reg_plate,
                registration_number=reg_plate if reg_plate else "UNREADABLE",
                ocr_raw=raw_text,
                raw_ocr=raw_text if raw_text else "UNREADABLE",
                plate_confidence=active_conf,
                ocr_confidence=active_conf,
                plate_status=status,
                plate_detection_confidence=best_candidate.get("plate_confidence", 0.0),
                timestamp=best_candidate["timestamp"],
                frame_number=best_candidate["frame_number"]
            )

            # Save evidence snapshots (both raw and preprocessed plate crops)
            veh_snap_url, plate_snap_url, raw_snap_url, prep_snap_url = self._save_evidence_snapshots(
                record.event_id,
                best_candidate["vehicle_crop"] if best_candidate.get("vehicle_crop") is not None else meta["best_vehicle_crop"],
                best_candidate.get("plate_crop"),
                best_candidate.get("preprocessed_crop")
            )
            record.media["vehicle_snapshot"] = veh_snap_url
            record.media["plate_snapshot"] = plate_snap_url
            record.media["raw_plate_snapshot"] = raw_snap_url
            record.media["preproc_plate_snapshot"] = prep_snap_url
            record.media["video"] = video_filename

            records.append(record)

        return records

    def _select_best_candidate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Quality-Based Selection & Temporal Voting Algorithm:
        1. Filters out unvalidated or empty candidates.
        2. Groups candidates by validated plate string.
        3. If any validated candidates exist with confidence >= 0.40, select consensus winner.
        4. If no candidate validates, select representative unvalidated observation for evidence.
        """
        # Validated candidates: must have plate string, is_valid_pattern=True, confidence >= 0.40
        validated_candidates = [
            c for c in candidates
            if c.get("is_valid_pattern") and c.get("plate") and c.get("ocr_confidence", 0.0) >= 0.40
        ]

        if not validated_candidates:
            # No candidate met full validation criteria (C3)
            # Pick candidate with highest OCR confidence or crop quality score
            best = max(candidates, key=lambda x: (x.get("ocr_confidence", 0.0), x.get("quality_score", 0.0)))
            return best

        # Group validated candidates by plate string
        text_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in validated_candidates:
            text_groups[c["plate"]].append(c)

        scored_groups = []
        total_valid_obs = len(validated_candidates)

        for text, items in text_groups.items():
            freq = len(items)
            best_item = max(items, key=lambda x: (x.get("quality_score", 0.0), x.get("ocr_confidence", 0.0)))
            consensus_ratio = freq / float(total_valid_obs)

            total_score = (freq * 2.0) + (consensus_ratio * 3.0) + (best_item.get("quality_score", 0.0) * 2.0)

            scored_groups.append({
                "score": total_score,
                "plate": text,
                "freq": freq,
                "ratio": consensus_ratio,
                "item": best_item
            })

        scored_groups.sort(key=lambda x: x["score"], reverse=True)
        return scored_groups[0]["item"]

    def _save_evidence_snapshots(
        self,
        event_id: str,
        vehicle_crop: Optional[np.ndarray],
        plate_crop: Optional[np.ndarray],
        preprocessed_crop: Optional[np.ndarray] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Saves vehicle snapshot, raw plate crop, and preprocessed plate crop to events directory."""
        event_folder = os.path.join(self.events_dir, event_id)
        os.makedirs(event_folder, exist_ok=True)

        veh_url = None
        plate_url = None
        raw_url = None
        prep_url = None

        if vehicle_crop is not None and vehicle_crop.size > 0:
            veh_path = os.path.join(event_folder, "vehicle.jpg")
            cv2.imwrite(veh_path, vehicle_crop)
            veh_url = f"/processed/events/vehicle/{event_id}/vehicle.jpg"

        if plate_crop is not None and plate_crop.size > 0:
            plate_path = os.path.join(event_folder, "plate.jpg")
            raw_path = os.path.join(event_folder, "raw_plate.jpg")
            cv2.imwrite(plate_path, plate_crop)
            cv2.imwrite(raw_path, plate_crop)
            plate_url = f"/processed/events/vehicle/{event_id}/plate.jpg"
            raw_url = f"/processed/events/vehicle/{event_id}/raw_plate.jpg"

        if preprocessed_crop is not None and preprocessed_crop.size > 0:
            prep_path = os.path.join(event_folder, "preprocessed_plate.jpg")
            cv2.imwrite(prep_path, preprocessed_crop)
            prep_url = f"/processed/events/vehicle/{event_id}/preprocessed_plate.jpg"

        return veh_url, plate_url, raw_url, prep_url
