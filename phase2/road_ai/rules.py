import os
import uuid
import logging
from collections import deque
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger("SIH_Rules")

class EventRule:
    """Base class for video and telemetry event detection rules."""
    def evaluate(self, tracked_objects: List[Any], frame: np.ndarray, frame_idx: int, timestamp: float, gps=None) -> List[Dict[str, Any]]:
        raise NotImplementedError

class RuleRegistry:
    def __init__(self):
        self._rules: List[EventRule] = []

    def register(self, rule: EventRule) -> EventRule:
        self._rules.append(rule)
        return rule

    def get_rules(self) -> List[EventRule]:
        return list(self._rules)

rule_registry = RuleRegistry()

class RashDrivingRule(EventRule):
    """
    Geometric Rash Driving Rule detecting lateral swerving or rapid vehicle approach.
    Consumes tracked vehicles and triggers an INCIDENT event with ANPR plate extraction.
    """
    def __init__(
        self,
        lateral_threshold: float = 0.15,
        approach_threshold: float = 0.12,
        min_box_height: float = 60.0,
        cooldown_seconds: float = 5.0,
        crops_dir: str = "data/crops/incidents",
        plate_detector=None,
        ocr_engine=None
    ):
        self.lateral_threshold = lateral_threshold
        self.approach_threshold = approach_threshold
        self.min_box_height = min_box_height
        self.cooldown_seconds = cooldown_seconds
        self.crops_dir = crops_dir
        self.plate_detector = plate_detector
        self.ocr_engine = ocr_engine

        self.track_history: Dict[int, deque] = {}
        self.consecutive_lateral: Dict[int, int] = {}
        self.consecutive_approach: Dict[int, int] = {}
        self.cooldowns: Dict[int, float] = {}

        self.total_evaluated = 0
        self.total_fired = 0
        self.counts_by_reason = {"lateral_swerve": 0, "rapid_approach": 0}

    def evaluate(
        self,
        tracked_objects: List[Any],
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        gps=None
    ) -> List[Dict[str, Any]]:
        fired_events: List[Dict[str, Any]] = []
        if frame is None or not tracked_objects:
            return fired_events

        frame_h, frame_w = frame.shape[:2]
        eval_count = 0

        for veh in tracked_objects:
            track_id = getattr(veh, "track_id", None)
            if track_id is None or getattr(veh, "class_name", "") in ["PERSON", "RIDER"]:
                continue

            x1, y1, x2, y2 = [float(v) for v in veh.bbox]
            cx, h = (x1 + x2) / 2.0, y2 - y1
            if h < self.min_box_height:
                continue

            eval_count += 1
            self.total_evaluated += 1

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=10)
                self.consecutive_lateral[track_id] = 0
                self.consecutive_approach[track_id] = 0

            hist = self.track_history[track_id]
            lat_vel, app_rate = 0.0, 0.0

            if len(hist) > 0:
                prev_cx, prev_h = hist[-1]
                lat_vel = abs(cx - prev_cx) / max(h, 1e-5)
                app_rate = (h - prev_h) / max(h, 1e-5)

                self.consecutive_lateral[track_id] = (self.consecutive_lateral[track_id] + 1) if lat_vel > self.lateral_threshold else 0
                self.consecutive_approach[track_id] = (self.consecutive_approach[track_id] + 1) if app_rate > self.approach_threshold else 0

            hist.append((cx, h))

            fire_reason, measured_val = None, 0.0
            if self.consecutive_lateral[track_id] >= 3:
                fire_reason, measured_val = "lateral_swerve", lat_vel
            elif self.consecutive_approach[track_id] >= 3:
                fire_reason, measured_val = "rapid_approach", app_rate

            last_fired = self.cooldowns.get(track_id, -999.0)
            if fire_reason and (timestamp - last_fired >= self.cooldown_seconds):
                self.cooldowns[track_id] = timestamp
                self.consecutive_lateral[track_id] = 0
                self.consecutive_approach[track_id] = 0
                self.total_fired += 1
                self.counts_by_reason[fire_reason] += 1

                # 1. Save evidence crop of vehicle's bounding box
                os.makedirs(self.crops_dir, exist_ok=True)
                vx1, vy1 = max(0, int(x1)), max(0, int(y1))
                vx2, vy2 = min(frame_w, int(x2)), min(frame_h, int(y2))
                crop = frame[vy1:vy2, vx1:vx2]
                filename = f"incident_{track_id}_{frame_idx}_{int(timestamp * 1000)}.jpg"
                crop_path = os.path.join(self.crops_dir, filename)
                if crop.size > 0:
                    cv2.imwrite(crop_path, crop)
                snap_url = f"/data/crops/incidents/{filename}"

                # 2. Invoke plate detector & OCR on vehicle bounding box only
                plate_str, plate_conf, plate_status, ocr_raw = None, 0.0, "no_plate_detected", ""
                if self.plate_detector is not None:
                    p_dets = self.plate_detector.detect_plates_in_vehicles(frame, [veh], frame_number=frame_idx, timestamp=timestamp, top_n_vehicles=1)
                    if track_id in p_dets and self.ocr_engine is not None:
                        _, p_crop = p_dets[track_id]
                        ocr_res = self.ocr_engine.extract_text(p_crop, frame_number=frame_idx, track_id=track_id)
                        plate_str = ocr_res.plate
                        plate_conf = round(ocr_res.plate_confidence if ocr_res.plate_confidence > 0 else ocr_res.ocr_confidence, 2)
                        plate_status = ocr_res.plate_status
                        ocr_raw = ocr_res.raw_ocr

                # 3. Location from GPS telemetry
                loc_dict = None
                if gps and hasattr(gps, "get_frame"):
                    g_data = gps.get_frame(frame_idx)
                    if g_data:
                        loc_dict = {"latitude": g_data["latitude"], "longitude": g_data["longitude"], "speed_kmph": g_data.get("speed_kmph", 0.0), "heading_deg": g_data.get("heading_deg", 0.0)}

                event = {
                    "event_id": f"INC-{uuid.uuid4().hex[:8].upper()}",
                    "event_type": "INCIDENT",
                    "timestamp": round(timestamp, 2),
                    "confidence": round(float(getattr(veh, "confidence", 0.90) or 0.90), 2),
                    "location": loc_dict,
                    "source": {"camera": "dashcam", "track_id": track_id},
                    "media": {"frame": frame_idx, "snapshot": snap_url},
                    "attrs": {
                        "track_id": track_id,
                        "reason": fire_reason,
                        "measured_value": round(float(measured_val), 3),
                        "plate": plate_str,
                        "plate_confidence": plate_conf,
                        "plate_status": plate_status,
                        "ocr_raw": ocr_raw,
                        "vehicle_type": getattr(veh, "class_name", "VEHICLE"),
                        "vehicle_confidence": round(float(getattr(veh, "confidence", 0.0) or 0.0), 2),
                        "crop_path": crop_path
                    }
                }
                fired_events.append(event)
                logger.info(f"[RashDrivingRule] Track #{track_id} fired '{fire_reason}' (val={measured_val:.3f}) at t={timestamp:.2f}s, Plate: {plate_str or plate_status}")

        if fired_events:
            logger.info(f"[RashDrivingRule Frame {frame_idx}] Evaluated {eval_count} tracks, fired {len(fired_events)} incidents.")
        return fired_events

# Register default RashDrivingRule into the rule registry
rule_registry.register(RashDrivingRule())
