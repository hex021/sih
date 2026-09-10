import os
import time
import cv2
import logging
import numpy as np

from typing import Dict, Any, Optional


from phase1 import config
from phase1 import utils
from phase1 import roi
from phase1.tracker import Tracker
from phase1.counter import Counter
from phase1.density import classify_density
from phase1.visualizer import Visualizer
from phase1.main import filter_pedestrians

from phase2.anpr.plate_detector import PlateDetector
from phase2.anpr.ocr_engine import OCREngine
from phase2.anpr.anpr_aggregator import ANPRAggregator
from phase2.anpr import plate_config

logger = logging.getLogger("SIH_Server")


def run_anpr_pipeline(
    input_path: str,
    output_path: str,
    debug: bool = False,
    camera_mode: str = "stationary",
    top_n_vehicles: int = plate_config.DEFAULT_TOP_N_VEHICLES
) -> Dict[str, Any]:
    """
    Orchestrates Phase 2.2 Vehicle Identification & Number Plate Detection Pipeline.
    Reuses Phase 1 vehicle tracking outputs, performs spatial plate detection & OCR,
    aggregates multi-frame observations, generates evidence snapshots, and exports annotated MP4.
    """
    logger.info(f"Starting Phase 2.2 ANPR Pipeline on input: {input_path}")
    start_time = time.time()

    if not os.path.exists(input_path):
        logger.error(f"Input video file not found: {input_path}")
        return {"success": False, "error": f"Input video not found: {input_path}"}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file stream: {input_path}")
        return {"success": False, "error": "Failed to open video stream"}

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Configure output writer
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, source_fps, (frame_width, frame_height))

    # Determine Camera Mode ROI & Count line overrides
    if camera_mode == "moving":
        roi_relative = config.MOVING_CAMERA_ROI_RELATIVE
        line_y_rel = config.MOVING_CAMERA_COUNT_LINE_RELATIVE_Y
    else:
        roi_relative = config.ROI_RELATIVE
        line_y_rel = config.COUNT_LINE_RELATIVE_Y

    # Initialize frozen Phase 1 components
    tracker = Tracker()
    counter = Counter()
    visualizer = Visualizer()
    seen_objects = {cls: set() for cls in config.TARGET_CLASSES}

    # Initialize Phase 2.2 ANPR components
    plate_detector = PlateDetector()
    ocr_engine = OCREngine(use_easyocr=True)
    aggregator = ANPRAggregator()

    frame_count = 0
    device_name = getattr(tracker, "device", "CPU")

    # Mapping of current frame active plate detections for visual overlay
    current_frame_plates = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time_sec = frame_count / float(source_fps)

        # 1. Phase 1 YOLO Tracking (Frozen execution call)
        raw_tracked_objects = tracker.update(frame)
        tracked_objects = filter_pedestrians(raw_tracked_objects)

        # Update seen_objects sets for unique count tracking
        for obj in tracked_objects:
            if obj.track_id is not None:
                cls = obj.class_name
                for other_cls in seen_objects:
                    if other_cls != cls and obj.track_id in seen_objects[other_cls]:
                        seen_objects[other_cls].remove(obj.track_id)
                if cls in seen_objects:
                    seen_objects[cls].add(obj.track_id)

        # 2. Phase 1 Counter & Density Updates
        counter.update(tracked_objects, frame_width, frame_height, line_y_rel=line_y_rel)
        density_category, active_vehicle_count = classify_density(
            tracked_objects, frame_width, frame_height, roi_relative=roi_relative
        )

        # 3. Phase 2.2 ANPR Engine Updates (Only attempt OCR on top N vehicle boxes per frame)
        aggregator.update_track_metadata(tracked_objects, frame_count, current_time_sec)
        plate_detections = plate_detector.detect_plates_in_vehicles(
            frame, tracked_objects, frame_count, current_time_sec, top_n_vehicles=top_n_vehicles
        )

        current_frame_plates.clear()

        # 4. OCR Processing for Detected Plates
        for track_id, (plate_det, plate_crop) in plate_detections.items():
            veh_obj = next((v for v in tracked_objects if v.track_id == track_id), None)
            veh_conf = veh_obj.confidence if veh_obj else 0.80

            existing_obs = aggregator.track_candidates.get(track_id, [])
            valid_pattern_count = sum(1 for o in existing_obs if o.get("is_valid_pattern"))

            # Run OCR if: < 6 observations, or valid pattern not yet found, or frame_count % 3 == 0
            if len(existing_obs) < 6 or valid_pattern_count < 2 or (frame_count % 3 == 0):
                ocr_res = ocr_engine.extract_text(plate_crop, frame_number=frame_count, track_id=track_id)
            else:
                best_prev = max(existing_obs, key=lambda x: x.get("ocr_confidence", 0.0))
                from phase2.anpr.plate_models import OCRResult
                ocr_res = OCRResult(
                    raw_ocr=best_prev.get("raw_ocr", ""),
                    normalized_text=best_prev.get("normalized_text", "UNREADABLE"),
                    plate=best_prev.get("plate"),
                    ocr_confidence=best_prev.get("ocr_confidence", 0.0),
                    plate_confidence=best_prev.get("plate_confidence", 0.0),
                    plate_status=best_prev.get("plate_status", "no_text"),
                    is_valid_pattern=best_prev.get("is_valid_pattern", False),
                    raw_crop=plate_crop,
                    preprocessed_crop=best_prev.get("preprocessed_crop")
                )

            vehicle_crop = None
            if veh_obj:
                vx1, vy1, vx2, vy2 = [int(v) for v in veh_obj.bbox]
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(frame_width, vx2), min(frame_height, vy2)
                if (vx2 - vx1) > 10 and (vy2 - vy1) > 10:
                    vehicle_crop = frame[vy1:vy2, vx1:vx2].copy()

            aggregator.add_observation(
                track_id=track_id,
                plate_det=plate_det,
                ocr_res=ocr_res,
                vehicle_crop=vehicle_crop,
                plate_crop=plate_crop,
                vehicle_confidence=veh_conf
            )

            current_frame_plates[track_id] = {
                "bbox": plate_det.bbox,
                "text": ocr_res.plate if ocr_res.plate else ocr_res.plate_status.upper(),
                "ocr_conf": ocr_res.plate_confidence if ocr_res.plate_confidence > 0 else ocr_res.ocr_confidence,
                "status": ocr_res.plate_status
            }

        # 5. Render Visual Overlay Frame
        annotated_frame = visualizer.draw(
            frame=frame,
            tracked_objects=tracked_objects,
            counter=counter,
            density_class=density_category,
            active_roi_count=active_vehicle_count,
            line_y_rel=line_y_rel,
            roi_relative=roi_relative,
            camera_mode=camera_mode,
            unique_counts={c: len(seen_objects[c]) for c in config.TARGET_CLASSES}
        )

        # Phase 2.2 Plate Bounding Box & Registration Tag Overlay
        annotated_frame = draw_anpr_overlay(annotated_frame, current_frame_plates)

        out.write(annotated_frame)

    cap.release()
    out.release()

    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

    # 6. Finalize Vehicle Records & Evidence Snapshots
    vehicle_records = aggregator.finalize_vehicle_records(os.path.basename(output_path))
    records_dict = [r.to_dict() for r in vehicle_records]

    # Summary Statistics
    total_vehicles = len(vehicle_records)
    plates_validated = sum(1 for r in vehicle_records if r.plate_status == "validated")
    format_rejected = sum(1 for r in vehicle_records if r.plate_status == "format_rejected")
    low_confidence = sum(1 for r in vehicle_records if r.plate_status == "low_confidence")
    no_text = sum(1 for r in vehicle_records if r.plate_status == "no_text")
    unreadable_plates = total_vehicles - plates_validated

    logger.info(
        f"Phase 2.2 ANPR Pipeline Complete. Processed {frame_count} frames in {elapsed_time:.2f}s "
        f"({avg_fps:.2f} FPS). Vehicles identified: {total_vehicles} "
        f"(Validated: {plates_validated}, Format Rejected: {format_rejected}, Low Conf: {low_confidence}, No Text: {no_text})."
    )

    return {
        "success": True,
        "mode": "vehicle",
        "avg_fps": avg_fps,
        "elapsed_time": elapsed_time,
        "total_frames": frame_count,
        "source_fps": source_fps,
        "input_resolution": f"{frame_width}x{frame_height}",
        "processing_resolution": f"{frame_width}x{frame_height}",
        "inference_size": f"{config.INFERENCE_SIZE}x{config.INFERENCE_SIZE}",
        "device": str(device_name),
        "vehicle_records": records_dict,
        "summary": {
            "total_vehicles": total_vehicles,
            "plates_read": plates_validated,
            "plates_validated": plates_validated,
            "format_rejected": format_rejected,
            "low_confidence": low_confidence,
            "no_text": no_text,
            "unreadable_plates": unreadable_plates,
            "unique_tracks": total_vehicles
        }
    }


def draw_anpr_overlay(frame: np.ndarray, current_frame_plates: Dict[int, Dict[str, Any]]) -> np.ndarray:
    """Draws number plate bounding box rectangles and registration text tags on the frame."""
    for track_id, plate_info in current_frame_plates.items():
        px1, py1, px2, py2 = [int(v) for v in plate_info["bbox"]]
        text = plate_info.get("text", "UNREADABLE")
        status = plate_info.get("status", "")
        ocr_conf = int(plate_info.get("ocr_conf", 0.0) * 100)

        # Color: Cyan (255, 255, 0) if validated, Amber/Orange (0, 165, 255) if unreadable
        box_color = (255, 255, 0) if status == "validated" else (0, 165, 255)

        # Bounding box for number plate
        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

        # Label tag above plate box
        label = f"PLATE: {text} ({ocr_conf}%)"
        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        ty1 = max(0, py1 - lbl_h - 4)
        cv2.rectangle(frame, (px1, ty1), (px1 + lbl_w + 6, ty1 + lbl_h + 4), box_color, -1)
        cv2.putText(frame, label, (px1 + 3, ty1 + lbl_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

