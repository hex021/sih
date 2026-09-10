import os
import time
import cv2
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from phase1 import utils
from phase1 import config as phase1_config
from phase1.models import TrackedObject
from phase1.tracker import Tracker
from phase1.counter import Counter
from phase1.visualizer import Visualizer
from phase1 import density
from phase1.main import filter_pedestrians
from phase1.camera_motion import detect_camera_motion

from phase2.road_ai.pothole_config import POTHOLE_MODEL_PATH, POTHOLE_CONFIDENCE_THRESHOLD, POTHOLE_IOU_THRESHOLD
from phase2.road_ai.pothole_detector import PotholeDetector
from phase2.road_ai.pothole_models import PotholeEvent

from phase2.anpr.plate_detector import PlateDetector
from phase2.anpr.ocr_engine import OCREngine
from phase2.anpr.anpr_aggregator import ANPRAggregator
from phase2.anpr.anpr_pipeline import draw_anpr_overlay

logger = logging.getLogger("SIH_Server")

def run_unified_pipeline(
    input_path: str,
    output_path: str,
    confidence: Optional[float] = None,
    debug: bool = False,
    camera_mode: Optional[str] = None,
    top_n_vehicles: int = 3
) -> Dict[str, Any]:

    """
    Executes ONE UNIFIED pipeline combining Traffic AI (Vehicle Detection & Tracking),
    Road AI (Pothole Detection), and ANPR AI (License Plate Detection & OCR) on a
    SINGLE frame decoding loop.

    :param input_path: Path to input video file.
    :param output_path: Path to write output annotated MP4 video.
    :param confidence: YOLO vehicle detection confidence threshold.
    :param debug: If True, enables debug visual rendering.
    :param camera_mode: "stationary", "moving", "auto", or None. If None or "auto",
                        automatically detects camera motion using optical flow.
    :return: Structured telemetry dictionary containing traffic, ANPR, pothole, and metric fields.
    """
    logger.info(f"Initializing SIH Core Unified AI Pipeline on input: {input_path}")
    start_time = time.time()

    if not os.path.exists(input_path):
        logger.error(f"Input video file not found: {input_path}")
        return {"success": False, "error": f"Input video not found: {input_path}"}

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Execution Device: {device}")

    # 1. Resolve Camera Mode (Auto Motion Determination if None / "auto")
    if camera_mode is None or camera_mode.lower() == "auto":
        logger.info("Camera mode unspecified or set to 'auto'. Running auto-camera motion detection...")
        resolved_camera_mode = detect_camera_motion(input_path, max_frames=60)
        logger.info(f"Auto-detected camera mode: '{resolved_camera_mode}'")
    else:
        resolved_camera_mode = camera_mode.lower()
        logger.info(f"Using explicitly specified camera mode: '{resolved_camera_mode}'")

    # Configure ROI and line crossing based on resolved camera mode
    if resolved_camera_mode == "moving":
        roi_relative_coords = (0.1, 0.4, 0.9, 0.85)
        relative_line_y = 0.5
    else:
        roi_relative_coords = phase1_config.ROI_RELATIVE
        relative_line_y = phase1_config.COUNT_LINE_RELATIVE_Y

    # 2. Read input video properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file stream: {input_path}")
        return {"success": False, "error": "Failed to open video stream"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    if fps_read is not None and not np.isnan(fps_read) and 5 < fps_read < 120:
        video_fps = float(fps_read)
    else:
        video_fps = 25.0
        logger.warning(f"Invalid or missing source FPS value ({fps_read}). Falling back to 25.0 FPS.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    # 3. High-resolution scaling setup
    max_dim = max(width, height)
    scale = 1.0
    target_width = width
    target_height = height
    temp_resized_path = None
    video_source_path = input_path

    if max_dim > phase1_config.MAX_PROCESSING_DIMENSION:
        scale = phase1_config.MAX_PROCESSING_DIMENSION / max_dim
        target_width = int(width * scale)
        target_height = int(height * scale)
        if target_width % 2 != 0: target_width += 1
        if target_height % 2 != 0: target_height += 1

        logger.info(f"High-res input {width}x{height} pre-resizing to {target_width}x{target_height} using FFmpeg...")
        temp_resized_path = os.path.join(os.path.dirname(output_path), f"temp_resized_uni_{os.path.basename(input_path)}")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={target_width}:{target_height}",
            "-c:v", "libx264",
            "-crf", "23",
            "-vsync", "cfr",
            "-r", str(video_fps),
            temp_resized_path
        ]
        try:
            import subprocess
            subprocess.run(ffmpeg_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            video_source_path = temp_resized_path
            scale = 1.0
            logger.info("FFmpeg pre-resizing completed successfully.")
        except Exception as e:
            logger.warning(f"FFmpeg pre-resizing failed: {e}. Falling back to on-the-fly resizing.")
            if os.path.exists(temp_resized_path):
                try: os.remove(temp_resized_path)
                except: pass
            temp_resized_path = None

    # Adjust relative line y if aspect ratio is portrait
    if target_height > target_width and resolved_camera_mode != "moving":
        relative_line_y = 0.75

    # 4. Initialize Models (Loaded ONCE before frame decoding loop)
    tracker = Tracker()
    counter = Counter()
    visualizer = Visualizer(debug=debug)

    from phase2.telemetry import GPSTrack
    gps = GPSTrack()
    if gps.available:
        logger.info(f"GPS track loaded: {gps.total_frames()} frames (simulated)")

    pothole_detector = PotholeDetector(model_path=POTHOLE_MODEL_PATH, gps_track=gps)
    plate_detector = PlateDetector()
    ocr_engine = OCREngine(use_easyocr=True)
    aggregator = ANPRAggregator()

    logger.info("All AI modules (YOLO Vehicle Tracker, Pothole Detector, Plate Detector, OCR Engine) initialized.")

    # 5. Output Video Writer Setup
    output_dir = os.path.dirname(output_path)
    if output_dir:
        utils.ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video source stream: {video_source_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (target_width, target_height))

    # Trackers for unique objects across frames
    seen_objects = {cls: set() for cls in phase1_config.TARGET_CLASSES}
    all_pothole_events: List[PotholeEvent] = []
    current_frame_plates: Dict[int, Dict[str, Any]] = {}

    # State caches for frame decimation interpolation
    cached_tracked_objects: List[TrackedObject] = []
    cached_density_class: str = "LOW"
    cached_active_roi_count: int = 0
    cached_pothole_dets: List[Any] = []
    cached_current_frame_plates: Dict[int, Dict[str, Any]] = {}

    frame_idx = 0
    frame_step = 1
    if device == "CPU" and total_frames > 50:
        frame_step = 2 if total_frames < 150 else 3
        logger.info(f"Adaptive CPU optimization active: processing keyframes with step={frame_step} (total_frames={total_frames})")

    # 6. Single Video Frame Decoding Loop
    try:
        if not out.isOpened():
            raise IOError(f"Failed to initialize VideoWriter for output: {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time_sec = frame_idx / float(video_fps)

            if scale != 1.0:
                frame_to_process = cv2.resize(frame, (target_width, target_height))
            else:
                frame_to_process = frame

            is_keyframe = (frame_step <= 1) or ((frame_idx - 1) % frame_step == 0)

            if is_keyframe:
                # --- A. Phase 1 Traffic AI ---
                conf_val = confidence if confidence is not None else phase1_config.CONFIDENCE_THRESHOLD
                raw_tracked = tracker.update(frame_to_process, conf=conf_val)
                tracked_objects = filter_pedestrians(raw_tracked)
                counter.update(tracked_objects, target_width, target_height, line_y_rel=relative_line_y)

                # Update unique object class sets
                for obj in tracked_objects:
                    if obj.track_id is not None:
                        cls = obj.class_name
                        for other_cls in seen_objects:
                            if other_cls != cls and obj.track_id in seen_objects[other_cls]:
                                seen_objects[other_cls].remove(obj.track_id)
                        if cls in seen_objects:
                            seen_objects[cls].add(obj.track_id)

                density_class, active_roi_count = density.classify_density(
                    tracked_objects, target_width, target_height, roi_relative=roi_relative_coords
                )

                # --- B. Phase 2.1 Road AI (Potholes) ---
                pothole_dets, new_potholes = pothole_detector.detect(frame_to_process, frame_idx, current_time_sec)
                all_pothole_events.extend(new_potholes)

                # --- C. Phase 2.2 ANPR & License Plate Recognition ---
                aggregator.update_track_metadata(tracked_objects, frame_idx, current_time_sec)
                plate_detections = plate_detector.detect_plates_in_vehicles(
                    frame_to_process, tracked_objects, frame_idx, current_time_sec, top_n_vehicles=top_n_vehicles
                )

                current_frame_plates.clear()

                for track_id, (plate_det, plate_crop) in plate_detections.items():
                    veh_obj = next((v for v in tracked_objects if v.track_id == track_id), None)
                    veh_conf = veh_obj.confidence if veh_obj else 0.80

                    existing_obs = aggregator.track_candidates.get(track_id, [])
                    valid_pattern_count = sum(1 for o in existing_obs if o.get("is_valid_pattern"))

                    # Run OCR if: < 6 observations, or valid pattern not yet found, or frame_idx % 3 == 0
                    if len(existing_obs) < 6 or valid_pattern_count < 2 or (frame_idx % 3 == 0):
                        ocr_res = ocr_engine.extract_text(plate_crop, frame_number=frame_idx, track_id=track_id)
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

                    # Crop vehicle bounding box for evidence snapshot saving
                    vehicle_crop = None
                    if veh_obj:
                        vx1, vy1, vx2, vy2 = [int(v) for v in veh_obj.bbox]
                        vx1, vy1 = max(0, vx1), max(0, vy1)
                        vx2, vy2 = min(target_width, vx2), min(target_height, vy2)
                        if (vx2 - vx1) > 10 and (vy2 - vy1) > 10:
                            vehicle_crop = frame_to_process[vy1:vy2, vx1:vx2].copy()

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

                # Update caches
                cached_tracked_objects = tracked_objects
                cached_density_class = density_class
                cached_active_roi_count = active_roi_count

                cached_pothole_dets = pothole_dets
                cached_current_frame_plates = current_frame_plates.copy()

            # --- D. Unified Visualization & Overlay ---
            avg_fps = frame_idx / (time.time() - start_time)
            unique_counts_mapped = {c: len(seen_objects[c]) for c in phase1_config.TARGET_CLASSES}

            annotated_frame = visualizer.draw(
                frame=frame_to_process,
                tracked_objects=cached_tracked_objects,
                counter=counter,
                density_class=cached_density_class,
                active_roi_count=cached_active_roi_count,
                fps=avg_fps,
                frame_num=frame_idx,
                track_history=tracker.track_history,
                line_y_rel=relative_line_y,
                roi_relative=roi_relative_coords,
                camera_mode=resolved_camera_mode,
                unique_counts=unique_counts_mapped
            )

            # Overlay ANPR Cyan Plate bounding boxes & Registration tags
            annotated_frame = draw_anpr_overlay(annotated_frame, cached_current_frame_plates)

            # Overlay Pothole Orange bounding boxes
            for det in cached_pothole_dets:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
                label = f"POTHOLE {det.confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - lh - 6), (x1 + lw, y1), (0, 120, 255), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Overlay Faint Pothole Road ROI Outline (thin grey line)
            prx1 = int(target_width * pothole_detector.road_roi[0])
            pry1 = int(target_height * pothole_detector.road_roi[1])
            prx2 = int(target_width * pothole_detector.road_roi[2])
            pry2 = int(target_height * pothole_detector.road_roi[3])
            cv2.rectangle(annotated_frame, (prx1, pry1), (prx2, pry2), (128, 128, 128), 1)

            # Overlay Top-Right Road Quality HUD Panel safely clamped to frame bounds
            px1 = max(0, target_width - 320)
            px2 = min(target_width, px1 + 300)
            py1 = max(0, 20)
            py2 = min(target_height, 100)

            if (px2 - px1) > 50 and (py2 - py1) > 20:
                sub_hud = annotated_frame[py1:py2, px1:px2]
                overlay = sub_hud.copy()
                cv2.rectangle(overlay, (0, 0), (px2 - px1, py2 - py1), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.75, sub_hud, 0.25, 0, sub_hud)
                annotated_frame[py1:py2, px1:px2] = sub_hud

                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 120, 255), 1)
                cv2.putText(annotated_frame, "ROAD QUALITY MONITOR", (px1 + 15, py1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Potholes Detected: {len(all_pothole_events)}", (px1 + 15, py1 + 47),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Simulated GPS: NULL (Waiting)", (px1 + 15, py1 + 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

            out.write(annotated_frame)

            if frame_idx % 100 == 0 or frame_idx == total_frames:
                logger.info(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

    finally:
        cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        if temp_resized_path and os.path.exists(temp_resized_path):
            try: os.remove(temp_resized_path)
            except: pass

    elapsed_time = time.time() - start_time
    avg_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0.0

    # 7. Finalize ANPR Vehicle Records & Summary
    vehicle_records = aggregator.finalize_vehicle_records(os.path.basename(output_path))
    records_dict = [r.to_dict() for r in vehicle_records]

    vehicle_classes = [c for c in phase1_config.TARGET_CLASSES if c != "PERSON"]
    total_vehicles_unique = sum(len(seen_objects[c]) for c in vehicle_classes)
    total_persons_unique = len(seen_objects.get("PERSON", set()))

    unique_objects = {}
    for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items():
        unique_objects[yolo_cls] = len(seen_objects.get(internal_cls, set()))

    line_crossings_total = {}
    for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items():
        line_crossings_total[yolo_cls] = counter.counts["incoming"].get(internal_cls, 0) + counter.counts["outgoing"].get(internal_cls, 0)

    plates_validated = sum(1 for r in vehicle_records if r.plate_status == "validated")
    format_rejected = sum(1 for r in vehicle_records if r.plate_status == "format_rejected")
    low_confidence = sum(1 for r in vehicle_records if r.plate_status == "low_confidence")
    no_text = sum(1 for r in vehicle_records if r.plate_status == "no_text")
    unreadable_plates = len(vehicle_records) - plates_validated

    logger.info(
        f"Unified AI Pipeline Complete. Processed {frame_idx} frames in {elapsed_time:.2f}s "
        f"({avg_fps:.2f} FPS). Camera mode: {resolved_camera_mode.upper()} | Vehicles: {total_vehicles_unique} "
        f"| ANPR Records: {len(vehicle_records)} (Validated: {plates_validated}, Format Rejected: {format_rejected}, Low Conf: {low_confidence}, No Text: {no_text}) | Potholes: {len(all_pothole_events)}"
    )

    return {
        "success": True,
        "mode": "unified",
        "camera_mode": resolved_camera_mode,
        "traffic": {
            "unique_objects": unique_objects,
            "line_crossings": {
                "enabled": resolved_camera_mode == "stationary",
                "incoming": {yolo_cls: counter.counts["incoming"].get(internal_cls, 0) for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items()},
                "outgoing": {yolo_cls: counter.counts["outgoing"].get(internal_cls, 0) for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items()},
                "total": line_crossings_total
            },
            "active_roi_vehicles": active_roi_count,
            "density": density_class
        },
        "statistics": unique_objects,
        "line_crossing_statistics": line_crossings_total,
        "total_vehicles": total_vehicles_unique,
        "total_persons": total_persons_unique,
        "traffic_density": density_class,
        "vehicle_records": records_dict,
        "events": [event.to_dict() for event in all_pothole_events],
        "total_potholes": len(all_pothole_events),
        "summary": {
            "total_vehicles": len(vehicle_records),
            "plates_read": plates_validated,
            "plates_validated": plates_validated,
            "format_rejected": format_rejected,
            "low_confidence": low_confidence,
            "no_text": no_text,
            "unreadable_plates": unreadable_plates,
            "potholes": len(all_pothole_events)
        },

        "elapsed_time": round(elapsed_time, 2),
        "avg_fps": round(avg_fps, 2),
        "total_frames": frame_idx,
        "source_fps": video_fps,
        "input_resolution": f"{width}x{height}",
        "processing_resolution": f"{target_width}x{target_height}",
        "device": device,
        "inference_size": 640
    }
