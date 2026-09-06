import os
import time
import cv2
import torch
import numpy as np
from typing import List, Dict, Any

from phase1 import utils
from phase1 import config as phase1_config
from phase1.models import TrackedObject
from phase1.tracker import Tracker
from phase1.counter import Counter
from phase1.visualizer import Visualizer
from phase1 import density
from phase1.main import filter_pedestrians

from .pothole_config import POTHOLE_MODEL_PATH, POTHOLE_CONFIDENCE_THRESHOLD, POTHOLE_IOU_THRESHOLD, EVENTS_FOLDER
from .pothole_detector import PotholeDetector
from .pothole_models import PotholeEvent

def run_combined_pipeline(
    input_path: str,
    output_path: str,
    confidence: float = None,
    debug: bool = False,
    camera_mode: str = "stationary"
) -> dict:
    """
    Executes BOTH Traffic (Phase 1) and Road (Phase 2.1) AI models on a single frame decoding loop.
    Merges HUD dashboards and output annotations.
    """
    logger = utils.setup_logger("SIH_Combined_Pipeline")
    logger.info(f"Initializing SIH Combined Traffic + Road AI Pipeline in {camera_mode.upper()} mode...")
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Execution Device: {device}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
        
    # Track unique object IDs seen during the entire video
    seen_objects = {cls: set() for cls in phase1_config.TARGET_CLASSES}
    
    # Configure ROI based on camera mode
    if camera_mode == "moving":
        roi_relative_coords = (0.1, 0.4, 0.9, 0.85) # Exclude sky and dashboard
    else:
        roi_relative_coords = phase1_config.ROI_RELATIVE

    # Load Models
    tracker = Tracker()
    pothole_detector = PotholeDetector(model_path=POTHOLE_MODEL_PATH)
    logger.info("Both YOLO models successfully loaded.")
    
    counter = Counter()
    visualizer = Visualizer(debug=debug)
    
    # Read video properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open input video: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if video_fps <= 0:
        video_fps = 30.0
        
    # Scale calculation
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
        
        logger.info(f"High-res input {width}x{height} will be pre-resized to {target_width}x{target_height} using FFmpeg...")
        temp_resized_path = os.path.join(os.path.dirname(output_path), f"temp_resized_comb_{os.path.basename(input_path)}")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={target_width}:{target_height}",
            "-c:v", "libx264",
            "-crf", "23",
            temp_resized_path
        ]
        try:
            import subprocess
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            video_source_path = temp_resized_path
            scale = 1.0
            logger.info("FFmpeg pre-resizing completed successfully.")
        except Exception as e:
            logger.warning(f"FFmpeg pre-resizing failed: {e}. Falling back to on-the-fly resizing.")
            if os.path.exists(temp_resized_path):
                try: os.remove(temp_resized_path)
                except: pass
            temp_resized_path = None
            
    utils.ensure_dir(os.path.dirname(output_path))
    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video source: {video_source_path}")
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (target_width, target_height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Failed to initialize video writer for: {output_path}")
        
    frame_idx = 0
    start_time = time.time()
    all_pothole_events: List[PotholeEvent] = []
    
    # Portrait aspect ratio check for traffic count line
    relative_line_y = 0.75 if target_height > target_width else phase1_config.COUNT_LINE_RELATIVE_Y
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            timestamp = frame_idx / video_fps
            
            if scale != 1.0:
                frame_to_process = cv2.resize(frame, (target_width, target_height))
            else:
                frame_to_process = frame
                
            # --- 1. TRAFFIC AI PIPELINE ---
            conf_val = confidence if confidence is not None else phase1_config.CONFIDENCE_THRESHOLD
            tracked_objects = tracker.update(frame_to_process, conf=conf_val)
            tracked_objects = filter_pedestrians(tracked_objects)
            counter.update(tracked_objects, target_width, target_height, line_y_rel=relative_line_y)
            
            # Update unique object tracker with current frame detections
            for obj in tracked_objects:
                if obj.track_id is not None:
                    cls = obj.class_name
                    # If track ID was previously counted under a different class, remove it
                    for other_cls in seen_objects:
                        if other_cls != cls and obj.track_id in seen_objects[other_cls]:
                            seen_objects[other_cls].remove(obj.track_id)
                    # Add to the current class set if it is one of target classes
                    if cls in seen_objects:
                        seen_objects[cls].add(obj.track_id)

            density_class, active_roi_count = density.classify_density(tracked_objects, target_width, target_height, roi_relative=roi_relative_coords)
            
            # --- 2. ROAD AI PIPELINE ---
            pothole_dets, new_potholes = pothole_detector.detect(frame_to_process, frame_idx, timestamp)
            all_pothole_events.extend(new_potholes)
            
            # --- 3. COMBINED DRAW & VISUALIZATION ---
            # Run existing Phase 1 visualizer first
            avg_fps = frame_idx / (time.time() - start_time)
            unique_counts_mapped = {c: len(seen_objects[c]) for c in phase1_config.TARGET_CLASSES}
            annotated_frame = visualizer.draw(
                frame=frame_to_process,
                tracked_objects=tracked_objects,
                counter=counter,
                density_class=density_class,
                active_roi_count=active_roi_count,
                fps=avg_fps,
                frame_num=frame_idx,
                track_history=tracker.track_history,
                line_y_rel=relative_line_y,
                roi_relative=roi_relative_coords,
                camera_mode=camera_mode,
                unique_counts=unique_counts_mapped
            )
            
            # Overlay pothole boxes
            for det in pothole_dets:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
                label = f"POTHOLE {det.confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - lh - 6), (x1 + lw, y1), (0, 120, 255), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                            
            # Overlay top-right Road AI HUD dashboard
            px1, py1, px2, py2 = target_width - 320, 20, target_width - 20, 100
            # Blend backdrop
            sub_hud = annotated_frame[py1:py2, px1:px2]
            overlay = sub_hud.copy()
            cv2.rectangle(overlay, (0, 0), (px2 - px1, py2 - py1), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, sub_hud, 0.25, 0, sub_hud)
            annotated_frame[py1:py2, px1:px2] = sub_hud
            
            # Border & Text
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
        out.release()
        if temp_resized_path and os.path.exists(temp_resized_path):
            try: os.remove(temp_resized_path)
            except: pass
            
    elapsed = time.time() - start_time
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    
    # Calculate unique tracked object totals
    vehicle_classes = [c for c in phase1_config.TARGET_CLASSES if c != "PERSON"]
    total_vehicles_unique = sum(len(seen_objects[c]) for c in vehicle_classes)
    total_persons_unique = len(seen_objects.get("PERSON", set()))
    
    # Map internal unique class counts back to YOLO class names for the frontend table
    unique_objects = {}
    for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items():
        unique_objects[yolo_cls] = len(seen_objects.get(internal_cls, set()))
        
    # Generate line crossings details
    line_crossings_total = {}
    for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items():
        line_crossings_total[yolo_cls] = counter.counts["incoming"].get(internal_cls, 0) + counter.counts["outgoing"].get(internal_cls, 0)
        
    return {
        "success": True,
        "mode": "combined",
        "camera_mode": camera_mode,
        "events": all_pothole_events,
        "total_potholes": len(all_pothole_events),
        "traffic": {
            "unique_objects": unique_objects,
            "line_crossings": {
                "enabled": camera_mode == "stationary",
                "incoming": {yolo_cls: counter.counts["incoming"].get(internal_cls, 0) for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items()},
                "outgoing": {yolo_cls: counter.counts["outgoing"].get(internal_cls, 0) for yolo_cls, internal_cls in phase1_config.CLASS_MAP.items()},
                "total": line_crossings_total
            },
            "active_roi_vehicles": active_roi_count,
            "density": density_class
        },
        # Keep old/backward-compatible keys to satisfy Phase 1 integrations
        "statistics": unique_objects,  # Maps to unique counts for table display
        "line_crossing_statistics": line_crossings_total,
        "total_vehicles": total_vehicles_unique,
        "total_persons": total_persons_unique,
        "traffic_density": density_class,
        # Execution Metrics
        "total_frames": frame_idx,
        "elapsed_time": elapsed,
        "avg_fps": avg_fps,
        "input_resolution": f"{width}x{height}",
        "processing_resolution": f"{target_width}x{target_height}",
        "source_fps": video_fps,
        "device": device,
        "inference_size": 640
    }
