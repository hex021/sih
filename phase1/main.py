import os
import sys
import time
import argparse
import cv2
import torch
from typing import List
from phase1 import config
from phase1 import utils
from phase1 import roi
from phase1 import density
from phase1.models import TrackedObject
from phase1.tracker import Tracker
from phase1.counter import Counter
from phase1.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description="SIH Phase 1 AI Traffic Detection Pipeline")
    parser.add_argument("--input", type=str, default=config.DEFAULT_INPUT_VIDEO,
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default=config.DEFAULT_OUTPUT_VIDEO,
                        help="Path to save annotated output video")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH,
                        help="Path to YOLO model file (.pt)")
    parser.add_argument("--confidence", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for detections")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug visualizations and metrics overlay")
    parser.add_argument("--display", action="store_true",
                        help="Display the video output in a window (press 'q' to quit)")
    return parser.parse_args()

def filter_pedestrians(tracked_objects: List[TrackedObject]) -> List[TrackedObject]:
    """
    Suppresses persons who are motorcycle riders, car occupants, etc.
    Updates their class name to 'RIDER' or 'OCCUPANT' to prevent them
    from being counted as independent road pedestrians.
    """
    persons = [obj for obj in tracked_objects if obj.class_name == "PERSON"]
    vehicles = [obj for obj in tracked_objects if obj.class_name in ("CAR", "MOTORCYCLE", "BUS", "TRUCK", "BICYCLE")]
    
    for person in persons:
        px1, py1, px2, py2 = person.bbox
        p_w = px2 - px1
        p_h = py2 - py1
        p_cx = person.center_x
        p_cy = person.center_y
        
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle.bbox
            v_w = vx2 - vx1
            v_h = vy2 - vy1
            
            # Heuristic 1: Person center lies inside vehicle bounding box
            if vx1 <= p_cx <= vx2 and vy1 <= p_cy <= vy2:
                person.class_name = "RIDER" if vehicle.class_name in ("MOTORCYCLE", "BICYCLE") else "OCCUPANT"
                break
                
            # Heuristic 2: Large overlap ratio (Intersection over Person Area)
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersect = (ix2 - ix1) * (iy2 - iy1)
                overlap = intersect / (p_w * p_h)
                if overlap > 0.35:
                    person.class_name = "RIDER" if vehicle.class_name in ("MOTORCYCLE", "BICYCLE") else "OCCUPANT"
                    break
                    
            # Heuristic 3: Bottom-center of the person is on/near the vehicle box
            # Checks if person is sitting on motorcycle/bicycle (within 15px of top edge)
            if vx1 <= p_cx <= vx2 and (vy1 - 15) <= py2 <= vy2:
                person.class_name = "RIDER" if vehicle.class_name in ("MOTORCYCLE", "BICYCLE") else "OCCUPANT"
                break
                
    return tracked_objects

def run_pipeline(
    input_path: str,
    output_path: str,
    model_path: str = None,
    confidence: float = None,
    debug: bool = False,
    display: bool = False,
    camera_mode: str = "stationary"
) -> dict:
    """
    Executes the Phase 1 AI Traffic Detection Pipeline on a video.
    Returns: A dictionary of final telemetry counts and execution metrics.
    """
    logger = utils.setup_logger()
    
    logger.info(f"Initializing SIH Phase 1 AI Traffic Detection Pipeline in {camera_mode.upper()} mode...")
    
    # Track unique object IDs seen during the entire video
    seen_objects = {cls: set() for cls in config.TARGET_CLASSES}
    
    # Configure ROI based on camera mode
    if camera_mode == "moving":
        roi_relative_coords = (0.1, 0.4, 0.9, 0.85) # Exclude sky and dashboard
    else:
        roi_relative_coords = config.ROI_RELATIVE
    
    # 1. Device detection
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Execution Device: {device}")
    
    # 2. Verify input video
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
        
    logger.info(f"Input Video: {input_path}")
    
    # 3. Initialize core modules
    logger.info("Loading YOLO model...")
    actual_model_path = model_path if model_path else config.MODEL_PATH
    tracker = Tracker(model_path=actual_model_path)
    logger.info(f"Model successfully loaded from {actual_model_path}")
        
    counter = Counter()
    visualizer = Visualizer(debug=debug)
    
    # 4. Open video capture briefly to read video properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open input video: {input_path}")
        
    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Adjust for invalid or missing FPS metadata
    if video_fps <= 0:
        video_fps = 30.0
        
    # Resolution scaling calculation
    max_dim = max(width, height)
    scale = 1.0
    target_width = width
    target_height = height
    
    temp_resized_path = None
    video_source_path = input_path
    
    if max_dim > config.MAX_PROCESSING_DIMENSION:
        scale = config.MAX_PROCESSING_DIMENSION / max_dim
        target_width = int(width * scale)
        target_height = int(height * scale)
        # Ensure even dimensions (h264/mp4 encoder requirement)
        if target_width % 2 != 0:
            target_width += 1
        if target_height % 2 != 0:
            target_height += 1
            
        # Pre-resize using FFmpeg to avoid heavy CPU decoding inside Python loop
        logger.info(f"High-res input {width}x{height} will be pre-resized to {target_width}x{target_height} using FFmpeg...")
        temp_resized_path = os.path.join(os.path.dirname(output_path), f"temp_resized_{os.path.basename(input_path)}")
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
            subprocess.run(ffmpeg_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            video_source_path = temp_resized_path
            # Since video is pre-resized, scaling factor inside frame loop is 1.0
            scale = 1.0
            logger.info("FFmpeg pre-resizing completed successfully.")
        except Exception as pre_err:
            logger.warning(f"FFmpeg pre-resizing failed: {pre_err}. Falling back to on-the-fly resizing.")
            if os.path.exists(temp_resized_path):
                try: os.remove(temp_resized_path)
                except: pass
            temp_resized_path = None
    else:
        logger.info(f"Input dimensions {width}x{height} are within limit. Processing at native resolution.")
        
    # Dynamic relative count line based on aspect ratio:
    # Use Y = 0.75 for portrait layouts, else use default 0.65
    relative_line_y = 0.75 if target_height > target_width else config.COUNT_LINE_RELATIVE_Y
    logger.info(f"Using relative count line Y: {relative_line_y} (absolute pixel height: {relative_line_y * target_height:.1f})")
    
    logger.info(f"Video Properties — Resolution: {width}x{height} (Processing: {target_width}x{target_height}) | FPS: {video_fps:.2f} | Total Frames: {total_frames}")
    
    # 5. Initialize video writer
    utils.ensure_dir(os.path.dirname(output_path))
    
    # Open target video source for execution
    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video source: {video_source_path}")
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (target_width, target_height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Failed to initialize video writer for: {output_path}")
        
    logger.info(f"Output Video will be saved to: {output_path}")
    logger.info("Processing frames...")
    
    frame_idx = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Downscale frame if scaling is active (scale is 1.0 if pre-resized)
            if scale != 1.0:
                frame_to_process = cv2.resize(frame, (target_width, target_height))
            else:
                frame_to_process = frame
            
            # Execute tracking (Runs single-stage YOLO tracking per frame)
            conf_val = confidence if confidence is not None else config.CONFIDENCE_THRESHOLD
            tracked_objects = tracker.update(frame_to_process, conf=conf_val)
            
            # Run rider/occupant suppression filter (Modifies tracked_objects in place)
            tracked_objects = filter_pedestrians(tracked_objects)
            
            # Execute counting
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

            # Execute density estimation
            density_class, active_roi_count = density.classify_density(tracked_objects, target_width, target_height, roi_relative=roi_relative_coords)
            
            # Performance profiling
            elapsed_so_far = time.time() - start_time
            avg_fps = frame_idx / elapsed_so_far if elapsed_so_far > 0 else 0.0
            
            # Render overlays on the processed frame
            unique_counts_mapped = {c: len(seen_objects[c]) for c in config.TARGET_CLASSES}
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
            
            # Write annotated frame to output video
            out.write(annotated_frame)
            
            # Optional real-time GUI window
            if display:
                cv2.imshow("SIH Phase 1 AI Traffic Detection", annotated_frame)
                # Press 'q' to quit processing early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.warning("Processing interrupted by user.")
                    break
                    
            # Print periodic progress logs
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                logger.info(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | Current Avg FPS: {avg_fps:.2f}")
                
    except Exception as e:
        logger.error(f"Error occurred during frame loop: {e}")
        raise e
    finally:
        # Release all video captures, writers and windows
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
        # Clean up temporary pre-resized video file if created
        if temp_resized_path and os.path.exists(temp_resized_path):
            try:
                os.remove(temp_resized_path)
                logger.info("Temporary pre-resized video file cleaned up.")
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete temp resized file {temp_resized_path}: {cleanup_err}")
        
    end_time = time.time()
    total_elapsed = end_time - start_time
    final_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0
    
    logger.info("==========================================")
    logger.info("PROCESSING COMPLETED SUCCESSFULLY")
    logger.info(f"Total Frames Processed: {frame_idx}")
    logger.info(f"Elapsed Time: {total_elapsed:.2f} seconds")
    logger.info(f"Average Processing FPS: {final_fps:.2f}")
    logger.info(f"Target Video FPS: {video_fps:.2f}")
    logger.info(f"Output Video Location: {output_path}")
    logger.info("------------------------------------------")
    logger.info("FINAL TELEMETRY COUNTS")
    
    # Calculate unique tracked object totals
    vehicle_classes = [c for c in config.TARGET_CLASSES if c != "PERSON"]
    total_vehicles_unique = sum(len(seen_objects[c]) for c in vehicle_classes)
    total_persons_unique = len(seen_objects.get("PERSON", set()))
    
    logger.info(f"UNIQUE DETECTED — Vehicles: {total_vehicles_unique} | Persons: {total_persons_unique}")
    tot_veh_inc, tot_pers_inc = counter.get_totals("incoming")
    tot_veh_out, tot_pers_out = counter.get_totals("outgoing")
    logger.info(f"LINE CROSSINGS  — Incoming: {tot_veh_inc} | Outgoing: {tot_veh_out}")
    logger.info("==========================================")
    
    # Map internal unique class counts back to YOLO class names for the frontend table
    unique_objects = {}
    for yolo_cls, internal_cls in config.CLASS_MAP.items():
        unique_objects[yolo_cls] = len(seen_objects.get(internal_cls, set()))
        
    # Generate line crossings details
    line_crossings_total = {}
    for yolo_cls, internal_cls in config.CLASS_MAP.items():
        line_crossings_total[yolo_cls] = counter.counts["incoming"].get(internal_cls, 0) + counter.counts["outgoing"].get(internal_cls, 0)
        
    return {
        "success": True,
        "video_path": output_path,
        "camera_mode": camera_mode,
        "traffic": {
            "unique_objects": unique_objects,
            "line_crossings": {
                "enabled": camera_mode == "stationary",
                "incoming": {yolo_cls: counter.counts["incoming"].get(internal_cls, 0) for yolo_cls, internal_cls in config.CLASS_MAP.items()},
                "outgoing": {yolo_cls: counter.counts["outgoing"].get(internal_cls, 0) for yolo_cls, internal_cls in config.CLASS_MAP.items()},
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
        "elapsed_time": total_elapsed,
        "avg_fps": final_fps,
        # Performance Telemetry
        "input_resolution": f"{width}x{height}",
        "processing_resolution": f"{target_width}x{target_height}",
        "source_fps": round(video_fps, 2),
        "total_frames": total_frames,
        "device": device,
        "inference_size": config.MAX_PROCESSING_DIMENSION
    }

def main():
    args = parse_args()
    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            confidence=args.confidence,
            debug=args.debug,
            display=args.display
        )
    except Exception as e:
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
