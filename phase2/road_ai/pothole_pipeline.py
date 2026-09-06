import os
import time
import cv2
import torch
import numpy as np
from typing import List, Dict, Any

from .pothole_config import POTHOLE_MODEL_PATH, POTHOLE_CONFIDENCE_THRESHOLD, POTHOLE_IOU_THRESHOLD, EVENTS_FOLDER
from .pothole_detector import PotholeDetector
from .pothole_models import PotholeEvent

def run_road_pipeline(
    input_path: str,
    output_path: str,
    confidence: float = None,
    debug: bool = False,
    camera_mode: str = "stationary"
) -> dict:
    """
    Executes the Phase 2.1 Road Condition Analysis Pipeline on a video.
    Detects potholes, suppresses frame duplicates, saves event snapshots,
    and writes annotated H.264 video.
    """
    from phase1 import utils
    from phase1 import config as phase1_config
    
    logger = utils.setup_logger("SIH_Road_Pipeline")
    logger.info("Initializing SIH Phase 2.1 Pothole Detection Pipeline...")
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"Execution Device: {device}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
        
    detector = PotholeDetector(model_path=POTHOLE_MODEL_PATH)
    logger.info(f"Pothole model successfully loaded from {POTHOLE_MODEL_PATH}")
    
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
        temp_resized_path = os.path.join(os.path.dirname(output_path), f"temp_resized_road_{os.path.basename(input_path)}")
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
    all_events: List[PotholeEvent] = []
    
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
                
            # Run pothole detection and deduplication
            detections, new_events = detector.detect(frame_to_process, frame_idx, timestamp)
            all_events.extend(new_events)
            
            # Annotated Frame
            annotated_frame = frame_to_process.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                # Draw pothole box (color: orange/red (0, 120, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
                
                label = f"POTHOLE {det.confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - lh - 6), (x1 + lw, y1), (0, 120, 255), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                            
            # Render Road AI HUD Overlay
            cv2.rectangle(annotated_frame, (15, 15), (315, 100), (30, 30, 30), -1)
            cv2.rectangle(annotated_frame, (15, 15), (315, 100), (0, 120, 255), 1)
            cv2.putText(annotated_frame, "ROAD QUALITY ANALYSIS", (25, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Pothole Events: {len(all_events)}", (25, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}", (25, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
                        
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
    
    return {
        "success": True,
        "mode": "road",
        "events": all_events,
        "total_potholes": len(all_events),
        "total_frames": frame_idx,
        "elapsed_time": elapsed,
        "avg_fps": avg_fps,
        "input_resolution": f"{width}x{height}",
        "processing_resolution": f"{target_width}x{target_height}",
        "source_fps": video_fps,
        "device": device,
        "inference_size": 640
    }
