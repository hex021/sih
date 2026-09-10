import os
import cv2
import numpy as np
import logging
from typing import Optional

from . import utils

logger = utils.setup_logger("SIH_CameraMotion")

def detect_camera_motion(
    video_path: str,
    max_frames: int = 60,
    motion_threshold: float = 1.2,
    sample_stride: int = 2
) -> str:
    """
    Automatically determines whether a video input represents a 'stationary' (fixed CCTV/junction)
    or 'moving' (vehicle-mounted mobile) camera using Lucas-Kanade optical flow on background regions.

    :param video_path: Path to input video file.
    :param max_frames: Maximum number of initial frames to sample for motion analysis.
    :param motion_threshold: Average background displacement threshold in pixels/frame.
                             Displacements >= threshold are classified as 'moving'.
    :param sample_stride: Frame stride during analysis (processes every Nth frame for efficiency).
    :return: "stationary" or "moving"
    """
    if not video_path or not isinstance(video_path, str) or not os.path.exists(video_path):
        logger.warning(f"Video file path invalid or missing: '{video_path}'. Defaulting camera_mode to 'stationary'.")
        return "stationary"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Failed to open video stream: {video_path}. Defaulting camera_mode to 'stationary'.")
        return "stationary"

    prev_gray: Optional[np.ndarray] = None
    prev_pts: Optional[np.ndarray] = None
    frame_count = 0
    frame_displacements = []

    # Optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    feature_params = dict(
        maxCorners=150,
        qualityLevel=0.03,
        minDistance=10,
        blockSize=7
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            frame_count += 1
            if frame_count % sample_stride != 0 and prev_gray is not None:
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Create background ROI mask: Top 45% of frame (excluding lower region where vehicles move)
            mask = np.zeros_like(gray, dtype=np.uint8)
            bg_h = int(h * 0.45)
            if bg_h > 10:
                mask[0:bg_h, :] = 255
            else:
                mask[:, :] = 255

            if prev_gray is None:
                prev_gray = gray
                prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                continue

            if prev_pts is None or len(prev_pts) < 5:
                # Re-detect background keypoints if tracking lost or sparse
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)
                if prev_pts is None or len(prev_pts) < 5:
                    prev_gray = gray
                    continue

            # Calculate Lucas-Kanade optical flow
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

            if next_pts is not None and status is not None:
                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]

                if len(good_new) > 0:
                    # Calculate vector displacements
                    dx = good_new[:, 0] - good_old[:, 0]
                    dy = good_new[:, 1] - good_old[:, 1]
                    magnitudes = np.sqrt(dx ** 2 + dy ** 2)

                    # Filter out tracking noise/outliers (displacements > 50px/frame)
                    valid_mags = magnitudes[magnitudes < 50.0]

                    if len(valid_mags) > 0:
                        # Normalize displacement by frame stride
                        frame_disp = np.median(valid_mags) / float(sample_stride)
                        frame_displacements.append(frame_disp)

                    # Prepare points for next step
                    prev_pts = good_new.reshape(-1, 1, 2)
                else:
                    prev_pts = None
            else:
                prev_pts = None

            prev_gray = gray

    except Exception as e:
        logger.warning(f"Error during camera motion detection on {video_path}: {e}. Defaulting to 'stationary'.")
        return "stationary"
    finally:
        cap.release()

    if not frame_displacements:
        logger.info("Insufficient background feature motion detected. Defaulting to 'stationary'.")
        return "stationary"

    avg_motion = float(np.mean(frame_displacements))
    camera_mode = "moving" if avg_motion >= motion_threshold else "stationary"

    logger.info(f"Camera motion analysis complete for {os.path.basename(video_path)}: avg_motion={avg_motion:.2f} px/frame -> camera_mode='{camera_mode}'")
    return camera_mode
