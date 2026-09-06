from typing import Tuple
import cv2
import numpy as np
from phase1 import config

def get_pixel_roi(frame_width: int, frame_height: int, roi_relative: Tuple[float, float, float, float] = None) -> Tuple[int, int, int, int]:
    """
    Converts relative ROI coordinates (0.0 to 1.0) to absolute pixel boundaries
    based on the current frame dimensions.
    Returns: (x_min, y_min, x_max, y_max) in pixel space.
    """
    rx_min, ry_min, rx_max, ry_max = roi_relative if roi_relative is not None else config.ROI_RELATIVE
    
    x_min = int(rx_min * frame_width)
    y_min = int(ry_min * frame_height)
    x_max = int(rx_max * frame_width)
    y_max = int(ry_max * frame_height)
    
    return x_min, y_min, x_max, y_max

def is_inside(point: Tuple[float, float], frame_width: int, frame_height: int, roi_relative: Tuple[float, float, float, float] = None) -> bool:
    """
    Determines if a given center point (x, y) lies inside the ROI.
    Supports boundary conditions.
    """
    px, py = point
    x_min, y_min, x_max, y_max = get_pixel_roi(frame_width, frame_height, roi_relative)
    return x_min <= px <= x_max and y_min <= py <= y_max
