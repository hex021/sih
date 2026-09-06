from typing import List, Tuple
from phase1 import config
from phase1 import roi
from phase1.models import TrackedObject

def classify_density(tracked_objects: List[TrackedObject], frame_width: int, frame_height: int, roi_relative: Tuple[float, float, float, float] = None) -> Tuple[str, int]:
    """
    Classifies the current traffic density based on the number of active
    tracked vehicles inside the Region of Interest (ROI).
    Pedestrians (PERSON) are excluded.
    
    Returns: (density_class, active_vehicles_count)
             where density_class is 'LOW', 'MEDIUM', or 'HIGH'.
    """
    active_vehicles_in_roi = 0
    
    for obj in tracked_objects:
        # Exclude PERSON from traffic density
        if obj.class_name != "PERSON":
            center = (obj.center_x, obj.center_y)
            if roi.is_inside(center, frame_width, frame_height, roi_relative):
                active_vehicles_in_roi += 1
                
    # Classify based on config thresholds
    low_max = config.DENSITY_THRESHOLDS.get("LOW_MAX", 5)
    medium_max = config.DENSITY_THRESHOLDS.get("MEDIUM_MAX", 15)
    
    if active_vehicles_in_roi <= low_max:
        density_class = "LOW"
    elif active_vehicles_in_roi <= medium_max:
        density_class = "MEDIUM"
    else:
        density_class = "HIGH"
        
    return density_class, active_vehicles_in_roi
