from typing import Dict, List, Set, Tuple
from phase1 import config
from phase1.models import TrackedObject

class Counter:
    """
    Handles virtual line-crossing vehicle counting.
    Distinguishes INCOMING and OUTGOING traffic, counts by class,
    separates vehicles from pedestrians, and prevents duplicate counting.
    """
    def __init__(self):
        # Initialize class-wise counters for incoming and outgoing directions
        self.counts = {
            "incoming": {cls: 0 for cls in config.TARGET_CLASSES},
            "outgoing": {cls: 0 for cls in config.TARGET_CLASSES}
        }
        
        # Track IDs already counted for each direction to prevent duplicate counts
        self.counted_incoming: Set[int] = set()
        self.counted_outgoing: Set[int] = set()

    def update(self, tracked_objects: List[TrackedObject], frame_width: int, frame_height: int, line_y_rel: float = None):
        """
        Evaluates tracked objects for line crossings and updates count states.
        Uses center coordinate history to check relative line crossings.
        """
        rel_y = line_y_rel if line_y_rel is not None else config.COUNT_LINE_RELATIVE_Y
        line_y = rel_y * frame_height
        
        for obj in tracked_objects:
            prev_y = obj.previous_center_y
            curr_y = obj.center_y
            track_id = obj.track_id
            cls_name = obj.class_name
            
            # Check for downward crossing (INCOMING)
            # prev_y is strictly above the line, curr_y is on or below the line
            if prev_y < line_y <= curr_y:
                if track_id not in self.counted_incoming:
                    self.counted_incoming.add(track_id)
                    if cls_name in self.counts["incoming"]:
                        self.counts["incoming"][cls_name] += 1
                        print(f"[COUNT] track_id={track_id} class={cls_name.lower()} previous_y={prev_y:.2f} current_y={curr_y:.2f} line_y={line_y:.2f} direction=INCOMING counted=true", flush=True)
                        
            # Check for upward crossing (OUTGOING)
            # prev_y is strictly below the line, curr_y is on or above the line
            elif prev_y > line_y >= curr_y:
                if track_id not in self.counted_outgoing:
                    self.counted_outgoing.add(track_id)
                    if cls_name in self.counts["outgoing"]:
                        self.counts["outgoing"][cls_name] += 1
                        print(f"[COUNT] track_id={track_id} class={cls_name.lower()} previous_y={prev_y:.2f} current_y={curr_y:.2f} line_y={line_y:.2f} direction=OUTGOING counted=true", flush=True)

    def get_totals(self, direction: str = None) -> Tuple[int, int]:
        """
        Calculates and returns (total_vehicles, total_persons)
        Parameters:
            direction: 'incoming', 'outgoing', or None (sum of both).
        """
        vehicle_classes = [c for c in config.TARGET_CLASSES if c != "PERSON"]
        
        if direction in ("incoming", "outgoing"):
            vehicles = sum(self.counts[direction][c] for c in vehicle_classes)
            persons = self.counts[direction].get("PERSON", 0)
            return vehicles, persons
        else:
            vehicles_incoming = sum(self.counts["incoming"][c] for c in vehicle_classes)
            vehicles_outgoing = sum(self.counts["outgoing"][c] for c in vehicle_classes)
            persons_incoming = self.counts["incoming"].get("PERSON", 0)
            persons_outgoing = self.counts["outgoing"].get("PERSON", 0)
            
            return (vehicles_incoming + vehicles_outgoing), (persons_incoming + persons_outgoing)
