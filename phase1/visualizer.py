import cv2
import numpy as np
from typing import List, Dict, Tuple
from phase1 import config
from phase1 import roi
from phase1.models import TrackedObject
from phase1.counter import Counter

class Visualizer:
    """
    Responsible only for rendering overlays on video frames.
    Decoupled from detection, tracking, counting, and density estimation business logic.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        counter: Counter,
        density_class: str,
        active_roi_count: int,
        fps: float = 0.0,
        frame_num: int = 0,
        track_history: Dict[int, List[Tuple[float, float]]] = None,
        line_y_rel: float = None,
        roi_relative: Tuple[float, float, float, float] = None,
        camera_mode: str = "stationary",
        unique_counts: Dict[str, int] = None
    ) -> np.ndarray:
        """
        Draws HUD panel, ROI, count line, and bounding boxes on the frame.
        """
        h, w = frame.shape[:2]
        
        # 1. Draw ROI Zone
        x1_roi, y1_roi, x2_roi, y2_roi = roi.get_pixel_roi(w, h, roi_relative)
        if self.debug:
            # Draw semi-transparent cyan overlay inside the ROI, optimized to blend only target crop
            sub_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            overlay = sub_roi.copy()
            cv2.rectangle(overlay, (0, 0), (x2_roi - x1_roi, y2_roi - y1_roi), config.ROI_COLOR, -1)
            cv2.addWeighted(overlay, 0.08, sub_roi, 0.92, 0, sub_roi)
            frame[y1_roi:y2_roi, x1_roi:x2_roi] = sub_roi
            
            # Draw ROI boundary border
            cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), config.ROI_COLOR, 2)
            cv2.putText(frame, f"ROI ({camera_mode.upper()} ZONE)", (x1_roi + 5, y1_roi + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.ROI_COLOR, 1, cv2.LINE_AA)
        else:
            # Clean mode: thin boundary border for ROI
            cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), config.ROI_COLOR, 1)

        # 2. Draw Count Line (Stationary Camera Mode only)
        if camera_mode == "stationary":
            rel_y = line_y_rel if line_y_rel is not None else config.COUNT_LINE_RELATIVE_Y
            line_y = int(rel_y * h)
            cv2.line(frame, (0, line_y), (w, line_y), config.COUNT_LINE_COLOR, 2)
            cv2.putText(frame, "COUNT LINE", (w - 120, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COUNT_LINE_COLOR, 1, cv2.LINE_AA)

        # 3. Draw Tracked Objects
        for obj in tracked_objects:
            # Contextual suppression: hide riders/occupants in normal mode
            if obj.class_name in ("RIDER", "OCCUPANT") and not self.debug:
                continue
                
            bx1, by1, bx2, by2 = int(obj.bbox[0]), int(obj.bbox[1]), int(obj.bbox[2]), int(obj.bbox[3])
            
            # Use orange/magenta for suppressed riders/occupants in debug mode, green for others
            color = (255, 128, 0) if obj.class_name in ("RIDER", "OCCUPANT") else config.BBOX_COLOR
            
            # Draw bounding box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, config.BBOX_THICKNESS)
            
            # Label string: "CLASS #ID CONF"
            label = f"{obj.class_name} #{obj.track_id} {obj.confidence:.2f}"
            
            # Draw label box
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, config.TEXT_THICKNESS)
            cv2.rectangle(frame, (bx1, by1 - lh - 6), (bx1 + lw, by1), color, -1)
            cv2.putText(frame, label, (bx1, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, config.TEXT_COLOR, config.TEXT_THICKNESS, cv2.LINE_AA)
            
            # Debug overlays: center point and trajectories
            if self.debug:
                cx, cy = int(obj.center_x), int(obj.center_y)
                cv2.circle(frame, (cx, cy), 4, config.COUNT_LINE_COLOR, -1)
                
                # Draw trailing trajectory from history
                if track_history and obj.track_id in track_history:
                    pts = list(track_history[obj.track_id])
                    for i in range(1, len(pts)):
                        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                        p2 = (int(pts[i][0]), int(pts[i][1]))
                        cv2.line(frame, p1, p2, config.TRAJECTORY_COLOR, 2)

        # 4. Draw HUD Dashboard Panel
        self._draw_hud_panel(frame, counter, density_class, active_roi_count, camera_mode, unique_counts)

        # 5. Draw Debug Info in top-right
        if self.debug:
            db_label_fps = f"Processing FPS: {fps:.1f}"
            db_label_frame = f"Frame: {frame_num}"
            cv2.putText(frame, db_label_fps, (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, db_label_frame, (w - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
        return frame

    def _draw_hud_panel(self, frame: np.ndarray, counter: Counter, density_class: str, active_roi_count: int, camera_mode: str = "stationary", unique_counts: Dict[str, int] = None):
        """
        Draws a high-polish, semi-transparent telemetry overlay panel.
        """
        # Coordinates for panel
        px1, py1, px2, py2 = 20, 20, 360, 340
        
        # Semi-transparent box background on region of interest to optimize CPU performance
        sub_frame = frame[py1:py2, px1:px2]
        overlay = sub_frame.copy()
        cv2.rectangle(overlay, (0, 0), (px2 - px1, py2 - py1), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, sub_frame, 0.25, 0, sub_frame)
        frame[py1:py2, px1:px2] = sub_frame
        
        # Draw border
        cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)
        
        # Draw stats text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.40
        thick = 1
        color = config.HUD_COLOR
        
        y_offset = py1 + 25
        
        # Header
        header_text = f"TRAFFIC MONITOR ({camera_mode.upper()})"
        cv2.putText(frame, header_text, (px1 + 15, y_offset), font, 0.5, (255, 165, 0), 1, cv2.LINE_AA)
        y_offset += 25
        
        if camera_mode == "stationary":
            # Subheaders
            cv2.putText(frame, f"{'CLASS':<15}{'INCOMING':<12}{'OUTGOING':<12}", (px1 + 15, y_offset), font, scale, (180, 180, 180), thick, cv2.LINE_AA)
            y_offset += 12
            cv2.line(frame, (px1 + 15, y_offset), (px2 - 15, y_offset), (80, 80, 80), 1)
            y_offset += 18
            
            # Class counts
            for cls in config.TARGET_CLASSES:
                if cls == "PERSON":
                    continue # Print person separately below
                inc_val = counter.counts["incoming"].get(cls, 0)
                out_val = counter.counts["outgoing"].get(cls, 0)
                
                # Format text line
                row_text = f"{cls.capitalize():<18}{inc_val:<15}{out_val:<15}"
                cv2.putText(frame, row_text, (px1 + 15, y_offset), font, scale, color, thick, cv2.LINE_AA)
                y_offset += 18
                
            cv2.line(frame, (px1 + 15, y_offset), (px2 - 15, y_offset), (80, 80, 80), 1)
            y_offset += 18
            
            # Totals
            tot_veh_inc, tot_pers_inc = counter.get_totals("incoming")
            tot_veh_out, tot_pers_out = counter.get_totals("outgoing")
            
            veh_row = f"{'Total Vehicles':<18}{tot_veh_inc:<15}{tot_veh_out:<15}"
            cv2.putText(frame, veh_row, (px1 + 15, y_offset), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
            y_offset += 18
            
            pers_row = f"{'Total Persons':<18}{tot_pers_inc:<15}{tot_pers_out:<15}"
            cv2.putText(frame, pers_row, (px1 + 15, y_offset), font, scale, (255, 255, 0), thick, cv2.LINE_AA)
            y_offset += 18
        else:
            # Moving mode HUD uses Unique Tracked Object Counts
            cv2.putText(frame, f"{'CLASS':<20}{'UNIQUE DETECTED':<15}", (px1 + 15, y_offset), font, scale, (180, 180, 180), thick, cv2.LINE_AA)
            y_offset += 12
            cv2.line(frame, (px1 + 15, y_offset), (px2 - 15, y_offset), (80, 80, 80), 1)
            y_offset += 18
            
            # Class counts
            for cls in config.TARGET_CLASSES:
                if cls == "PERSON":
                    continue
                cnt_val = unique_counts.get(cls, 0) if unique_counts else 0
                row_text = f"{cls.capitalize():<24}{cnt_val:<15}"
                cv2.putText(frame, row_text, (px1 + 15, y_offset), font, scale, color, thick, cv2.LINE_AA)
                y_offset += 18
                
            cv2.line(frame, (px1 + 15, y_offset), (px2 - 15, y_offset), (80, 80, 80), 1)
            y_offset += 18
            
            # Totals
            tot_veh = sum(unique_counts.get(c, 0) for c in config.TARGET_CLASSES if c != "PERSON") if unique_counts else 0
            tot_pers = unique_counts.get("PERSON", 0) if unique_counts else 0
            
            veh_row = f"{'Total Vehicles':<24}{tot_veh:<15}"
            cv2.putText(frame, veh_row, (px1 + 15, y_offset), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
            y_offset += 18
            
            pers_row = f"{'Total Persons':<24}{tot_pers:<15}"
            cv2.putText(frame, pers_row, (px1 + 15, y_offset), font, scale, (255, 255, 0), thick, cv2.LINE_AA)
            y_offset += 18
            
        cv2.line(frame, (px1 + 15, y_offset), (px2 - 15, y_offset), (80, 80, 80), 1)
        y_offset += 22
        
        # Density telemetry
        density_label = f"Density: {density_class} (Active Vehicles in ROI: {active_roi_count})"
        density_color = (0, 255, 0) # Green
        if density_class == "MEDIUM":
            density_color = (0, 255, 255) # Yellow
        elif density_class == "HIGH":
            density_color = (0, 0, 255) # Red
            
        cv2.putText(frame, density_label, (px1 + 15, y_offset), font, 0.45, density_color, 1, cv2.LINE_AA)
