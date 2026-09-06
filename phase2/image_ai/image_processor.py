import os
import cv2
import time
import uuid
import datetime
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from phase1.detector import Detector
from phase1.models import TrackedObject
from phase2.anpr import PlateDetector, OCREngine, PlateDetection, OCRResult
from phase2.image_ai import image_config
from phase2.image_ai.exif_parser import extract_image_exif_metadata

logger = logging.getLogger("SIH_Server")

class ImageProcessor:
    """
    Dedicated Still Image Processor for UrbanPulse AI Platform.
    Performs vehicle detection, plate localization, real OCR text extraction,
    spatial vehicle-plate association, and evidence snapshot generation on photographs.
    Strictly prohibits data fabrication.
    """
    def __init__(self):
        self.detector = Detector()
        self.plate_detector = PlateDetector()
        self.ocr_engine = OCREngine(use_easyocr=False)

    def process_image(self, input_image_path: str, output_image_path: str) -> Dict[str, Any]:
        """
        Executes still photo analysis pipeline on a single image file.
        Returns structured JSON response.
        """
        logger.info(f"Starting Still Photo Analysis on image: {input_image_path}")
        start_time = time.time()

        if not os.path.exists(input_image_path):
            return {"success": False, "error": f"Image file not found: {input_image_path}"}

        # 1. Read Image
        frame = cv2.imread(input_image_path)
        if frame is None or frame.size == 0:
            return {"success": False, "error": f"Failed to decode image file: {input_image_path}"}

        h_img, w_img = frame.shape[:2]

        # 2. Extract genuine EXIF Metadata
        exif_meta = extract_image_exif_metadata(input_image_path)
        analysis_iso_time = datetime.datetime.now().astimezone().isoformat()

        timestamp_info = {
            "capture_time": exif_meta["capture_timestamp"],
            "analysis_time": analysis_iso_time,
            "label": "Capture Time" if exif_meta["has_exif_timestamp"] else "Analysis Time"
        }

        location_info = {
            "latitude": exif_meta["gps_latitude"],
            "longitude": exif_meta["gps_longitude"],
            "available": exif_meta["has_exif_gps"]
        }

        # 3. Detect Vehicles on Image
        raw_detections = self.detector.detect(frame)
        
        # Convert Detection objects to TrackedObject structures for spatial association
        tracked_vehicles: List[TrackedObject] = []
        veh_id_counter = 1
        for det in raw_detections:
            if det.class_name in ["PERSON", "RIDER"]:
                continue
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tracked_vehicles.append(TrackedObject(
                track_id=veh_id_counter,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                center_x=cx,
                center_y=cy,
                previous_center_x=cx,
                previous_center_y=cy
            ))
            veh_id_counter += 1

        # 4. Detect Number Plates & Run OCR
        plate_detections = self.plate_detector.detect_plates_in_vehicles(
            frame=frame,
            tracked_vehicles=tracked_vehicles,
            frame_number=1,
            timestamp=0.0
        )

        vehicle_records: List[Dict[str, Any]] = []
        annotated_frame = frame.copy()

        event_seq = 1
        plates_detected_count = 0
        plates_read_count = 0
        unreadable_count = 0

        for veh in tracked_vehicles:
            vx1, vy1, vx2, vy2 = [int(v) for v in veh.bbox]
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w_img, vx2), min(h_img, vy2)

            vehicle_crop = frame[vy1:vy2, vx1:vx2].copy() if (vx2 - vx1) > 5 and (vy2 - vy1) > 5 else None

            # Check if plate was detected for this vehicle
            plate_info = plate_detections.get(veh.track_id)

            if plate_info is not None:
                plate_det, plate_crop = plate_info
                ocr_res = self.ocr_engine.extract_text(plate_crop)
                plates_detected_count += 1

                reg_text = ocr_res.normalized_text if ocr_res.normalized_text != "UNREADABLE" else "UNREADABLE"
                raw_ocr_text = ocr_res.raw_ocr if ocr_res.raw_ocr != "UNREADABLE" else "UNREADABLE"

                if reg_text != "UNREADABLE":
                    plates_read_count += 1
                else:
                    unreadable_count += 1

                plate_conf = plate_det.confidence
                ocr_conf = ocr_res.ocr_confidence
            else:
                plate_det = None
                plate_crop = None
                reg_text = "UNREADABLE"
                raw_ocr_text = "UNREADABLE"
                plate_conf = 0.0
                ocr_conf = 0.0
                unreadable_count += 1

            # Save real evidence image crops
            event_id = f"EVENT_{event_seq:03d}"
            veh_crop_url, plate_crop_url = self._save_image_evidence(event_id, vehicle_crop, plate_crop)
            event_seq += 1

            # Draw annotations on image
            annotated_frame = self._draw_vehicle_plate_annotation(
                annotated_frame, veh, plate_det, reg_text, ocr_conf
            )

            # Record entry matching schema
            vehicle_records.append({
                "vehicle_index": veh.track_id,
                "vehicle_type": veh.class_name,
                "vehicle_confidence": round(veh.confidence, 2),
                "vehicle_bbox": [round(v, 1) for v in veh.bbox],
                "plate": {
                    "text": reg_text,
                    "raw_ocr": raw_ocr_text,
                    "detection_confidence": round(plate_conf, 2),
                    "ocr_confidence": round(ocr_conf, 2) if ocr_conf > 0 else None
                },
                "evidence": {
                    "vehicle_crop": veh_crop_url,
                    "plate_crop": plate_crop_url
                }
            })

        # 5. Write Annotated Image Output
        output_dir = os.path.dirname(output_image_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_image_path, annotated_frame)

        elapsed_time = time.time() - start_time

        return {
            "success": True,
            "source_type": "image",
            "image": {
                "filename": os.path.basename(input_image_path),
                "width": w_img,
                "height": h_img,
                "output_url": f"/processed/images/{os.path.basename(output_image_path)}"
            },
            "timestamp": timestamp_info,
            "location": location_info,
            "sensor_data": {
                "gps": location_info if location_info["available"] else None,
                "imu": None
            },
            "summary": {
                "total_vehicles": len(tracked_vehicles),
                "plates_detected": plates_detected_count,
                "plates_read": plates_read_count,
                "unreadable_plates": unreadable_count
            },
            "detections": vehicle_records,
            "elapsed_time": round(elapsed_time, 2)
        }

    def _draw_vehicle_plate_annotation(
        self,
        frame: np.ndarray,
        veh: TrackedObject,
        plate_det: Optional[PlateDetection],
        reg_text: str,
        ocr_conf: float
    ) -> np.ndarray:
        """Draws bounding boxes and labels on output image."""
        vx1, vy1, vx2, vy2 = [int(v) for v in veh.bbox]

        # Vehicle Box (Green)
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
        veh_label = f"{veh.class_name} ({int(veh.confidence * 100)}%)"
        cv2.putText(frame, veh_label, (vx1, max(15, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Number Plate Box (Cyan)
        if plate_det:
            px1, py1, px2, py2 = [int(v) for v in plate_det.bbox]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)

            plate_lbl = f"PLATE: {reg_text}" + (f" ({int(ocr_conf * 100)}%)" if ocr_conf > 0 else "")
            (tw, th), _ = cv2.getTextSize(plate_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty1 = max(0, py1 - th - 4)
            cv2.rectangle(frame, (px1, ty1), (px1 + tw + 6, ty1 + th + 4), (255, 255, 0), -1)
            cv2.putText(frame, plate_lbl, (px1 + 3, ty1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def _save_image_evidence(
        self,
        event_id: str,
        vehicle_crop: Optional[np.ndarray],
        plate_crop: Optional[np.ndarray]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Saves actual vehicle and plate image crops into processed/events/number_plate/."""
        veh_url = None
        plate_url = None

        if vehicle_crop is not None and vehicle_crop.size > 0:
            veh_filename = f"{event_id}_vehicle.jpg"
            veh_path = os.path.join(image_config.PLATE_EVIDENCE_FOLDER, veh_filename)
            cv2.imwrite(veh_path, vehicle_crop)
            veh_url = f"/processed/events/number_plate/{veh_filename}"

        if plate_crop is not None and plate_crop.size > 0:
            plate_filename = f"{event_id}_plate.jpg"
            plate_path = os.path.join(image_config.PLATE_EVIDENCE_FOLDER, plate_filename)
            cv2.imwrite(plate_path, plate_crop)
            plate_url = f"/processed/events/number_plate/{plate_filename}"

        return veh_url, plate_url
