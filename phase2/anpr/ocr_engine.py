import os
import re
import cv2
import time
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from phase2.anpr import plate_config
from phase2.anpr.plate_models import OCRResult

logger = logging.getLogger("SIH_Server")

class OCREngine:
    """
    Dedicated OCR Engine for Vehicle License Registration Plates.
    Phase B & C Compliant:
    - 4x INTER_CUBIC proportional upscaling
    - Greyscale + CLAHE contrast enhancement
    - Aspect ratio pre-filtering (2:1 to 5:1)
    - EasyOCR alphanumeric allowlist restriction
    - Position-aware Indian registration plate normalization (Standard & Bharat series)
    - Strict validation: 'validated' | 'format_rejected' | 'low_confidence' | 'no_text'
    - Raw and preprocessed crop persistence
    """
    # Position-aware character confusion mappings (C1)
    DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G'}
    LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6'}

    # Regex patterns (C2)
    STANDARD_PATTERN = re.compile(plate_config.INDIAN_STANDARD_PATTERN)
    BHARAT_PATTERN = re.compile(plate_config.INDIAN_BHARAT_PATTERN)

    def __init__(self, use_easyocr: bool = True, allowlist: str = plate_config.OCR_ALLOWLIST):
        self.reader = None
        self.allowlist = allowlist
        if use_easyocr:
            try:
                import easyocr
                # Instantiate offline EasyOCR reader
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info(f"EasyOCR initialized successfully with allowlist: {self.allowlist}")
            except Exception as e:
                logger.warning(f"EasyOCR initialization unavailable ({e}). Using fallback CV engine.")

    @staticmethod
    def check_aspect_ratio(
        plate_crop: np.ndarray,
        min_aspect_ratio: float = plate_config.PLATE_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = plate_config.PLATE_MAX_ASPECT_RATIO
    ) -> bool:
        """
        C5 — Optional pre-filter: Rejects crops whose aspect ratio (width:height)
        falls outside roughly 2:1 to 5:1 before calling OCR.
        """
        if plate_crop is None or plate_crop.size == 0:
            return False
        h, w = plate_crop.shape[:2]
        if h <= 0 or w <= 0:
            return False
        aspect_ratio = float(w) / float(h)
        return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio

    def preprocess_plate_crop(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        PHASE B Preprocessing:
        1. Proportional 4x upscaling with cv2.resize(..., interpolation=cv2.INTER_CUBIC).
        2. Conversion to greyscale.
        3. CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=(8, 8)).
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]

        # 1. Upscale 4x with INTER_CUBIC
        target_w = max(1, int(w * plate_config.PREPROC_UPSCALE_FACTOR))
        target_h = max(1, int(h * plate_config.PREPROC_UPSCALE_FACTOR))
        upscaled = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to Grayscale
        if len(upscaled.shape) == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = upscaled.copy()

        # 3. CLAHE Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        preprocessed = clahe.apply(gray)

        return preprocessed

    def normalize_indian_plate(self, raw_text: str) -> Tuple[str, bool]:
        """
        C1 & C2 — Normalises common OCR confusions using a position-aware correction pass
        and validates against Standard and Bharat series formats.
        - Strips spaces, hyphens, and non-alphanumerics, then uppercases.
        - Letter positions: maps 0->O, 1->I, 5->S, 8->B, 2->Z, 6->G.
        - Digit positions: maps O->0, I->1, S->5, B->8, Z->2, G->6.
        - Validates against:
          Standard: ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$ (e.g. GJ01AB1234, MH12A5678)
          Bharat:   ^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$ (e.g. 22BH1234AA)
        """
        if not raw_text:
            return "", False

        # Strip all spaces, hyphens and non-alphanumeric characters, and uppercase everything
        clean = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        if len(clean) < 7 or len(clean) > 12:
            return clean, False

        # If already directly matching either pattern, return immediately
        if self.STANDARD_PATTERN.match(clean) or self.BHARAT_PATTERN.match(clean):
            return clean, True

        L = len(clean)
        chars = list(clean)

        # --- Candidate 1: Standard Indian Registration (^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$) ---
        # Total length between 8 and 11
        if 8 <= L <= 11:
            # First 2 positions MUST be letters
            p0 = self.DIGIT_TO_LETTER.get(chars[0], chars[0])
            p1 = self.DIGIT_TO_LETTER.get(chars[1], chars[1])

            # Last 4 positions MUST be digits
            last4 = [self.LETTER_TO_DIGIT.get(chars[i], chars[i]) for i in range(L - 4, L)]

            # Middle segment: clean[2 : L-4] has length M = L - 6 (where 2 <= M <= 5)
            # Middle consists of `d` district digits (1..2) followed by `s` series letters (1..3) where d + s == M
            middle_len = L - 6
            for d in [2, 1]:  # Try 2 district digits first (standard like MH12, GJ01), then 1 digit (DL1)
                s = middle_len - d
                if 1 <= s <= 3:
                    # District digits: indices 2 to 2+d
                    dist = [self.LETTER_TO_DIGIT.get(chars[2 + j], chars[2 + j]) for j in range(d)]
                    # Series letters: indices 2+d to 2+d+s
                    series = [self.DIGIT_TO_LETTER.get(chars[2 + d + k], chars[2 + d + k]) for k in range(s)]

                    candidate = p0 + p1 + "".join(dist) + "".join(series) + "".join(last4)
                    if self.STANDARD_PATTERN.match(candidate):
                        return candidate, True

        # --- Candidate 2: Bharat (BH) Series (^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$) ---
        # Total length 9 or 10
        if 9 <= L <= 10:
            # Positions 0, 1 MUST be digits
            y0 = self.LETTER_TO_DIGIT.get(chars[0], chars[0])
            y1 = self.LETTER_TO_DIGIT.get(chars[1], chars[1])

            # Positions 2, 3 MUST be letters "BH"
            b_char = self.DIGIT_TO_LETTER.get(chars[2], chars[2])
            h_char = chars[3]  # 'H'
            if b_char == '8': b_char = 'B'

            # Positions 4, 5, 6, 7 MUST be digits
            bh_digits = [self.LETTER_TO_DIGIT.get(chars[i], chars[i]) for i in range(4, 8)]

            # Last 1 or 2 characters MUST be letters
            bh_series = [self.DIGIT_TO_LETTER.get(chars[i], chars[i]) for i in range(8, L)]

            bh_candidate = y0 + y1 + b_char + h_char + "".join(bh_digits) + "".join(bh_series)
            if self.BHARAT_PATTERN.match(bh_candidate):
                return bh_candidate, True

        return clean, False

    def extract_text(
        self,
        plate_crop: np.ndarray,
        frame_number: int = 0,
        track_id: int = 0,
        min_aspect_ratio: float = plate_config.PLATE_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = plate_config.PLATE_MAX_ASPECT_RATIO,
        save_crops: bool = True
    ) -> OCRResult:
        """
        Executes complete Phase B & C OCR pipeline on a plate crop:
        1. Aspect ratio pre-filter (2:1 to 5:1).
        2. Proportional 4x INTER_CUBIC upscaling + greyscale + CLAHE.
        3. Saves raw and preprocessed crops to disk.
        4. EasyOCR inference with strict uppercase alphanumeric allowlist.
        5. Position-aware Indian plate format normalization (C1/C2).
        6. Reject rather than guess rule (C3/C4):
           - plate: validated string or None
           - plate_confidence: float
           - plate_status: 'validated' | 'format_rejected' | 'low_confidence' | 'no_text'
           - ocr_raw: raw text
        """
        if plate_crop is None or plate_crop.size == 0:
            return OCRResult(
                raw_ocr="",
                normalized_text="UNREADABLE",
                plate=None,
                ocr_confidence=0.0,
                plate_confidence=0.0,
                plate_status="no_text",
                is_valid_pattern=False
            )

        # C5 Pre-filter: Reject crops whose aspect ratio is outside 2:1 to 5:1
        if not self.check_aspect_ratio(plate_crop, min_aspect_ratio, max_aspect_ratio):
            logger.debug(f"Plate crop rejected by aspect ratio pre-filter ({plate_crop.shape[1]}:{plate_crop.shape[0]})")
            return OCRResult(
                raw_ocr="",
                normalized_text="UNREADABLE",
                plate=None,
                ocr_confidence=0.0,
                plate_confidence=0.0,
                plate_status="no_text",
                is_valid_pattern=False,
                raw_crop=plate_crop
            )

        # Phase B Preprocessing: 4x upscale + Greyscale + CLAHE
        preprocessed = self.preprocess_plate_crop(plate_crop)

        # Phase B: Save both raw crop and preprocessed crop to disk
        if save_crops and plate_config.OCR_DEBUG_CROPS_DIR:
            try:
                crop_ts = int(time.time() * 1000) % 1000000
                raw_crop_path = os.path.join(
                    plate_config.OCR_DEBUG_CROPS_DIR,
                    f"f{frame_number}_t{track_id}_{crop_ts}_raw.jpg"
                )
                prep_crop_path = os.path.join(
                    plate_config.OCR_DEBUG_CROPS_DIR,
                    f"f{frame_number}_t{track_id}_{crop_ts}_preprocessed.jpg"
                )
                cv2.imwrite(raw_crop_path, plate_crop)
                cv2.imwrite(prep_crop_path, preprocessed)
            except Exception as e:
                logger.debug(f"Failed to persist debug crops to disk: {e}")

        raw_ocr = ""
        confidence = 0.0

        if self.reader is not None:
            try:
                # Restrict EasyOCR with alphanumeric allowlist
                results = self.reader.readtext(
                    preprocessed,
                    allowlist=self.allowlist
                )
                if results:
                    valid_res = [r for r in results if r[1] and str(r[1]).strip()]
                    if valid_res:
                        if len(valid_res) > 1:
                            # Sort detections top-to-bottom, left-to-right for multi-line plates
                            sorted_res = sorted(valid_res, key=lambda x: (x[0][0][1], x[0][0][0]))
                            joined_text = "".join([str(r[1]) for r in sorted_res])
                            avg_conf = sum([float(r[2]) for r in sorted_res]) / len(sorted_res)
                            # Check if joined string matches valid pattern
                            _, is_v = self.normalize_indian_plate(joined_text)
                            if is_v:
                                raw_ocr = joined_text
                                confidence = avg_conf
                            else:
                                best_res = max(valid_res, key=lambda x: x[2])
                                raw_ocr = str(best_res[1])
                                confidence = float(best_res[2])
                        else:
                            best_res = valid_res[0]
                            raw_ocr = str(best_res[1])
                            confidence = float(best_res[2])
            except Exception as e:
                logger.warning(f"EasyOCR read error: {e}")

        # Position-aware normalization (C1 & C2)
        normalized, is_valid_pattern = self.normalize_indian_plate(raw_ocr)

        # C3 & C4: Explicit outcome classification and reject-rather-than-guess
        clean_raw = raw_ocr.strip()
        if not clean_raw:
            plate_status = "no_text"
            plate = None
            final_conf = 0.0
        elif confidence < plate_config.OCR_CONF_THRESHOLD:
            # Below 0.40 confidence -> low_confidence
            plate_status = "low_confidence"
            plate = None
            final_conf = round(confidence, 2)
        elif not is_valid_pattern:
            # Confidence >= 0.40 but format doesn't match standard or Bharat series -> format_rejected
            plate_status = "format_rejected"
            plate = None
            final_conf = round(confidence, 2)
        else:
            # Validated plate: matches regex and confidence >= 0.40
            plate_status = "validated"
            plate = normalized
            final_conf = round(confidence, 2)

        return OCRResult(
            raw_ocr=clean_raw,
            normalized_text=plate if plate else ("UNREADABLE" if not clean_raw else normalized),
            plate=plate,
            ocr_confidence=final_conf,
            plate_confidence=final_conf,
            plate_status=plate_status,
            is_valid_pattern=(plate_status == "validated"),
            raw_crop=plate_crop,
            preprocessed_crop=preprocessed
        )


