import os
import csv
import logging

logger = logging.getLogger("SIH_GPSTrack")


class GPSTrack:
    """
    Loads frame-by-frame GPS track data from CSV and provides frame-level
    geolocations. Degrades gracefully if input/track.csv is missing.
    """

    def __init__(self, csv_path: str = None):
        self._available = False
        self._frames = {}

        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(base_dir, "input", "track.csv")

        self.csv_path = csv_path

        if not os.path.exists(csv_path):
            logger.warning(f"GPS track file not found at '{csv_path}' — degrading gracefully with location=None")
            return

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_idx = int(row["frame_idx"])
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    speed = float(row.get("speed_kmph", 0.0) or 0.0)
                    heading = float(row.get("heading_deg", 0.0) or 0.0)

                    self._frames[frame_idx] = {
                        "latitude": lat,
                        "longitude": lon,
                        "speed_kmph": speed,
                        "heading_deg": heading,
                        "source": "simulated"
                    }

            if self._frames:
                self._available = True
                self._sorted_keys = sorted(self._frames.keys())
                self._min_frame = self._sorted_keys[0]
                self._max_frame = self._sorted_keys[-1]
            else:
                logger.warning(f"GPS track file '{csv_path}' is empty.")

        except Exception as e:
            logger.warning(f"Failed to read GPS track file '{csv_path}': {e}")
            self._available = False
            self._frames = {}

    @property
    def available(self) -> bool:
        """True if a valid track was loaded."""
        return self._available

    def get_location(self, frame_idx: int) -> dict | None:
        """
        Returns the TASK 00.5 location object for this frame.
        - not available            -> None
        - frame_idx present        -> that row
        - frame_idx beyond the end -> nearest available frame (clamp, never extrapolate)
        Never raises.
        """
        if not self._available or not self._frames:
            return None

        try:
            if frame_idx in self._frames:
                return dict(self._frames[frame_idx])

            if frame_idx < self._min_frame:
                return dict(self._frames[self._min_frame])

            if frame_idx > self._max_frame:
                return dict(self._frames[self._max_frame])

            # Nearest key fallback if gaps exist
            nearest_key = min(self._frames.keys(), key=lambda k: abs(k - frame_idx))
            return dict(self._frames[nearest_key])

        except Exception as e:
            logger.warning(f"Error accessing GPS track location for frame {frame_idx}: {e}")
            return None

    def total_frames(self) -> int:
        """Number of rows loaded, 0 if unavailable."""
        return len(self._frames) if self._available else 0
