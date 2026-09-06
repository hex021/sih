import logging
from typing import Optional, Dict, Tuple, Any
from PIL import Image, ExifTags

logger = logging.getLogger("SIH_Server")

def extract_image_exif_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts genuine EXIF timestamp and GPS coordinates from an image file.
    Returns None for timestamp or GPS when EXIF data is not present.
    NEVER fabricates fake timestamps or GPS coordinates.
    """
    metadata = {
        "capture_timestamp": None,
        "gps_latitude": None,
        "gps_longitude": None,
        "has_exif_gps": False,
        "has_exif_timestamp": False
    }

    try:
        with Image.open(image_path) as img:
            exif_raw = img._getexif()
            if not exif_raw:
                return metadata

            exif = {
                ExifTags.TAGS[k]: v
                for k, v in exif_raw.items()
                if k in ExifTags.TAGS
            }


        # 1. Genuine EXIF Timestamp
        dt = exif.get("DateTimeOriginal") or exif.get("DateTimeDigitized") or exif.get("DateTime")
        if dt and isinstance(dt, str):
            metadata["capture_timestamp"] = dt
            metadata["has_exif_timestamp"] = True

        # 2. Genuine EXIF GPS Coordinates
        gps_info = exif.get("GPSInfo")
        if gps_info and isinstance(gps_info, dict):
            gps_tags = {}
            for t in gps_info:
                sub_tag = ExifTags.GPSTAGS.get(t, t)
                gps_tags[sub_tag] = gps_info[t]

            lat, lon = _convert_gps_to_deg(gps_tags)
            if lat is not None and lon is not None:
                metadata["gps_latitude"] = round(lat, 6)
                metadata["gps_longitude"] = round(lon, 6)
                metadata["has_exif_gps"] = True

    except Exception as e:
        logger.debug(f"EXIF parsing skipped or unavailable for {image_path}: {e}")

    return metadata


def _convert_gps_to_deg(gps_tags: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Converts EXIF GPS coordinate tuples (degrees, minutes, seconds) into decimal degrees."""
    try:
        lat_tuple = gps_tags.get("GPSLatitude")
        lat_ref = gps_tags.get("GPSLatitudeRef", "N")
        lon_tuple = gps_tags.get("GPSLongitude")
        lon_ref = gps_tags.get("GPSLongitudeRef", "E")

        if not lat_tuple or not lon_tuple:
            return None, None

        def to_float(val):
            if isinstance(val, (int, float)):
                return float(val)
            if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                return float(val.numerator) / float(val.denominator) if val.denominator != 0 else 0.0
            if isinstance(val, tuple) and len(val) == 2:
                return float(val[0]) / float(val[1]) if val[1] != 0 else 0.0
            return float(val)

        lat_deg = to_float(lat_tuple[0]) + (to_float(lat_tuple[1]) / 60.0) + (to_float(lat_tuple[2]) / 3600.0)
        if lat_ref.upper() == "S":
            lat_deg = -lat_deg

        lon_deg = to_float(lon_tuple[0]) + (to_float(lon_tuple[1]) / 60.0) + (to_float(lon_tuple[2]) / 3600.0)
        if lon_ref.upper() == "W":
            lon_deg = -lon_deg

        return lat_deg, lon_deg
    except Exception as e:
        logger.debug(f"Error parsing GPS EXIF values: {e}")
        return None, None
