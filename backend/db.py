import os
import json
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "urbanpulse.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Creates data/ dir and runs the TASK 00.5 DDL. Idempotent."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id        TEXT    NOT NULL,
                event_type      TEXT    NOT NULL,   -- 'POTHOLE' | 'VEHICLE' | 'CONGESTION'
                bus_id          TEXT    NOT NULL,
                session_id      TEXT    NOT NULL,
                created_utc     TEXT    NOT NULL,
                video_timestamp REAL,
                lat             REAL,
                lon             REAL,
                lat_key         REAL,               -- ROUND(lat, 5)  <- dedup key
                lon_key         REAL,               -- ROUND(lon, 5)  <- dedup key
                speed_kmph      REAL,
                heading_deg     REAL,
                confidence      REAL,
                snapshot_url    TEXT,
                sighting_count  INTEGER NOT NULL DEFAULT 1,
                first_seen_utc  TEXT,
                last_seen_utc   TEXT,
                attrs_json      TEXT    NOT NULL DEFAULT '{}'
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup
                ON events (event_type, lat_key, lon_key);

            CREATE INDEX IF NOT EXISTS idx_type ON events (event_type);
        """)
        conn.commit()


def insert_event(event_dict: dict, bus_id: str, session_id: str) -> str:
    """
    Applies the TASK 00.5 dedup rule.
    event_dict is the output of PotholeEvent.to_dict().
    Returns 'inserted' or 'deduped'.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    event_id = event_dict.get("event_id", "")
    event_type = event_dict.get("event_type", "POTHOLE")
    video_timestamp = event_dict.get("timestamp")
    confidence = event_dict.get("confidence")

    location = event_dict.get("location")
    lat, lon, lat_key, lon_key, speed_kmph, heading_deg = None, None, None, None, None, None
    if isinstance(location, dict) and location.get("latitude") is not None and location.get("longitude") is not None:
        lat = float(location["latitude"])
        lon = float(location["longitude"])
        lat_key = round(lat, 5)
        lon_key = round(lon, 5)
        speed_kmph = location.get("speed_kmph")
        heading_deg = location.get("heading_deg")

    media = event_dict.get("media", {})
    snapshot_url = media.get("snapshot") if isinstance(media, dict) else None

    attrs = {
        "source": event_dict.get("source"),
        "media": event_dict.get("media"),
        "metadata": event_dict.get("metadata")
    }
    attrs_json = json.dumps(attrs)

    with _get_conn() as conn:
        cursor = conn.cursor()
        if lat_key is not None and lon_key is not None:
            cursor.execute(
                "SELECT id, sighting_count, confidence FROM events WHERE event_type = ? AND lat_key = ? AND lon_key = ?",
                (event_type, lat_key, lon_key)
            )
            existing = cursor.fetchone()
            if existing:
                new_count = existing["sighting_count"] + 1
                ex_conf = existing["confidence"] if existing["confidence"] is not None else 0.0
                new_conf = max(ex_conf, float(confidence) if confidence is not None else 0.0)
                cursor.execute(
                    "UPDATE events SET sighting_count = ?, last_seen_utc = ?, confidence = ? WHERE id = ?",
                    (new_count, now_utc, new_conf, existing["id"])
                )
                conn.commit()
                return "deduped"

        cursor.execute(
            """
            INSERT INTO events (
                event_id, event_type, bus_id, session_id, created_utc,
                video_timestamp, lat, lon, lat_key, lon_key,
                speed_kmph, heading_deg, confidence, snapshot_url,
                sighting_count, first_seen_utc, last_seen_utc, attrs_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                event_id, event_type, bus_id, session_id, now_utc,
                video_timestamp, lat, lon, lat_key, lon_key,
                speed_kmph, heading_deg, confidence, snapshot_url,
                now_utc, now_utc, attrs_json
            )
        )
        conn.commit()
        return "inserted"


def get_events(event_type: str = None, limit: int = 1000) -> list[dict]:
    """Rows as dicts, newest first, attrs_json parsed back into a dict."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        if event_type:
            cursor.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                (event_type, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (limit,)
            )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            item = dict(row)
            if item.get("attrs_json"):
                try:
                    item["attrs_json"] = json.loads(item["attrs_json"])
                except Exception:
                    pass
            results.append(item)
        return results


def get_stats() -> dict:
    """
    Returns aggregated stats across events.
    """
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT lat_key, lon_key FROM events WHERE lat_key IS NOT NULL AND lon_key IS NOT NULL)"
        )
        unique_locations = cursor.fetchone()[0]

        cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
        by_type_rows = cursor.fetchall()
        by_type = {row[0]: row[1] for row in by_type_rows}

        cursor.execute("SELECT COUNT(DISTINCT bus_id) FROM events")
        buses_reporting = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(sighting_count), 0) FROM events")
        total_sightings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM events WHERE sighting_count > 1")
        repeat_confirmed = cursor.fetchone()[0]

        return {
            "total_events": total_events,
            "unique_locations": unique_locations,
            "by_type": by_type,
            "buses_reporting": buses_reporting,
            "total_sightings": total_sightings,
            "repeat_confirmed": repeat_confirmed
        }


def reset_db() -> None:
    """Deletes all rows. Demo reset button."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events")
        conn.commit()
