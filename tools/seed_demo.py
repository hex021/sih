import os
import sys
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend import db


def seed_demo(clear: bool = False) -> None:
    db.init_db()
    if clear:
        db.reset_db()
        print("Database cleared.")

    # 25 seed locations along SG Highway route (Ahmedabad)
    start_lat, start_lon = 23.0225, 72.5714
    end_lat, end_lon = 23.0389, 72.5601

    passes_pattern = [1, 3, 1, 2, 5, 1, 1, 6, 2, 1, 4, 1, 2, 1, 3, 1, 5, 2, 1, 1, 3, 1, 2, 4, 1]
    event_types = ["POTHOLE", "POTHOLE", "POTHOLE", "VEHICLE", "CONGESTION"]

    inserted_count = 0
    deduped_count = 0

    for i in range(25):
        t = i / 24.0
        lat = round(start_lat + t * (end_lat - start_lat), 6)
        lon = round(start_lon + t * (end_lon - start_lon), 6)

        num_passes = passes_pattern[i]
        event_type = event_types[i % len(event_types)]

        for p in range(num_passes):
            bus_num = ((i + p) % 3) + 1
            bus_id = f"BUS-0{bus_num}"
            session_id = f"sess_{bus_id.lower()}_{i:02d}"

            # Vary confidence between 0.45 and 0.92
            conf = round(0.45 + ((i * 3 + p * 7) % 48) / 100.0, 2)
            if conf > 0.92:
                conf = 0.92

            speed = round(28.0 + ((i + p) % 6) * 2.5, 1)
            heading = round(310.0 + (i % 4) * 2.0, 1)

            location_obj = {
                "latitude": lat,
                "longitude": lon,
                "speed_kmph": speed,
                "heading_deg": heading,
                "source": "simulated"
            }

            event_dict = {
                "event_id": f"EVT-SEED-{i+1:03d}",
                "event_type": event_type,
                "timestamp": round(i * 4.2 + p * 0.5, 2),
                "confidence": conf,
                "location": location_obj,
                "source": {"camera": "dashcam_seed", "vehicle_id": bus_id},
                "media": {"frame": (i + 1) * 30, "snapshot": None, "video": None},
                "sensor_data": {"gps": location_obj, "imu": None},
                "metadata": {"seeded": True}
            }

            outcome = db.insert_event(event_dict, bus_id, session_id)
            if outcome == "inserted":
                inserted_count += 1
            else:
                deduped_count += 1

    stats = db.get_stats()
    print(f"Seeding finished. Inserted: {inserted_count}, Deduped passes: {deduped_count}")
    print(f"Current DB Stats: {stats}")


if __name__ == "__main__":
    should_clear = "--clear" in sys.argv
    seed_demo(clear=should_clear)
