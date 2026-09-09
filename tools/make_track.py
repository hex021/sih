import os
import sys
import json
import csv
import math
import argparse
import random
import cv2


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def interpolate_polyline(waypoints: list, target_dist: float, total_dist: float):
    if target_dist <= 0 or total_dist <= 0:
        p0, p1 = waypoints[0], waypoints[1]
        return p0[0], p0[1], initial_bearing(p0[0], p0[1], p1[0], p1[1])

    accum = 0.0
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        seg_dist = haversine_m(p0[0], p0[1], p1[0], p1[1])
        if accum + seg_dist >= target_dist:
            fraction = (target_dist - accum) / seg_dist if seg_dist > 0 else 0.0
            lat = p0[0] + fraction * (p1[0] - p0[0])
            lon = p0[1] + fraction * (p1[1] - p0[1])
            bearing = initial_bearing(p0[0], p0[1], p1[0], p1[1])
            return lat, lon, bearing
        accum += seg_dist

    last = waypoints[-1]
    prev = waypoints[-2]
    return last[0], last[1], initial_bearing(prev[0], prev[1], last[0], last[1])


def generate_track(video_path: str, route_path: str, out_csv_path: str, avg_speed: float):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(route_path):
        raise FileNotFoundError(f"Route file not found: {route_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if total_frames <= 0 or fps <= 0:
        total_frames = 300
        fps = 30.0

    duration_s = total_frames / fps

    with open(route_path, "r", encoding="utf-8") as f:
        waypoints = json.load(f)

    if len(waypoints) < 2:
        raise ValueError("Route must contain at least 2 waypoints")

    cum_dists = [0.0]
    for i in range(len(waypoints) - 1):
        d = haversine_m(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1])
        cum_dists.append(cum_dists[-1] + d)

    total_route_dist = cum_dists[-1]

    # Precalculate sine-modulated speed integration
    speed_factors = []
    for i in range(total_frames):
        t = i / max(1, total_frames - 1)
        # Vary speed multiplier between 0.4x and 1.3x of avg_speed
        m = 0.85 + 0.45 * math.sin(2.0 * math.pi * t * 1.5)
        speed_factors.append(m)

    sum_factors = sum(speed_factors)
    normalized_cum = []
    curr = 0.0
    for factor in speed_factors:
        curr += factor / sum_factors
        normalized_cum.append(curr)

    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    random.seed(42)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_s", "lat", "lon", "speed_kmph", "heading_deg"])

        for i in range(total_frames):
            frame_idx = i + 1
            timestamp_s = round(i / fps, 3)
            dist = normalized_cum[i] * total_route_dist

            lat_clean, lon_clean, heading_deg = interpolate_polyline(waypoints, dist, total_route_dist)

            # Add Gaussian jitter ≈ ±3m (≈ 0.000027 degrees)
            jitter_lat = random.gauss(0, 0.000027)
            jitter_lon = random.gauss(0, 0.000027)

            lat = round(lat_clean + jitter_lat, 6)
            lon = round(lon_clean + jitter_lon, 6)

            speed_kmph = round(avg_speed * speed_factors[i], 1)
            heading_deg = round(heading_deg, 1)

            writer.writerow([frame_idx, timestamp_s, f"{lat:.6f}", f"{lon:.6f}", speed_kmph, heading_deg])

    print(f"Generated track CSV: '{out_csv_path}' ({total_frames} frames, {duration_s:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic frame-by-frame GPS track from route polyline")
    parser.add_argument("--video", type=str, default="input/road_video.mp4", help="Input video file path")
    parser.add_argument("--route", type=str, default="tools/routes/sg_highway.json", help="Route polyline JSON file path")
    parser.add_argument("--out", type=str, default="input/track.csv", help="Output track CSV file path")
    parser.add_argument("--avg-speed", type=float, default=32.0, help="Average bus speed in km/h")

    args = parser.parse_args()
    generate_track(args.video, args.route, args.out, args.avg_speed)
