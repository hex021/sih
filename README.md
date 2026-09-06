# Smart India Hackathon (SIH) — PS 26124 — Phase 1 AI Traffic Detection Pipeline

## 1. Project Context & Purpose
This repository implements Phase 1 of our project for **Smart India Hackathon Problem Statement ID 26124**: *AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet* (organized by **Bharat Electronics Limited**). 

The eventual goal of the platform is to transform public transport buses into mobile urban sensing units that detect road defects, traffic congestion, missing infrastructure, and safety issues. 

**Phase 1** focuses on building and validating a core, local **AI Traffic Detection Pipeline** that reads prerecorded road video, tracks vehicles using persistent IDs, calculates directional counts across virtual lines, determines traffic density in a Region of Interest (ROI), and exports an annotated output video.

---

## 2. Phase 1 Architecture

The pipeline processes input frame-by-frame with **exactly one YOLO inference/tracking call per frame** to prevent redundant execution:

```text
                         [ main.py ] (Orchestrator)
                             │
                             ▼
                        Video / Frame
                             │
                             ▼
                        [ tracker.py ]
                             │
                    YOLO model.track()
                             │
                             ▼
                      TrackedObject[]
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          [ roi.py ]   [ counter.py ]  [ density.py ]
              │              │              │
              │              ▼              │
              │         Crossing Events     │
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                     [ visualizer.py ] (HUD Dashboard Overlay)
                             │
                             ▼
                        output.mp4
```

---

## 3. Project Structure

```text
SIH_Urban_Intelligence/
│
├── phase1/
│   ├── __init__.py
│   ├── main.py          # Orchestrates video reading/writing, profiling, and loop
│   ├── config.py        # Centralized settings (thresholds, mappings, coordinates)
│   ├── models.py        # Decoupled dataclasses (Detection, TrackedObject)
│   ├── detector.py      # Independent still-image inference (testing/verification only)
│   ├── tracker.py       # YOLO tracking wrapper with bounded center history
│   ├── roi.py           # Relative ROI containment checks
│   ├── counter.py       # Direction-aware line crossing counting & state
│   ├── density.py       # Telemetry active vehicle density estimation
│   ├── visualizer.py    # Telemetry dashboard rendering overlay
│   └── utils.py         # Logger setup and directory helpers
│
├── input/               # Contains test.jpg & road_video.mp4 (locally downloaded)
├── output/              # Contains output videos (e.g. traffic_detection.mp4)
├── models/              # Contains weights (e.g. yolo11n.pt)
│
├── tests/
│   ├── run_tests.py     # Multi-level Unit and Integration test suite
│   └── download_assets.py # Separate downloader for sample assets
│
├── requirements.txt     # Pinned Python package dependencies
├── README.md            # Setup and execution guide
└── .gitignore
```

---

## 4. Setup & Installation

### Prerequisites
* Python 3.10 to 3.12 (Recommended: 3.12)
* Windows OS with PowerShell

### 1. Clone & Initialize Virtual Environment
Open PowerShell inside the workspace directory:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Download Model Weights & Sample Videos
The core pipeline is built to run entirely offline. Download the model weights, test image, and sample traffic video using our dedicated asset downloader:
```powershell
python tests/download_assets.py
```
This script downloads:
* Standard YOLO still-image test asset (`input/test.jpg`)
* Official YOLO11 nano model weights (`models/yolo11n.pt`, 5.3 MB)
* A high-density highway traffic surveillance clip (`input/road_video.mp4`, 335 frames)

---

## 5. Running the Pipeline

To run the pipeline on the sample video and write the results to `output/traffic_detection.mp4`:

```powershell
# Run in debug mode (shows trajectories, ROI zone, and execution stats)
python -m phase1.main --input input/road_video.mp4 --output output/traffic_detection.mp4 --debug

# Run in clean/production mode
python -m phase1.main --input input/road_video.mp4 --output output/traffic_detection.mp4
```

### CLI Options
* `--input`: Path to input video file (Default: `input/road_video.mp4`)
* `--output`: Path to write output video file (Default: `output/traffic_detection.mp4`)
* `--model`: Path to YOLO weights (Default: `models/yolo11n.pt`)
* `--confidence`: Minimum confidence threshold (Default: `0.25`)
* `--debug`: Renders processing FPS, frame numbers, trajectories, and the ROI region overlay
* `--display`: Launches an interactive OpenCV graphical window showing the pipeline frame-by-frame (press `q` to exit)

---

## 6. Pipeline Features

### Centralized Mappings (`config.py`)
All numeric class IDs are mapped immediately in `tracker.py` into internal standardized names: `CAR`, `MOTORCYCLE`, `BICYCLE`, `BUS`, `TRUCK`, `PERSON`. All downstream calculations use these names.

### Decoupled Data Models (`models.py`)
No raw Ultralytics objects are passed through the pipeline. Detections and Tracks are converted to clean, lightweight dataclasses (`Detection`, `TrackedObject`), allowing future sensor inputs (GPS/IMU) to plug in without modifications.

### Bounded Track History & OC/Stale Pruning (`tracker.py`)
Tracks are bounded to a maximum history length (`TRACK_HISTORY_LIMIT = 50`) to prevent memory growth. Tracks that are occluded or leave the frame are marked inactive, and their histories are deleted if they are not seen for 30 consecutive frames.

### Resolution-Independent ROI (`roi.py`)
The Region of Interest is defined using relative coordinates (`0.0` to `1.0`). Frame pixel bounds are calculated dynamically on each frame, supporting different resolutions (e.g. `960x540`, `1920x1080`) out-of-the-box.

### Direction-Aware Virtual Line Crossing (`counter.py`)
Counting uses a relative horizontal line (`COUNT_LINE_RELATIVE_Y`). The script compares `previous_center_y` and `current_center_y` to determine direction:
* **INCOMING**: Vehicle moves downwards across the line (`previous_center_y < line_y <= current_center_y`).
* **OUTGOING**: Vehicle moves upwards across the line (`previous_center_y > line_y >= current_center_y`).
Double counting and line jitter are prevented by registering track IDs inside directional state sets. Vehicles are counted by class, and `PERSON` counts are kept separate.

### Traffic Density Classification (`density.py`)
Density is defined as the number of *currently active tracked vehicles inside the ROI* (excluding pedestrians). It maps to `LOW`, `MEDIUM`, or `HIGH` categories using configurable thresholds:
* `LOW`: <= 5 vehicles inside ROI
* `MEDIUM`: 6 to 15 vehicles inside ROI
* `HIGH`: 16+ vehicles inside ROI

---

## 7. Testing Suite

The testing suite contains unit tests and integration tests:

```powershell
python tests/run_tests.py
```

### Level 1: Unit Tests
Do not require model weights or network access and run instantly. They test:
* BBox and track dataclass serialization
* Class name mappings
* ROI relative coordinate conversion & inclusion
* Counting logic (downward/upward crossings, direction, duplicate prevention, and separation of vehicles/persons)
* Traffic density classification thresholds

### Level 2: Integration Tests
Auto-detects if YOLO weights (`models/yolo11n.pt`) and images exist:
* Verifies `Tracker` initialization and execution on blank frames
* Verifies `Detector` initialization and still-image detection on `input/test.jpg`
* Verifies OpenCV video capture reader and writer streams

### Level 3: Manual Visual Validation
Done by running the pipeline with `--debug` or `--display` and playing the output video in a media player to inspect tracking boxes, persistent IDs, crossing increments, HUD stats, and density tags.

---

## 8. Limitations & Future Roadmap
* **ANPR / OCR**: Bounding box coordinates and crop regions are isolated, but Phase 1 does not execute OCR.
* **Complex Geometry**: The count line is currently horizontal and the ROI is rectangular. Future updates will support diagonal count lines and polygonal ROIs.
* **Hardware & Edge Integration**: Future phases will implement edge processing on gateway computers fed by ESP32-CAMs, IMUs, and GPS receivers.
* **Event Integration**: The metrics generated in `main.py` are structured to easily output JSON payloads for an event transmission engine in later phases.
