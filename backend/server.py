import os
import sys
import uuid
import time
import subprocess
import logging
from flask import Flask, request, jsonify, send_from_directory

# Programmatically add workspace root to Python path to resolve phase1 module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1.main import run_pipeline
from phase1 import utils
from phase2.road_ai.pothole_pipeline import run_road_pipeline
from phase2.road_ai.combined_pipeline import run_combined_pipeline
from phase2.anpr.anpr_pipeline import run_anpr_pipeline
from phase2.image_ai import (
    ImageProcessor,
    ALLOWED_IMAGE_EXTENSIONS,
    IMAGE_PROCESSED_FOLDER,
    PLATE_EVIDENCE_FOLDER
)
from phase2.road_ai.incident_generator import (


    create_rash_driving_event,
    generate_incident_report,
    get_sample_track27_incident_report,
    get_incident_report_by_id
)


# Initialize directories
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
PROCESSED_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processed"))
FRONTEND_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize logger
logger = utils.setup_logger("SIH_Server")

# Initialize Flask app
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path="")

# Allowed video extensions
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """
    Deletes files in uploads/ and processed/ directories that are older than 30 minutes
    to prevent server storage overflow.
    """
    now = time.time()
    max_age_seconds = 1800  # 30 minutes
    
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                # Check modification time
                if (now - os.path.getmtime(file_path)) > max_age_seconds:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {f}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old file {file_path}: {e}")

@app.route("/")
def index():
    """Serves the frontend homepage."""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serves uploaded video files (primarily for preview)."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/processed/<filename>")
def serve_processed(filename):
    """Serves processed output video files."""
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route("/processed/events/pothole/<filename>")
def serve_pothole_event_snapshot(filename):
    """Serves pothole snapshot images."""
    pothole_events_dir = os.path.join(PROCESSED_FOLDER, "events", "pothole")
    return send_from_directory(pothole_events_dir, filename)

@app.route("/processed/events/vehicle/<event_id>/<filename>")
def serve_vehicle_event_snapshot(event_id, filename):
    """Serves Phase 2.2 vehicle and number plate evidence snapshot images."""
    vehicle_event_dir = os.path.join(PROCESSED_FOLDER, "events", "vehicle", event_id)
    return send_from_directory(vehicle_event_dir, filename)

@app.route("/processed/images/<filename>")
def serve_processed_image(filename):
    """Serves annotated output image files."""
    return send_from_directory(IMAGE_PROCESSED_FOLDER, filename)

@app.route("/processed/events/number_plate/<filename>")
def serve_number_plate_evidence_snapshot(filename):
    """Serves number plate and vehicle crop evidence snapshot images for still photos."""
    return send_from_directory(PLATE_EVIDENCE_FOLDER, filename)



@app.route("/api/download/<filename>")
def download_file(filename):
    """
    Serves the processed video as a secure download attachment.
    Forces Content-Disposition: attachment to prevent operating system
    playback errors and support clean downloads.
    """
    secure_name = os.path.basename(filename)
    file_path = os.path.join(PROCESSED_FOLDER, secure_name)
    if not os.path.exists(file_path):
        logger.error(f"Download request failed: file not found {file_path}")
        return jsonify({"success": False, "error": "Processed file not found"}), 404
        
    logger.info(f"Serving secure download for: {secure_name}")
    return send_from_directory(
        PROCESSED_FOLDER,
        secure_name,
        as_attachment=True,
        download_name="urbanpulse_processed.mp4",
        mimetype="video/mp4"
    )

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns the backend system status and GPU/CPU hardware device type."""
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return jsonify({
        "success": True,
        "device": device,
        "status": "ready"
    })

@app.route("/api/incidents/rash-driving/sample", methods=["GET"])
def get_sample_incident_report():
    """Returns sample Incident Report for Vehicle Track #27 & License Plate GJ01AB1234."""
    report = get_sample_track27_incident_report()
    return jsonify({
        "success": True,
        "report": report.to_dict(),
        "html_summary": report.generate_html_summary()
    })

@app.route("/api/incidents/rash-driving", methods=["POST"])
def log_rash_driving_incident():
    """
    Endpoint to create and issue a Rash Driving incident report from vehicle telemetry.
    Expected JSON payload (optional overrides): track_id, license_plate, timestamp, speed_kmh, etc.
    """
    data = request.get_json(silent=True) or {}
    event = create_rash_driving_event(
        track_id=data.get("track_id", 27),
        license_plate=data.get("license_plate", "GJ01AB1234"),
        timestamp=data.get("timestamp", 14.25),
        speed_kmh=data.get("speed_kmh", 84.5),
        speed_limit_kmh=data.get("speed_limit_kmh", 50.0)
    )
    report = generate_incident_report(event)
    return jsonify({
        "success": True,
        "report": report.to_dict(),
        "html_summary": report.generate_html_summary()
    }), 201

@app.route("/api/incidents/<report_id>", methods=["GET"])
def get_incident_report(report_id):
    """Retrieves generated incident report by ID."""
    report = get_incident_report_by_id(report_id)
    if not report:
        return jsonify({"success": False, "error": f"Report '{report_id}' not found"}), 404
    return jsonify({
        "success": True,
        "report": report.to_dict(),
        "html_summary": report.generate_html_summary()
    })

@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    """
    Dedicated endpoint to analyze an uploaded photograph/image file (.jpg, .jpeg, .png, .webp).
    Runs vehicle detection, plate localization, real OCR text extraction, EXIF metadata extraction,
    and returns structured JSON response with evidence crops and annotated output image.
    """
    cleanup_old_files()

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename provided"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({"success": False, "error": f"Unsupported image format '{ext}'. Allowed formats: JPG, JPEG, PNG, WEBP"}), 400

    file_id = str(uuid.uuid4())[:8]
    input_filename = f"{file_id}_image_input{ext}"
    output_filename = f"{file_id}_urbanpulse_image_analysis.jpg"

    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(IMAGE_PROCESSED_FOLDER, output_filename)

    try:
        logger.info(f"Saving uploaded image to {input_path}")
        file.save(input_path)

        processor = ImageProcessor()
        result = processor.process_image(input_path, output_path)

        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error", "Image analysis failed")}), 500

        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to process image upload: {e}")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500



@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    Endpoint to receive an uploaded traffic video, process it using the existing
    YOLO tracking pipeline, convert the output to H.264 format, and return telemetry stats.
    """
    # Run routine file cleanup
    cleanup_old_files()
    
    # 1. Validate request
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
        
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename provided"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported video format. Allowed formats: MP4, MOV, AVI, MKV"}), 400
        
    # 2. Generate unique filenames
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower()
    
    input_filename = f"{file_id}_input{ext}"
    raw_output_filename = f"{file_id}_raw_output.mp4"
    web_output_filename = f"{file_id}_output.mp4"
    
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    raw_output_path = os.path.join(PROCESSED_FOLDER, raw_output_filename)
    web_output_path = os.path.join(PROCESSED_FOLDER, web_output_filename)
    
    try:
        # 3. Save uploaded file
        logger.info(f"Saving uploaded video to {input_path}")
        file.save(input_path)
        
        # 4. Determine mode and execute corresponding pipeline
        mode = request.form.get("mode", "traffic")
        camera_mode = request.form.get("camera_mode", "stationary")
        logger.info(f"Analysis mode requested: {mode} | Camera mode: {camera_mode}")
        
        if mode == "road":
            result = run_road_pipeline(
                input_path=input_path,
                output_path=raw_output_path,
                debug=True,
                camera_mode=camera_mode
            )
        elif mode in ["vehicle", "anpr"]:
            result = run_anpr_pipeline(
                input_path=input_path,
                output_path=raw_output_path,
                debug=True,
                camera_mode=camera_mode
            )
        elif mode == "combined":
            result = run_combined_pipeline(
                input_path=input_path,
                output_path=raw_output_path,
                debug=True,
                camera_mode=camera_mode
            )
        else:
            result = run_pipeline(
                input_path=input_path,
                output_path=raw_output_path,
                confidence=0.25,
                debug=True,
                display=False,
                camera_mode=camera_mode
            )
            
        if not result.get("success"):
            return jsonify({"success": False, "error": f"{mode.capitalize()} detection pipeline failed"}), 500
            
        # 5. Convert raw OpenCV video to H.264 for native browser playback
        logger.info("Converting output video to web-playable format using FFmpeg...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", raw_output_path,
            "-vcodec", "libx264",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            web_output_path
        ]
        
        # Execute conversion
        conversion_result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if conversion_result.returncode != 0:
            logger.warning("FFmpeg conversion failed. Falling back to raw OpenCV video.")
            # Fallback to raw output if FFmpeg is unavailable/fails
            os.rename(raw_output_path, web_output_path)
        else:
            # Clean up the raw unconverted OpenCV video
            if os.path.exists(raw_output_path):
                os.remove(raw_output_path)
            logger.info("Video conversion successful.")
            
        # 6. Format and return response based on mode
        response_data = {
            "success": True,
            "mode": mode,
            "camera_mode": camera_mode,
            "video_url": f"/processed/{web_output_filename}",
            "download_url": f"/api/download/{web_output_filename}",
            "avg_fps": round(result["avg_fps"], 2),
            "elapsed_time": round(result["elapsed_time"], 2),
            # Performance Telemetry
            "input_resolution": result.get("input_resolution"),
            "processing_resolution": result.get("processing_resolution"),
            "source_fps": result.get("source_fps"),
            "total_frames": result.get("total_frames"),
            "device": result.get("device"),
            "inference_size": result.get("inference_size")
        }
        
        # Add mode-specific statistics
        if mode in ["vehicle", "anpr"]:
            response_data["vehicle_records"] = result["vehicle_records"]
            response_data["summary"] = result["summary"]
        elif mode == "road":
            response_data["events"] = [event.to_dict() for event in result["events"]]
            response_data["summary"] = {
                "potholes": result["total_potholes"]
            }
        elif mode == "combined":
            response_data["traffic"] = result["traffic"]
            response_data["statistics"] = result["statistics"]
            response_data["total_vehicles"] = result["total_vehicles"]
            response_data["total_persons"] = result["total_persons"]
            response_data["traffic_density"] = result["traffic_density"]
            response_data["events"] = [event.to_dict() for event in result["events"]]
            response_data["summary"] = {
                "potholes": result["total_potholes"]
            }
        else: # traffic
            response_data["traffic"] = result["traffic"]
            response_data["statistics"] = result["statistics"]
            response_data["total_vehicles"] = result["total_vehicles"]
            response_data["total_persons"] = result["total_persons"]
            response_data["traffic_density"] = result["traffic_density"]

            
        logger.info(f"Processing complete. Mode={mode}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Failed to process video upload: {e}")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("Starting SIH Server on http://localhost:5000")
    
    # Optional auto-open browser in Brave for development ease
    import threading
    def open_browser():
        time.sleep(1.5) # Wait for Flask to boot
        brave_paths = [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe"
        ]
        opened = False
        for path in brave_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"Automatically launching Brave browser from {path}")
                    subprocess.Popen([path, "http://localhost:5000"])
                    opened = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to launch Brave from {path}: {e}")
        if not opened:
            import webbrowser
            try:
                logger.info("Brave not found at default path. Launching default system browser...")
                webbrowser.open("http://localhost:5000")
            except Exception as e:
                logger.warning(f"Failed to open default browser: {e}")
                
    if os.environ.get("OPEN_BROWSER") == "1":
        threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
