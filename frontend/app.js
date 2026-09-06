/*
   UrbanPulse — Client-Side Application Logic
   Handles video drop/upload lifecycle, states, async API requests, and rendering.
*/

document.addEventListener("DOMContentLoaded", () => {
    // 1. DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    
    const previewZone = document.getElementById("preview-zone");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const inputPreview = document.getElementById("input-preview");
    const removeFileBtn = document.getElementById("remove-file-btn");
    
    const processBtn = document.getElementById("process-btn");
    const analysisMode = document.getElementById("analysis-mode");
    
    const outputIdle = document.getElementById("output-idle");
    const outputLoading = document.getElementById("output-loading");
    const outputPlayerContainer = document.getElementById("output-player-container");
    const outputVideo = document.getElementById("output-video");
    const downloadBtn = document.getElementById("download-btn");
    const outputFooter = document.querySelector(".output-card-footer");
    
    const statsEmpty = document.getElementById("stats-empty");
    const statsDashboard = document.getElementById("stats-dashboard");
    
    const kpiVehicles = document.getElementById("kpi-vehicles");
    const kpiPersons = document.getElementById("kpi-persons");
    const kpiDensity = document.getElementById("kpi-density");
    const densityCard = document.getElementById("density-card");
    const telemetryTableBody = document.querySelector("#telemetry-table tbody");
    
    const deviceLabel = document.getElementById("device-label");
    const statusLabel = document.querySelector(".status-label");
    const statusDot = document.querySelector(".status-dot");

    // 2. Global State Variables
    let selectedFile = null;
    let isProcessing = false;
    let inputSourceMode = "video"; // "video" or "image"

    const tabVideo = document.getElementById("tab-video");
    const tabImage = document.getElementById("tab-image");
    const dropzoneTitle = document.querySelector(".dropzone-title");
    const cardDesc = document.querySelector(".input-card .card-desc");

    if (tabVideo && tabImage) {
        tabVideo.addEventListener("click", () => {
            if (isProcessing) return;
            inputSourceMode = "video";
            tabVideo.classList.add("active");
            tabImage.classList.remove("active");
            tabVideo.style.background = "#FFFFFF";
            tabVideo.style.color = "#1E293B";
            tabImage.style.background = "transparent";
            tabImage.style.color = "#64748B";
            
            fileInput.accept = ".mp4,.mov,.avi,.mkv";
            cardDesc.textContent = "Supported formats: MP4, MOV, AVI, MKV";
            if (dropzoneTitle) dropzoneTitle.textContent = "Drag & drop traffic video here";
            resetUI();
        });

        tabImage.addEventListener("click", () => {
            if (isProcessing) return;
            inputSourceMode = "image";
            tabImage.classList.add("active");
            tabVideo.classList.remove("active");
            tabImage.style.background = "#FFFFFF";
            tabImage.style.color = "#1E293B";
            tabVideo.style.background = "transparent";
            tabVideo.style.color = "#64748B";
            
            fileInput.accept = ".jpg,.jpeg,.png,.webp";
            cardDesc.textContent = "Supported formats: JPG, JPEG, PNG, WEBP";
            if (dropzoneTitle) dropzoneTitle.textContent = "Drag & drop traffic photo here";
            resetUI();
        });
    }

    // 3. Class Names Formatting Helper

    const CLASS_METADATA = {
        "car": { display: "Cars", category: "Vehicle" },
        "motorcycle": { display: "Motorcycles", category: "Vehicle" },
        "bicycle": { display: "Bicycles", category: "Vehicle" },
        "bus": { display: "Buses", category: "Vehicle" },
        "truck": { display: "Trucks", category: "Vehicle" },
        "person": { display: "Persons", category: "Pedestrian" }
    };

    // 4. Initial System Check
    async function checkSystemStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                if (data.success) {
                    deviceLabel.textContent = data.device;
                    statusLabel.textContent = "AI Engine Ready";
                    statusDot.style.backgroundColor = "var(--success)";
                    return;
                }
            }
            throw new Error("Invalid response");
        } catch (e) {
            deviceLabel.textContent = "Unavailable";
            statusLabel.textContent = "API Service Offline";
            statusDot.style.backgroundColor = "var(--danger)";
            console.error("Failed to connect to backend service:", e);
        }
    }

    // Bind Camera Type change helper to update descriptive caption text
    const cameraTypeSelect = document.getElementById("camera-type");
    const cameraTypeHelp = document.getElementById("camera-type-help");
    if (cameraTypeSelect && cameraTypeHelp) {
        cameraTypeSelect.addEventListener("change", () => {
            if (cameraTypeSelect.value === "moving") {
                cameraTypeHelp.textContent = "Vehicle-mounted camera. Line crossing is disabled; total unique counts emphasized.";
            } else {
                cameraTypeHelp.textContent = "Fixed camera / CCTV / junction traffic flow monitoring";
            }
        });
    }

    // 5. File Validation and Selection Helper
    function handleFileSelection(file) {
        if (!file || isProcessing) return;

        const allowedExtensions = ["mp4", "mov", "avi", "mkv"];
        const fileExt = file.name.split(".").pop().toLowerCase();
        
        if (!allowedExtensions.includes(fileExt)) {
            alert(`Unsupported video format: .${fileExt}\nPlease upload an MP4, MOV, AVI, or MKV video.`);
            return;
        }

        selectedFile = file;
        
        // Populate preview meta
        previewFilename.textContent = file.name;
        previewFilesize.textContent = formatBytes(file.size);

        // Load preview URL locally in browser
        const localUrl = URL.createObjectURL(file);
        inputPreview.src = localUrl;
        inputPreview.load();

        // Toggle visibility
        dropzone.style.display = "none";
        previewZone.style.display = "flex";
        
        // Enable processing trigger button
        processBtn.disabled = false;
        updateProcessButtonText();
    }

    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // 6. Reset UI State Helper
    function resetUI() {
        if (isProcessing) return;
        selectedFile = null;
        fileInput.value = "";
        
        // Reset preview video player
        inputPreview.src = "";
        
        // Reset output player
        outputVideo.src = "";
        
        // Toggle view blocks
        dropzone.style.display = "block";
        previewZone.style.display = "none";
        processBtn.disabled = true;
        
        // Restore output card defaults
        outputIdle.style.display = "flex";
        outputLoading.style.display = "none";
        outputPlayerContainer.style.display = "none";
        outputFooter.style.display = "none";
        
        // Restore statistics empty state
        statsEmpty.style.display = "block";
        statsDashboard.style.display = "none";
        
        // Re-enable trigger buttons
        processBtn.disabled = true;
        updateProcessButtonText();
        removeFileBtn.style.display = "block";
    }

    // 7. Event Listeners for Upload / Input
    browseBtn.addEventListener("click", () => {
        if (isProcessing) return;
        fileInput.click();
    });
    
    fileInput.addEventListener("change", (e) => {
        if (isProcessing) return;
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag & Drop event bindings
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            if (isProcessing) return;
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            if (isProcessing) return;
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener("drop", (e) => {
        if (isProcessing) return;
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    removeFileBtn.addEventListener("click", () => {
        if (isProcessing) return;
        resetUI();
    });

    // 8. Processing Request Trigger (Video or Photo upload and analysis call)
    processBtn.addEventListener("click", async () => {
        if (!selectedFile || isProcessing) return;

        // Enter Processing State UI
        isProcessing = true;
        processBtn.disabled = true;
        processBtn.querySelector("span").textContent = "Processing...";
        removeFileBtn.style.display = "none"; // Block user from cancelling mid-run
        
        outputIdle.style.display = "none";
        outputPlayerContainer.style.display = "none";
        outputFooter.style.display = "none";
        outputLoading.style.display = "flex";
        
        // --- Photo / Image Analysis Path ---
        if (inputSourceMode === "image") {
            const formData = new FormData();
            formData.append("image", selectedFile);

            try {
                const response = await fetch("/api/analyze-image", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || "Failed to process image");
                }

                const data = await response.json();
                if (data.success) {
                    // Display annotated output image in player container
                    outputPlayerContainer.innerHTML = `<img src="${data.image.output_url}" style="max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border-color);" alt="Annotated Image Analysis">`;
                    downloadBtn.href = data.image.output_url;
                    downloadBtn.download = "urbanpulse_image_analysis.jpg";
                    downloadBtn.innerHTML = `
                        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                        </svg>
                        Download Annotated Image
                    `;
                    
                    outputLoading.style.display = "none";
                    outputPlayerContainer.style.display = "flex";
                    outputFooter.style.display = "block";

                    renderPhotoDashboard(data);
                } else {
                    throw new Error(data.error || "Image analysis failed");
                }
            } catch (e) {
                console.error("Image Analysis Pipeline Error:", e);
                alert("Error during image processing: " + e.message);
                resetUI();
            }
            return;
        }

        // --- Existing Video Analysis Path (Frozen & Intact) ---
        const mode = analysisMode.value;
        const cameraMode = document.getElementById("camera-type").value;
        const formData = new FormData();
        formData.append("video", selectedFile);
        formData.append("mode", mode);
        formData.append("camera_mode", cameraMode);

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Failed to process video");
            }

            const data = await response.json();
            if (data.success) {
                // Populate Output Player & Download URL
                outputPlayerContainer.innerHTML = `<video id="output-video" controls playsinline src="${data.video_url}"></video>`;
                downloadBtn.href = data.download_url;
                downloadBtn.download = "urbanpulse_processed.mp4";
                downloadBtn.innerHTML = `
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                    </svg>
                    Download Processed Video
                `;
                
                // Show completed player frame
                outputLoading.style.display = "none";
                outputPlayerContainer.style.display = "flex";
                outputFooter.style.display = "block";
                
                // Render statistics dashboard
                renderTelemetryDashboard(data);
            } else {
                throw new Error(data.error || "Analysis failed");
            }
        } catch (e) {

            console.error("Analysis Pipeline Error:", e);
            alert(`Analysis Failed:\n${e.message || "The video could not be processed."}`);
            
            // Revert state to selection
            isProcessing = false;
            outputLoading.style.display = "none";
            outputIdle.style.display = "flex";
            processBtn.disabled = false;
            updateProcessButtonText();
            removeFileBtn.style.display = "block";
        }
    });

    // 9. Telemetry Rendering Helper
    function renderTelemetryDashboard(data) {
        // Toggle empty stats placeholder off
        statsEmpty.style.display = "none";
        statsDashboard.style.display = "block";

        const mode = data.mode || "traffic";
        
        // Hide/Show sections based on mode
        const trafficSection = document.getElementById("traffic-stats-section");
        const roadSection = document.getElementById("road-stats-section");
        
        if (mode === "traffic") {
            trafficSection.style.display = "block";
            roadSection.style.display = "none";
        } else if (mode === "road") {
            trafficSection.style.display = "none";
            roadSection.style.display = "block";
        } else if (mode === "combined") {
            trafficSection.style.display = "block";
            roadSection.style.display = "block";
        }

        // --- Populate Traffic AI Dashboard ---
        if (mode === "traffic" || mode === "combined") {
            kpiVehicles.textContent = data.total_vehicles;
            kpiPersons.textContent = data.total_persons;
            
            const densityVal = data.traffic_density || "LOW";
            kpiDensity.textContent = densityVal;

            densityCard.className = "kpi-card";
            if (densityVal === "LOW") {
                densityCard.classList.add("density-low");
            } else if (densityVal === "MEDIUM") {
                densityCard.classList.add("density-medium");
            } else if (densityVal === "HIGH") {
                densityCard.classList.add("density-high");
            }

            // Update KPI Labels and Subtitles based on camera mode
            const cameraMode = data.camera_mode || "stationary";
            const kpiVehLabel = kpiVehicles.previousElementSibling;
            const kpiVehSubtitle = kpiVehicles.nextElementSibling;
            const kpiPersLabel = kpiPersons.previousElementSibling;
            const kpiPersSubtitle = kpiPersons.nextElementSibling;

            if (cameraMode === "moving") {
                kpiVehLabel.textContent = "Unique Vehicles Detected";
                kpiVehSubtitle.textContent = "Total unique vehicles observed";
                kpiPersLabel.textContent = "Unique Pedestrians Detected";
                kpiPersSubtitle.textContent = "Total unique pedestrians observed";
            } else {
                kpiVehLabel.textContent = "Total Vehicles Counted";
                kpiVehSubtitle.textContent = "Unique vehicles seen in video";
                kpiPersLabel.textContent = "Total Persons Detected";
                kpiPersSubtitle.textContent = "Unique persons seen in video";
            }

            telemetryTableBody.innerHTML = "";
            const stats = data.statistics || {};
            const displayOrder = ["car", "motorcycle", "bicycle", "bus", "truck", "person"];
            
            displayOrder.forEach(key => {
                const count = stats[key] || 0;
                const meta = CLASS_METADATA[key] || { display: key.toUpperCase(), category: "Other" };
                
                let crossingsText = "";
                if (cameraMode === "moving") {
                    crossingsText = `<span style="color: var(--text-secondary); font-style: italic;">N/A (Moving)</span>`;
                } else {
                    const crossings = (data.traffic && data.traffic.line_crossings && data.traffic.line_crossings.total && data.traffic.line_crossings.total[key]) || 0;
                    crossingsText = `<strong>${crossings}</strong>`;
                }

                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${meta.display}</strong></td>
                    <td>${count}</td>
                    <td>${crossingsText}</td>
                    <td><span class="category-pill">${meta.category}</span></td>
                `;
                telemetryTableBody.appendChild(tr);
            });
        }

        // --- Populate Road AI Dashboard ---
        if (mode === "road" || mode === "combined") {
            const potholeCount = (data.summary && data.summary.potholes) || 0;
            document.getElementById("kpi-potholes").textContent = potholeCount;
            
            const potholesTableBody = document.querySelector("#potholes-table tbody");
            potholesTableBody.innerHTML = "";
            
            if (data.events && data.events.length > 0) {
                data.events.forEach(event => {
                    const tr = document.createElement("tr");
                    const sec = event.timestamp || 0.0;
                    const m = Math.floor(sec / 60);
                    const s = Math.floor(sec % 60);
                    const ms = Math.floor((sec % 1) * 100);
                    const formattedTime = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
                    
                    tr.innerHTML = `
                        <td><code>${event.event_id}</code></td>
                        <td><span class="category-pill" style="background-color: var(--danger-light); color: var(--danger);">POTHOLE</span></td>
                        <td>${formattedTime}</td>
                        <td>${Math.round(event.confidence * 100)}%</td>
                        <td><a href="${event.media.snapshot}" target="_blank" class="btn btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.75rem; display: inline-flex; align-items: center; justify-content: center; height: auto;">View Snapshot</a></td>
                    `;
                    potholesTableBody.appendChild(tr);
                });
            } else {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">No potholes detected in this video.</td>`;
                potholesTableBody.appendChild(tr);
            }
        }
        
        // --- Populate Phase 2.2 Vehicle Identification & ANPR Dashboard ---
        const vehicleStatsSection = document.getElementById("vehicle-stats-section");
        if (mode === "vehicle" || mode === "anpr") {
            vehicleStatsSection.style.display = "block";
            
            const summary = data.summary || {};
            document.getElementById("kpi-anpr-total").textContent = summary.total_vehicles || 0;
            document.getElementById("kpi-anpr-read").textContent = summary.plates_read || 0;
            document.getElementById("kpi-anpr-unreadable").textContent = summary.unreadable_plates || 0;
            document.getElementById("kpi-anpr-unique").textContent = summary.unique_tracks || 0;
            
            const vehiclesTableBody = document.querySelector("#vehicles-anpr-table tbody");
            vehiclesTableBody.innerHTML = "";
            
            const records = data.vehicle_records || [];
            if (records.length > 0) {
                records.forEach(rec => {
                    const tr = document.createElement("tr");
                    const sec = rec.timestamp || 0.0;
                    const m = Math.floor(sec / 60);
                    const s = Math.floor(sec % 60);
                    const formattedTime = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
                    
                    const isReadable = rec.number_plate.text !== "UNREADABLE";
                    const plateDisplay = isReadable
                        ? `<span class="plate-number">${rec.number_plate.text}</span>`
                        : `<span style="color: var(--text-secondary); font-style: italic;">UNREADABLE</span>`;
                        
                    const vehConf = Math.round(rec.vehicle.detection_confidence * 100);
                    const plateConf = Math.round(rec.number_plate.plate_detection_confidence * 100);
                    const ocrConf = Math.round(rec.number_plate.ocr_confidence * 100);
                    
                    const vehSnap = rec.media.vehicle_snapshot || "#";
                    const plateSnap = rec.media.plate_snapshot || vehSnap;
                    
                    tr.innerHTML = `
                        <td><strong>#${rec.vehicle.track_id}</strong></td>
                        <td><span class="category-pill">${rec.vehicle.type}</span></td>
                        <td>${plateDisplay}</td>
                        <td>${vehConf}%</td>
                        <td>${plateConf}%</td>
                        <td>${ocrConf}%</td>
                        <td>${formattedTime}</td>
                        <td>
                            <button type="button" class="btn btn-secondary view-evidence-btn" 
                                data-veh="${vehSnap}" data-plate="${plateSnap}" data-reg="${rec.number_plate.text}" data-track="${rec.vehicle.track_id}"
                                style="padding: 0.2rem 0.5rem; font-size: 0.75rem;">
                                View Evidence
                            </button>
                        </td>
                    `;
                    vehiclesTableBody.appendChild(tr);
                });
                
                // Add event listeners for evidence viewer buttons
                document.querySelectorAll(".view-evidence-btn").forEach(btn => {
                    btn.addEventListener("click", (e) => {
                        const vehUrl = e.currentTarget.getAttribute("data-veh");
                        const plateUrl = e.currentTarget.getAttribute("data-plate");
                        const reg = e.currentTarget.getAttribute("data-reg");
                        const track = e.currentTarget.getAttribute("data-track");
                        
                        incidentModalContent.innerHTML = `
                            <div class="evidence-modal-body">
                                <h3>Vehicle Evidence Snapshot (Track #${track})</h3>
                                <p style="margin-bottom: 1rem;">Registration Plate: <strong class="plate-number">${reg}</strong></p>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                    <div>
                                        <h4 style="font-size: 0.85rem; margin-bottom: 0.5rem;">Vehicle Context Crop</h4>
                                        <img src="${vehUrl}" style="width: 100%; border-radius: 6px; border: 1px solid var(--border-color);" alt="Vehicle Crop">
                                    </div>
                                    <div>
                                        <h4 style="font-size: 0.85rem; margin-bottom: 0.5rem;">Number Plate Crop</h4>
                                        <img src="${plateUrl}" style="width: 100%; border-radius: 6px; border: 1px solid var(--border-color);" alt="Plate Crop">
                                    </div>
                                </div>
                            </div>
                        `;
                        incidentModal.style.display = "flex";
                    });
                });
            } else {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">No vehicles identified.</td>`;
                vehiclesTableBody.appendChild(tr);
            }
        } else {
            if (vehicleStatsSection) vehicleStatsSection.style.display = "none";
        }
        
        // Smoothly scroll to telemetry dashboard
        statsDashboard.scrollIntoView({ behavior: "smooth" });
        
        // Re-enable select controls
        updateProcessButtonText();
        removeFileBtn.style.display = "block";
        processBtn.disabled = false;
        isProcessing = false;

    // 10. Render Still Photo Analysis Dashboard
    function renderPhotoDashboard(data) {
        statsEmpty.style.display = "none";
        statsDashboard.style.display = "block";

        // Hide video-only dashboard sections
        document.getElementById("traffic-stats-section").style.display = "none";
        document.getElementById("road-stats-section").style.display = "none";
        const vehicleStatsSec = document.getElementById("vehicle-stats-section");
        if (vehicleStatsSec) vehicleStatsSec.style.display = "none";

        const photoSection = document.getElementById("photo-stats-section");
        photoSection.style.display = "block";

        const summary = data.summary || {};
        document.getElementById("kpi-img-vehicles").textContent = summary.total_vehicles || 0;
        document.getElementById("kpi-img-plates").textContent = summary.plates_detected || 0;
        document.getElementById("kpi-img-read").textContent = summary.plates_read || 0;
        document.getElementById("kpi-img-unreadable").textContent = summary.unreadable_plates || 0;

        // Genuine Metadata Banner
        const tsInfo = data.timestamp || {};
        const locInfo = data.location || {};

        const tsLabel = tsInfo.label === "Capture Time" ? `Capture Time: ${tsInfo.capture_time}` : `Analysis Time: ${tsInfo.analysis_time}`;
        document.getElementById("img-timestamp-val").textContent = tsLabel;

        const gpsLabel = locInfo.available ? `${locInfo.latitude}° N, ${locInfo.longitude}° E` : "Unavailable";
        document.getElementById("img-gps-val").textContent = gpsLabel;

        // Number Plate Log Table
        const photoTableBody = document.querySelector("#photo-plate-log-table tbody");
        photoTableBody.innerHTML = "";

        const detections = data.detections || [];
        if (detections.length > 0) {
            detections.forEach(det => {
                const tr = document.createElement("tr");
                const isReadable = det.plate.text !== "UNREADABLE";
                const plateDisplay = isReadable
                    ? `<span class="plate-number">${det.plate.text}</span>`
                    : `<span style="color: var(--text-secondary); font-style: italic;">UNREADABLE</span>`;

                const vehConf = Math.round(det.vehicle_confidence * 100);
                const plateConf = Math.round(det.plate.detection_confidence * 100);
                const ocrConf = det.plate.ocr_confidence ? Math.round(det.plate.ocr_confidence * 100) + "%" : "N/A";

                const vehCrop = det.evidence.vehicle_crop || "#";
                const plateCrop = det.evidence.plate_crop || vehCrop;

                tr.innerHTML = `
                    <td><strong>#${det.vehicle_index}</strong></td>
                    <td><span class="category-pill">${det.vehicle_type}</span></td>
                    <td>${plateDisplay}</td>
                    <td>${vehConf}%</td>
                    <td>${plateConf}%</td>
                    <td>${ocrConf}</td>
                    <td>
                        <button type="button" class="btn btn-secondary view-evidence-btn" 
                            data-veh="${vehCrop}" data-plate="${plateCrop}" data-reg="${det.plate.text}" data-track="${det.vehicle_index}"
                            style="padding: 0.2rem 0.5rem; font-size: 0.75rem;">
                            View Evidence
                        </button>
                    </td>
                `;
                photoTableBody.appendChild(tr);
            });

            // Evidence modal handlers
            photoTableBody.querySelectorAll(".view-evidence-btn").forEach(btn => {
                btn.addEventListener("click", (e) => {
                    const vehUrl = e.currentTarget.getAttribute("data-veh");
                    const plateUrl = e.currentTarget.getAttribute("data-plate");
                    const reg = e.currentTarget.getAttribute("data-reg");
                    const idx = e.currentTarget.getAttribute("data-track");

                    incidentModalContent.innerHTML = `
                        <div class="evidence-modal-body">
                            <h3>Vehicle Evidence Snapshot (Vehicle #${idx})</h3>
                            <p style="margin-bottom: 1rem;">Number Plate: <strong class="plate-number">${reg}</strong></p>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <h4 style="font-size: 0.85rem; margin-bottom: 0.5rem;">Vehicle Crop</h4>
                                    <img src="${vehUrl}" style="width: 100%; border-radius: 6px; border: 1px solid var(--border-color);" alt="Vehicle Crop">
                                </div>
                                <div>
                                    <h4 style="font-size: 0.85rem; margin-bottom: 0.5rem;">Number Plate Crop</h4>
                                    <img src="${plateUrl}" style="width: 100%; border-radius: 6px; border: 1px solid var(--border-color);" alt="Plate Crop">
                                </div>
                            </div>
                        </div>
                    `;
                    incidentModal.style.display = "flex";
                });
            });
        } else {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">No vehicles or number plates detected in this image.</td>`;
            photoTableBody.appendChild(tr);
        }

        // Performance Telemetry
        document.getElementById("perf-input-res").textContent = `${data.image.width}x${data.image.height}`;
        document.getElementById("perf-process-res").textContent = "Full Still Image";
        document.getElementById("perf-source-fps").textContent = "N/A (Still Image)";
        document.getElementById("perf-total-frames").textContent = "1 Frame";
        document.getElementById("perf-device").textContent = "CPU/YOLO";
        document.getElementById("perf-avg-fps").textContent = "N/A";
        document.getElementById("perf-elapsed").textContent = `${data.elapsed_time}s`;

        statsDashboard.scrollIntoView({ behavior: "smooth" });
        removeFileBtn.style.display = "block";
        processBtn.disabled = false;
        isProcessing = false;
    }

    // 11. Start Status Check on Page Load
    checkSystemStatus();

    // Mode dropdown change listener
    analysisMode.addEventListener("change", () => {
        updateProcessButtonText();
    });

    function updateProcessButtonText() {
        if (isProcessing) return;
        if (inputSourceMode === "image") {
            processBtn.querySelector("span").textContent = "Analyze Image";
            return;
        }
        const mode = analysisMode.value;
        if (mode === "traffic") {
            processBtn.querySelector("span").textContent = "Analyze Traffic";
        } else if (mode === "road") {
            processBtn.querySelector("span").textContent = "Analyze Road Condition";
        } else if (mode === "vehicle") {
            processBtn.querySelector("span").textContent = "Analyze Vehicle Identification & Plates";
        } else if (mode === "combined") {
            processBtn.querySelector("span").textContent = "Analyze Combined System";
        }
    }



    // 11. Incident Report Modal Handlers
    const incidentModal = document.getElementById("incident-modal");
    const incidentModalContent = document.getElementById("incident-modal-content");
    const viewReportModalBtn = document.getElementById("view-report-modal-btn");
    const triggerReportBtn = document.getElementById("trigger-report-btn");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const dismissModalBtn = document.getElementById("dismiss-modal-btn");

    async function openIncidentReportModal() {
        try {
            const res = await fetch("/api/incidents/rash-driving/sample");
            if (res.ok) {
                const data = await res.json();
                incidentModalContent.innerHTML = data.html_summary;
                incidentModal.style.display = "flex";
            } else {
                alert("Failed to load Incident Report from server.");
            }
        } catch (err) {
            console.error("Error fetching incident report:", err);
            alert("Error connecting to server for incident report.");
        }
    }

    if (viewReportModalBtn) {
        viewReportModalBtn.addEventListener("click", openIncidentReportModal);
    }
    if (triggerReportBtn) {
        triggerReportBtn.addEventListener("click", openIncidentReportModal);
    }
    if (closeModalBtn) {
        closeModalBtn.addEventListener("click", () => {
            incidentModal.style.display = "none";
        });
    }
    if (dismissModalBtn) {
        dismissModalBtn.addEventListener("click", () => {
            incidentModal.style.display = "none";
        });
    }
});

