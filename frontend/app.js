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
    const inputPreviewImg = document.getElementById("input-preview-img");
    const removeFileBtn = document.getElementById("remove-file-btn");
    
    const processBtn = document.getElementById("process-btn");
    const cameraStatusBadge = document.getElementById("camera-status-badge");
    
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

    // Nav Switch Handling
    const navAnalyze = document.getElementById("nav-analyze");
    const navMap = document.getElementById("nav-map");
    const dashboardGrid = document.querySelector(".dashboard-grid");
    const mapView = document.getElementById("map-view");

    if (navAnalyze && navMap) {
        navAnalyze.addEventListener("click", () => {
            navAnalyze.classList.add("active");
            navMap.classList.remove("active");
            if (dashboardGrid) dashboardGrid.style.display = "grid";
            if (mapView) mapView.style.display = "none";
        });

        navMap.addEventListener("click", () => {
            navMap.classList.add("active");
            navAnalyze.classList.remove("active");
            if (dashboardGrid) dashboardGrid.style.display = "none";
            if (mapView) mapView.style.display = "grid";
            if (window.UrbanPulseMap && typeof window.UrbanPulseMap.load === "function") {
                window.UrbanPulseMap.load();
            }
        });
    }

    // 2. Global State Variables
    let selectedFile = null;
    let selectedFileType = "video"; // "video" or "image"
    let isProcessing = false;

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
                    if (deviceLabel) deviceLabel.textContent = data.device;
                    if (statusLabel) statusLabel.textContent = "AI Engine Ready";
                    if (statusDot) statusDot.style.backgroundColor = "var(--success)";
                    return;
                }
            }
            throw new Error("Invalid response");
        } catch (e) {
            if (deviceLabel) deviceLabel.textContent = "Unavailable";
            if (statusLabel) statusLabel.textContent = "API Service Offline";
            if (statusDot) statusDot.style.backgroundColor = "var(--danger)";
            console.error("Failed to connect to backend service:", e);
        }
    }

    // 5. Unified File Validation and Selection Helper
    function handleFileSelection(file) {
        if (!file || isProcessing) return;

        const imageExts = ["jpg", "jpeg", "png", "webp"];
        const videoExts = ["mp4", "mov", "avi", "mkv"];
        const allowedExtensions = [...imageExts, ...videoExts];

        const fileExt = file.name.split(".").pop().toLowerCase();
        
        if (!allowedExtensions.includes(fileExt)) {
            alert(`Unsupported file format: .${fileExt}\nPlease upload a Video (MP4, MOV, AVI, MKV) or Photo (JPG, PNG, WEBP).`);
            return;
        }

        selectedFile = file;
        const isImage = imageExts.includes(fileExt);
        selectedFileType = isImage ? "image" : "video";
        
        // Populate preview meta
        if (previewFilename) previewFilename.textContent = file.name;
        if (previewFilesize) previewFilesize.textContent = `${formatBytes(file.size)} • ${isImage ? "Photograph" : "Video"}`;

        // Load preview URL locally in browser
        const localUrl = URL.createObjectURL(file);

        if (isImage) {
            if (inputPreview) inputPreview.style.display = "none";
            if (inputPreviewImg) {
                inputPreviewImg.src = localUrl;
                inputPreviewImg.style.display = "block";
            }
        } else {
            if (inputPreviewImg) inputPreviewImg.style.display = "none";
            if (inputPreview) {
                inputPreview.src = localUrl;
                inputPreview.style.display = "block";
                inputPreview.load();
            }
        }

        // Camera status badge initial state
        if (cameraStatusBadge) {
            if (isImage) {
                cameraStatusBadge.textContent = "N/A (Still Photo)";
                cameraStatusBadge.style.background = "#F1F5F9";
                cameraStatusBadge.style.color = "#475569";
            } else {
                cameraStatusBadge.textContent = "Auto-Detecting on Upload";
                cameraStatusBadge.style.background = "#EFF6FF";
                cameraStatusBadge.style.color = "#1D4ED8";
            }
        }

        // Toggle visibility
        if (dropzone) dropzone.style.display = "none";
        if (previewZone) previewZone.style.display = "flex";
        
        // Enable processing trigger button
        if (processBtn) {
            processBtn.disabled = false;
            updateProcessButtonText();
        }
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
        if (fileInput) fileInput.value = "";
        
        // Reset preview video and image players
        if (inputPreview) {
            inputPreview.src = "";
            inputPreview.style.display = "none";
        }
        if (inputPreviewImg) {
            inputPreviewImg.src = "";
            inputPreviewImg.style.display = "none";
        }
        
        // Reset output player
        if (outputVideo) outputVideo.src = "";
        
        // Toggle view blocks
        if (dropzone) dropzone.style.display = "block";
        if (previewZone) previewZone.style.display = "none";
        
        // Restore output card defaults
        if (outputIdle) outputIdle.style.display = "flex";
        if (outputLoading) outputLoading.style.display = "none";
        if (outputPlayerContainer) outputPlayerContainer.style.display = "none";
        if (outputFooter) outputFooter.style.display = "none";
        
        // Restore statistics empty state
        if (statsEmpty) statsEmpty.style.display = "block";
        if (statsDashboard) statsDashboard.style.display = "none";
        
        if (cameraStatusBadge) {
            cameraStatusBadge.textContent = "Auto-Detecting on Upload";
            cameraStatusBadge.style.background = "#F1F5F9";
            cameraStatusBadge.style.color = "#475569";
        }

        // Re-enable trigger buttons
        if (processBtn) {
            processBtn.disabled = true;
            updateProcessButtonText();
        }
        if (removeFileBtn) removeFileBtn.style.display = "block";
    }

    // 7. Event Listeners for Upload / Input (Fixed Browse Files Click)
    if (browseBtn) {
        browseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (isProcessing) return;
            if (fileInput) fileInput.click();
        });
    }

    if (dropzone) {
        dropzone.addEventListener("click", (e) => {
            if (isProcessing) return;
            if (e.target !== browseBtn && (!browseBtn || !browseBtn.contains(e.target))) {
                if (fileInput) fileInput.click();
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
            if (files && files.length > 0) {
                handleFileSelection(files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (isProcessing) return;
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    if (removeFileBtn) {
        removeFileBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (isProcessing) return;
            resetUI();
        });
    }

    // 8. Processing Request Trigger (Unified Media Upload and Analysis Call)
    if (processBtn) {
        processBtn.addEventListener("click", async () => {
            if (!selectedFile || isProcessing) return;

            // Enter Processing State UI
            isProcessing = true;
            processBtn.disabled = true;
            processBtn.querySelector("span").textContent = "Processing...";
            if (removeFileBtn) removeFileBtn.style.display = "none";
            
            if (outputIdle) outputIdle.style.display = "none";
            if (outputPlayerContainer) outputPlayerContainer.style.display = "none";
            if (outputFooter) outputFooter.style.display = "none";
            if (outputLoading) outputLoading.style.display = "flex";

            if (cameraStatusBadge && selectedFileType === "video") {
                cameraStatusBadge.textContent = "Detecting Motion State...";
            }

            const busIdSelect = document.getElementById("bus-id");
            const formData = new FormData();
            formData.append("media", selectedFile);
            formData.append("video", selectedFile);
            formData.append("image", selectedFile);
            formData.append("mode", "unified");
            if (busIdSelect) {
                formData.append("bus_id", busIdSelect.value);
            }

            try {
                const response = await fetch("/api/analyze", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || "Failed to process media file");
                }

                const data = await response.json();
                if (data.success) {
                    // Update Camera Motion Badge Status Text from Backend Result
                    if (cameraStatusBadge) {
                        if (data.source_type === "image" || selectedFileType === "image") {
                            cameraStatusBadge.textContent = "N/A (Still Photo)";
                            cameraStatusBadge.style.background = "#F1F5F9";
                            cameraStatusBadge.style.color = "#475569";
                        } else if (data.camera_mode === "moving") {
                            cameraStatusBadge.textContent = "MOVING (Fleet Camera)";
                            cameraStatusBadge.style.background = "#FFEDD5";
                            cameraStatusBadge.style.color = "#C2410C";
                        } else {
                            cameraStatusBadge.textContent = "STATIONARY (Fixed CCTV)";
                            cameraStatusBadge.style.background = "#DBEAFE";
                            cameraStatusBadge.style.color = "#1D4ED8";
                        }
                    }

                    // Render output player / image viewer
                    if (data.source_type === "image" || selectedFileType === "image") {
                        const imgUrl = (data.image && data.image.output_url) ? data.image.output_url : data.video_url;
                        if (outputPlayerContainer) {
                            outputPlayerContainer.innerHTML = `<img src="${imgUrl}" style="max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border-color);" alt="Annotated Image Analysis">`;
                        }
                        if (downloadBtn) {
                            downloadBtn.href = imgUrl;
                            downloadBtn.download = "urbanpulse_image_analysis.jpg";
                            downloadBtn.innerHTML = `
                                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                                </svg>
                                Download Annotated Image
                            `;
                        }
                    } else {
                        if (outputPlayerContainer) {
                            outputPlayerContainer.innerHTML = `<video id="output-video" controls playsinline src="${data.video_url}"></video>`;
                        }
                        if (downloadBtn) {
                            downloadBtn.href = data.download_url;
                            downloadBtn.download = "urbanpulse_processed.mp4";
                            downloadBtn.innerHTML = `
                                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                                </svg>
                                Download Processed Video
                            `;
                        }
                    }
                    
                    if (outputLoading) outputLoading.style.display = "none";
                    if (outputPlayerContainer) outputPlayerContainer.style.display = "flex";
                    if (outputFooter) outputFooter.style.display = "block";

                    if (data.source_type === "image" || selectedFileType === "image") {
                        renderPhotoDashboard(data);
                    } else {
                        renderTelemetryDashboard(data);
                    }

                    if (window.UrbanPulseMap && typeof window.UrbanPulseMap.load === "function") {
                        window.UrbanPulseMap.load();
                    }
                } else {
                    throw new Error(data.error || "Analysis failed");
                }
            } catch (e) {
                console.error("Media Analysis Pipeline Error:", e);
                alert("Error during media processing: " + e.message);
                resetUI();
            }
        });
    }

    // 9. Telemetry Rendering Helper
    function renderTelemetryDashboard(data) {
        // Toggle empty stats placeholder off
        statsEmpty.style.display = "none";
        statsDashboard.style.display = "block";

        const kpiGps = document.getElementById("kpi-gps");
        if (kpiGps) kpiGps.textContent = "Simulated";

        if (data.bandwidth) {
            const videoBytes = data.bandwidth.video_bytes || 0;
            const eventsBytes = data.bandwidth.events_bytes || 0;
            const reductionPct = videoBytes > 0
                ? ((1 - eventsBytes / videoBytes) * 100).toFixed(2)
                : "0.00";
            const videoMB = (videoBytes / (1024 * 1024)).toFixed(1);
            const eventsKB = (eventsBytes / 1024).toFixed(1);

            const bandwidthText = `Bandwidth saved: ${reductionPct}% — ${videoMB} MB video vs ${eventsKB} KB events transmitted`;
            let bandwidthEl = document.getElementById("kpi-bandwidth");
            if (!bandwidthEl) {
                bandwidthEl = document.createElement("div");
                bandwidthEl.id = "kpi-bandwidth";
                bandwidthEl.style.cssText = "margin-top: 0.8rem; padding: 0.6rem 1rem; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4); color: #22c55e; border-radius: 6px; font-weight: 600; font-size: 0.9rem;";
                if (statsDashboard) statsDashboard.prepend(bandwidthEl);
            }
            bandwidthEl.textContent = bandwidthText;
        }

        const mode = data.mode || "unified";
        
        // Hide/Show sections based on mode
        const trafficSection = document.getElementById("traffic-stats-section");
        const roadSection = document.getElementById("road-stats-section");
        const vehicleSection = document.getElementById("vehicle-stats-section");
        const photoSection = document.getElementById("photo-stats-section");

        if (photoSection) photoSection.style.display = "none";
        
        // In unified / combined mode, show ALL sections simultaneously
        if (trafficSection) trafficSection.style.display = "block";
        if (roadSection) roadSection.style.display = "block";
        if (vehicleSection) vehicleSection.style.display = "block";

        // Update Camera Motion Badge Status Text
        const camStatusText = document.getElementById("camera-motion-status-text");
        const camModeVal = data.camera_mode || "stationary";
        if (camStatusText) {
            if (camModeVal === "moving") {
                camStatusText.textContent = "MOVING (Vehicle-Mounted Fleet Camera)";
                camStatusText.style.color = "#DD6B20";
            } else {
                camStatusText.textContent = "STATIONARY (Fixed Junction CCTV)";
                camStatusText.style.color = "#2B6CB0";
            }
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
        if (mode === "road" || mode === "combined" || mode === "unified") {
            const potholeCount = (data.summary && data.summary.total_potholes !== undefined)
                ? data.summary.total_potholes
                : ((data.summary && data.summary.potholes) || (data.potholes ? data.potholes.count : 0));
            
            const kpiPotholesEl = document.getElementById("kpi-potholes");
            if (kpiPotholesEl) kpiPotholesEl.textContent = potholeCount;
            
            const potholesTableBody = document.querySelector("#potholes-table tbody");
            if (potholesTableBody) {
                potholesTableBody.innerHTML = "";
                
                const events = data.events || (data.potholes ? data.potholes.events : []);
                if (events && events.length > 0) {
                    events.forEach(event => {
                        const tr = document.createElement("tr");
                        const sec = event.timestamp || 0.0;
                        const m = Math.floor(sec / 60);
                        const s = Math.floor(sec % 60);
                        const ms = Math.floor((sec % 1) * 100);
                        const formattedTime = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
                        
                        const snapshotUrl = (event.media && event.media.snapshot) ? event.media.snapshot : "#";

                        tr.innerHTML = `
                            <td><code>${event.event_id}</code></td>
                            <td><span class="category-pill" style="background-color: var(--danger-light); color: var(--danger);">POTHOLE</span></td>
                            <td>${formattedTime}</td>
                            <td>${Math.round(event.confidence * 100)}%</td>
                            <td><a href="${snapshotUrl}" target="_blank" class="btn btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.75rem; display: inline-flex; align-items: center; justify-content: center; height: auto;">View Snapshot</a></td>
                        `;
                        potholesTableBody.appendChild(tr);
                    });
                } else {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `<td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">No potholes detected in this video.</td>`;
                    potholesTableBody.appendChild(tr);
                }
            }
        }
        
        // --- Populate Phase 2.2 Vehicle Identification & ANPR Dashboard ---
        const vehicleStatsSection = document.getElementById("vehicle-stats-section");
        if (mode === "vehicle" || mode === "anpr" || mode === "unified" || mode === "combined") {
            if (vehicleStatsSection) vehicleStatsSection.style.display = "block";
            
            const summary = data.summary || {};
            const anprSummary = data.anpr ? data.anpr.summary : {};
            
            const totalVeh = summary.total_vehicles || anprSummary.total_vehicles || 0;
            const readPlates = summary.anpr_readable_plates !== undefined ? summary.anpr_readable_plates : (summary.plates_read || anprSummary.plates_read || 0);
            const unreadablePlates = (totalVeh - readPlates) >= 0 ? (totalVeh - readPlates) : (summary.unreadable_plates || 0);

            const elTotal = document.getElementById("kpi-anpr-total");
            const elRead = document.getElementById("kpi-anpr-read");
            const elUnread = document.getElementById("kpi-anpr-unreadable");
            const elUnique = document.getElementById("kpi-anpr-unique");

            if (elTotal) elTotal.textContent = totalVeh;
            if (elRead) elRead.textContent = readPlates;
            if (elUnread) elUnread.textContent = unreadablePlates;
            if (elUnique) elUnique.textContent = totalVeh;
            
            const vehiclesTableBody = document.querySelector("#vehicles-anpr-table tbody");
            if (vehiclesTableBody) {
                vehiclesTableBody.innerHTML = "";
                
                const records = data.vehicle_records || (data.anpr ? data.anpr.records : []);
                if (records && records.length > 0) {
                    records.forEach(rec => {
                        const tr = document.createElement("tr");
                        const sec = rec.timestamp || 0.0;
                        const m = Math.floor(sec / 60);
                        const s = Math.floor(sec % 60);
                        const formattedTime = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
                        
                        const regText = (rec.number_plate && rec.number_plate.text) ? rec.number_plate.text : rec.registration_number;
                        const isReadable = regText && regText !== "UNREADABLE";
                        const plateDisplay = isReadable
                            ? `<span class="plate-number">${regText}</span>`
                            : `<span style="color: var(--text-secondary); font-style: italic;">UNREADABLE</span>`;
                            
                        const vehType = (rec.vehicle && rec.vehicle.type) ? rec.vehicle.type : (rec.vehicle_type || "VEHICLE");
                        const trackId = (rec.vehicle && rec.vehicle.track_id) ? rec.vehicle.track_id : (rec.track_id || "-");
                        const vehConf = Math.round(((rec.vehicle && rec.vehicle.detection_confidence) ? rec.vehicle.detection_confidence : (rec.vehicle_confidence || 0.90)) * 100);
                        const plateConf = Math.round(((rec.number_plate && rec.number_plate.plate_detection_confidence) ? rec.number_plate.plate_detection_confidence : (rec.plate_detection_confidence || 0.85)) * 100);
                        const ocrConf = Math.round(((rec.number_plate && rec.number_plate.ocr_confidence) ? rec.number_plate.ocr_confidence : (rec.ocr_confidence || 0.0)) * 100);
                        
                        const vehSnap = (rec.media && rec.media.vehicle_snapshot) ? rec.media.vehicle_snapshot : (rec.vehicle_crop_url || "#");
                        const plateSnap = (rec.media && rec.media.plate_snapshot) ? rec.media.plate_snapshot : (rec.plate_crop_url || vehSnap);
                        
                        tr.innerHTML = `
                            <td><strong>#${trackId}</strong></td>
                            <td><span class="category-pill">${vehType}</span></td>
                            <td>${plateDisplay}</td>
                            <td>${vehConf}%</td>
                            <td>${plateConf}%</td>
                            <td>${ocrConf}%</td>
                            <td>${formattedTime}</td>
                            <td>
                                <button type="button" class="btn btn-secondary view-evidence-btn" 
                                    data-veh="${vehSnap}" data-plate="${plateSnap}" data-reg="${regText}" data-track="${trackId}"
                                    style="padding: 0.2rem 0.5rem; font-size: 0.75rem;">
                                    View Evidence
                                </button>
                            </td>
                        `;
                        vehiclesTableBody.appendChild(tr);
                    });
                } else {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `<td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">No vehicles identified.</td>`;
                    vehiclesTableBody.appendChild(tr);
                }
                
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
            }
        } else {
            if (vehicleStatsSection) vehicleStatsSection.style.display = "none";
        }
        
        // --- Populate Performance Telemetry Grid ---
        const perf = data.performance || {};
        const elInputRes = document.getElementById("perf-input-res");
        const elProcRes = document.getElementById("perf-process-res");
        const elSourceFps = document.getElementById("perf-source-fps");
        const elTotalFrames = document.getElementById("perf-total-frames");
        const elDevice = document.getElementById("perf-device");
        const elAvgFps = document.getElementById("perf-avg-fps");
        const elElapsed = document.getElementById("perf-elapsed");

        if (elInputRes) elInputRes.textContent = perf.resolution || data.input_resolution || "Auto (Adapted)";
        if (elProcRes) elProcRes.textContent = data.processing_resolution || "640x640 (YOLO)";
        if (elSourceFps) elSourceFps.textContent = data.source_fps ? `${data.source_fps} FPS` : "30.0 FPS";
        if (elTotalFrames) elTotalFrames.textContent = perf.total_frames || data.total_frames || "-";
        if (elDevice) elDevice.textContent = perf.device || data.device || "CPU";
        if (elAvgFps) elAvgFps.textContent = perf.avg_fps ? `${perf.avg_fps} FPS` : (data.avg_fps ? `${data.avg_fps} FPS` : "-");
        if (elElapsed) elElapsed.textContent = perf.elapsed_time ? `${perf.elapsed_time}s` : (data.elapsed_time ? `${data.elapsed_time}s` : "-");

        // Smoothly scroll to telemetry dashboard
        statsDashboard.scrollIntoView({ behavior: "smooth" });
        
        // Re-enable select controls
        updateProcessButtonText();
        removeFileBtn.style.display = "block";
        processBtn.disabled = false;
        isProcessing = false;
    }

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
        const kpiPotholesEl = document.getElementById("kpi-img-potholes");
        if (kpiPotholesEl) {
            kpiPotholesEl.textContent = summary.potholes_detected || 0;
        }

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
        updateProcessButtonText();
        processBtn.disabled = false;
        isProcessing = false;
    }

    // 11. Start Status Check on Page Load
    checkSystemStatus();

    function updateProcessButtonText() {
        if (isProcessing) return;
        if (processBtn && processBtn.querySelector("span")) {
            processBtn.querySelector("span").textContent = "Process Media";
        }
    }



    // 11. Modal Handlers for Evidence Viewer
    const incidentModal = document.getElementById("incident-modal");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const dismissModalBtn = document.getElementById("dismiss-modal-btn");

    if (closeModalBtn) {
        closeModalBtn.addEventListener("click", () => {
            if (incidentModal) incidentModal.style.display = "none";
        });
    }
    if (dismissModalBtn) {
        dismissModalBtn.addEventListener("click", () => {
            if (incidentModal) incidentModal.style.display = "none";
        });
    }
});

