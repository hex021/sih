(function () {
    let map = null;
    let markersGroup = null;
    let heatLayer = null;
    let firstLoadFinished = false;

    function initMap() {
        if (typeof L === 'undefined') return;
        const mapContainer = document.getElementById('map');
        if (!mapContainer || map) return;

        map = L.map('map').setView([23.0225, 72.5714], 12);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        markersGroup = L.layerGroup().addTo(map);

        const heatmapToggle = document.getElementById('heatmap-toggle');
        if (heatmapToggle) {
            heatmapToggle.addEventListener('change', function () {
                if (!map || !heatLayer) return;
                if (this.checked) {
                    if (!map.hasLayer(heatLayer)) map.addLayer(heatLayer);
                } else {
                    if (map.hasLayer(heatLayer)) map.removeLayer(heatLayer);
                }
            });
        }
    }

    function getColor(type) {
        switch ((type || '').toUpperCase()) {
            case 'INCIDENT': return '#DC2626';
            case 'VEHICLE': return '#3B82F6';
            case 'CONGESTION': return '#EF4444';
            case 'POTHOLE':
            default:
                return '#F97316';
        }
    }

    function loadEvents() {
        if (typeof L === 'undefined') return;
        if (!map) {
            initMap();
        }
        if (!map) return;

        fetch('/api/events')
            .then(res => res.json())
            .then(data => {
                if (!data.success) return;

                if (markersGroup) markersGroup.clearLayers();
                if (heatLayer && map.hasLayer(heatLayer)) {
                    map.removeLayer(heatLayer);
                }

                const heatPoints = [];
                const latLons = [];
                const events = data.events || [];

                events.forEach(ev => {
                    if (ev.lat === null || ev.lon === null || ev.lat === undefined || ev.lon === undefined) {
                        return;
                    }

                    const lat = parseFloat(ev.lat);
                    const lon = parseFloat(ev.lon);
                    const sightingCount = parseInt(ev.sighting_count || 1, 10);
                    const confidence = (parseFloat(ev.confidence || 0) * 100).toFixed(0);
                    const color = getColor(ev.event_type);
                    const radius = 6 + Math.min(sightingCount, 8) * 1.5;

                    latLons.push([lat, lon]);
                    heatPoints.push([lat, lon, sightingCount]);

                    let attrs = ev.attrs_json || {};
                    if (typeof attrs === 'string') {
                        try { attrs = JSON.parse(attrs); } catch (e) { attrs = {}; }
                    }

                    let popupHtml = '';
                    if ((ev.event_type || '').toUpperCase() === 'INCIDENT') {
                        const reasonText = (attrs.reason || 'Safety Anomaly').replace('_', ' ').toUpperCase();
                        const trackId = attrs.track_id !== undefined ? `#${attrs.track_id}` : 'N/A';
                        const plateText = attrs.plate || 'UNREADABLE';
                        const measuredVal = attrs.measured_value !== undefined ? attrs.measured_value : '';

                        popupHtml = `<div style="font-family: sans-serif; color: #1e293b; min-width: 220px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="font-weight: bold; font-size: 1.05rem; color: #DC2626;">INCIDENT</span>
                                <span style="background: #FEE2E2; color: #991B1B; font-size: 0.72rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">${reasonText}</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 2px;">Track ID: <b>${trackId}</b></div>
                            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 4px;">Number Plate: <b style="background: #FEF3C7; color: #92400E; padding: 1px 6px; border-radius: 3px; font-family: monospace;">${plateText}</b></div>
                            ${measuredVal ? `<div style="font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">Anomaly Rate: <b>${measuredVal}</b></div>` : ''}`;

                        if (ev.snapshot_url) {
                            popupHtml += `<div style="margin: 6px 0;">
                                <img src="${ev.snapshot_url}" alt="Incident Vehicle Evidence" style="width: 220px; max-width: 100%; border-radius: 6px; display: block;" />
                            </div>`;
                        }

                        popupHtml += `<div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                            Location: ${lat.toFixed(6)}, ${lon.toFixed(6)}<br>
                            <i>GPS: simulated</i>
                        </div></div>`;
                    } else {
                        popupHtml = `<div style="font-family: sans-serif; color: #1e293b; min-width: 180px;">
                            <div style="font-weight: bold; font-size: 1.05rem; margin-bottom: 4px;">${ev.event_type || 'POTHOLE'}</div>
                            <div style="font-size: 0.85rem; color: #475569;">Confidence: <b>${confidence}%</b></div>`;

                        if (ev.snapshot_url) {
                            popupHtml += `<div style="margin: 6px 0;">
                                <img src="${ev.snapshot_url}" alt="Event Snapshot" style="width: 220px; max-width: 100%; border-radius: 6px; display: block;" />
                            </div>`;
                        }

                        popupHtml += `<div style="font-size: 0.85rem; margin-top: 4px;">Sightings: <b>${sightingCount}</b></div>`;
                        if (sightingCount > 1) {
                            popupHtml += `<div style="font-size: 0.85rem; color: #ea580c; font-weight: bold;">Confirmed by ${sightingCount} fleet passes</div>`;
                        }

                        popupHtml += `<div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">
                            Location: ${lat.toFixed(6)}, ${lon.toFixed(6)}<br>
                            <i>GPS: simulated</i>
                        </div></div>`;
                    }

                    const marker = L.circleMarker([lat, lon], {
                        radius: radius,
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.85,
                        weight: 2
                    }).bindPopup(popupHtml);

                    markersGroup.addLayer(marker);
                });

                if (window.L && L.heatLayer) {
                    heatLayer = L.heatLayer(heatPoints, { radius: 25, blur: 15, maxZoom: 17 });
                    const heatmapToggle = document.getElementById('heatmap-toggle');
                    if (heatmapToggle && heatmapToggle.checked) {
                        map.addLayer(heatLayer);
                    }
                }

                if (!firstLoadFinished && latLons.length > 0) {
                    map.fitBounds(L.latLngBounds(latLons), { padding: [30, 30] });
                    firstLoadFinished = true;
                }

                renderStats(data.stats);
            })
            .catch(err => console.error('Failed to load map events:', err));
    }

    function renderStats(stats) {
        const kpisContainer = document.getElementById('map-kpis');
        if (!kpisContainer || !stats) return;

        kpisContainer.innerHTML = `
            <div class="kpi-tile" style="flex: 1; background: rgba(30, 41, 59, 0.7); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Total Events</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #f8fafc;">${stats.total_events || 0}</div>
            </div>
            <div class="kpi-tile" style="flex: 1; background: rgba(30, 41, 59, 0.7); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Unique Locations</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #38bdf8;">${stats.unique_locations || 0}</div>
            </div>
            <div class="kpi-tile" style="flex: 1; background: rgba(30, 41, 59, 0.7); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Buses Reporting</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #a855f7;">${stats.buses_reporting || 0}</div>
            </div>
            <div class="kpi-tile" style="flex: 1; background: rgba(30, 41, 59, 0.7); padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Repeat Confirmed</div>
                <div style="font-size: 1.4rem; font-weight: bold; color: #f97316;">${stats.repeat_confirmed || 0}</div>
            </div>
        `;
    }

    document.addEventListener('DOMContentLoaded', function () {
        initMap();

        setInterval(() => {
            const mapView = document.getElementById('map-view');
            if (mapView && mapView.style.display !== 'none') {
                loadEvents();
            }
        }, 5000);
    });

    window.NagarNetraMap = {
        load: loadEvents
    };
    window.UrbanPulseMap = window.NagarNetraMap;
})();
