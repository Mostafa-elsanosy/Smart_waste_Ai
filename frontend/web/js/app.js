/**
 * Smart Waste AI & DSS - Interactive Client Application
 */

const API_BASE = "";

// State
let state = {
  containers: [],
  statistics: {},
  alerts: [],
  activeDetection: null,
};

// DOM Elements
const kpiTotalItems = document.getElementById("kpi-total-items");
const kpiAccuracy = document.getElementById("kpi-accuracy");
const kpiWeight = document.getElementById("kpi-weight");
const kpiCarbon = document.getElementById("kpi-carbon");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const dropZonePlaceholder = document.getElementById("drop-zone-placeholder");
const classifiedImagePreview = document.getElementById("classified-image-preview");

const detectionDetailsBar = document.getElementById("detection-details");
const materialBreakdownBar = document.getElementById("material-breakdown-bar");
const mbGrid = document.getElementById("mb-grid");
const resCategory = document.getElementById("res-category");
const resConfidence = document.getElementById("res-confidence");
const resVolume = document.getElementById("res-volume");
const resLatency = document.getElementById("res-latency");

const dssEmptyState = document.getElementById("dss-empty-state");
const dssDecisionContainer = document.getElementById("dss-decision-container");
const decBinName = document.getElementById("dec-bin-name");
const decBinId = document.getElementById("dec-bin-id");
const decGateAction = document.getElementById("dec-gate-action");
const decMechanism = document.getElementById("dec-mechanism");
const decConveyorSpeed = document.getElementById("dec-conveyor-speed");
const decPurity = document.getElementById("dec-purity");
const dssReasoningList = document.getElementById("dss-reasoning-list");
const decisionStatusTag = document.getElementById("decision-status-tag");

const containersGrid = document.getElementById("containers-grid");
const alertsContainer = document.getElementById("alerts-container");
const alertsContent = document.getElementById("alerts-content");

const btnSimulateTick = document.getElementById("btn-simulate-tick");
const btnEmptyAll = document.getElementById("btn-empty-all");
const btnOptimizeRoute = document.getElementById("btn-optimize-route");
const sampleButtons = document.querySelectorAll(".btn-sample");

// Init
document.addEventListener("DOMContentLoaded", () => {
  setupUploadListeners();
  setupSamplePicker();
  setupActionButtons();
  refreshAllData();
  // Poll statistics & containers every 6 seconds
  setInterval(refreshAllData, 6000);
});

// Setup File Upload & Drag-and-Drop
function setupUploadListeners() {
  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      sampleButtons.forEach((b) => b.classList.remove("active"));
      uploadAndClassifyFile(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      sampleButtons.forEach((b) => b.classList.remove("active"));
      uploadAndClassifyFile(e.dataTransfer.files[0]);
    }
  });
}

// Setup Quick Sample Buttons
function setupSamplePicker() {
  sampleButtons.forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      sampleButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const filename = btn.getAttribute("data-sample");
      const hint = btn.getAttribute("data-type");
      try {
        const response = await fetch(`/api/sample-images/${filename}`);
        if (!response.ok) throw new Error("Sample fetch failed");
        const blob = await response.blob();
        const file = new File([blob], filename, { type: "image/jpeg" });
        uploadAndClassifyFile(file, hint);
      } catch (err) {
        console.error("Failed to load sample image:", err);
      }
    });
  });
}

// Upload & Classify API Call
async function uploadAndClassifyFile(file, categoryHint = null) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source", "gallery");
  if (categoryHint) {
    formData.append("category_hint", categoryHint);
  }

  // Visual loading feedback
  dropZonePlaceholder.innerHTML = `<p class="drop-prompt">Analyzing with YOLOv8 &amp; DSS...</p>`;

  try {
    const res = await fetch("/api/waste/classify", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Classification failed: " + res.statusText);
    const data = await res.json();
    renderClassificationResult(data);
    refreshAllData();
  } catch (err) {
    console.error("Upload error:", err);
    dropZonePlaceholder.innerHTML = `<p class="drop-prompt text-amber">Error analyzing image.</p><span class="drop-hint">Click to try again.</span>`;
  }
}

// Render YOLO Detection & DSS Decision
function renderClassificationResult(data) {
  const det = data.detection;
  const decisions = data.sorting_decisions;

  // 1. Annotated Image
  if (det.annotated_image_base64) {
    classifiedImagePreview.src = det.annotated_image_base64;
    classifiedImagePreview.classList.remove("hidden");
    dropZonePlaceholder.classList.add("hidden");
  }

  // 2. Detection Bar
  detectionDetailsBar.classList.remove("hidden");
  resCategory.textContent = det.primary_category;
  resConfidence.textContent = (det.confidence_avg * 100).toFixed(1) + "%";
  resLatency.textContent = det.inference_time_ms.toFixed(1) + " ms";

  const totalVol = det.detections.reduce((acc, d) => acc + (d.estimated_volume_liters || 0.5), 0);
  resVolume.textContent = totalVol.toFixed(2) + " L";

  // Category Colors
  const colors = {
    Metal: "text-amber",
    Plastic: "badge-category",
    Glass: "text-cyan",
    Paper: "text-emerald",
  };
  resCategory.className = `detail-val badge-category ${colors[det.primary_category] || ""}`;
 
  // 3. Material Probabilities Breakdown
  if (det.probabilities && materialBreakdownBar && mbGrid) {
    mbGrid.innerHTML = "";
    const colorMap = {
      Metal: "#f59e0b",
      Plastic: "#3b82f6",
      Glass: "#06b6d4",
      Paper: "#10b981",
    };
    for (const [mat, pct] of Object.entries(det.probabilities)) {
      const item = document.createElement("div");
      item.className = "mb-item";
      item.innerHTML = `
        <div class="mb-info">
          <span class="mb-name">${mat}</span>
          <span class="mb-val">${pct.toFixed(1)}%</span>
        </div>
        <div class="mb-track">
          <div class="mb-fill" style="width: ${Math.min(Math.max(pct, 0), 100)}%; background: ${colorMap[mat] || '#3b82f6'}; height: 100%; border-radius: 999px;"></div>
        </div>
      `;
      mbGrid.appendChild(item);
    }
    materialBreakdownBar.classList.remove("hidden");
  }

  // 4. DSS Sorting Decision
  if (decisions && decisions.length > 0) {
    const dec = decisions[0];
    dssEmptyState.classList.add("hidden");
    dssDecisionContainer.classList.remove("hidden");

    decBinName.textContent = dec.target_bin_name;
    decBinId.textContent = dec.target_bin_id;
    decGateAction.textContent = dec.gate_action;
    decMechanism.textContent = dec.actuation_mechanism;
    decConveyorSpeed.textContent = dec.conveyor_speed_mps.toFixed(1) + " m/s";

    if (dec.status === "ACCEPTED") {
      decisionStatusTag.textContent = "Accepted";
      decisionStatusTag.className = "tag tag-emerald";
      decPurity.textContent = `Purity Verified (${(dec.confidence * 100).toFixed(1)}%)`;
      decPurity.className = "spec-val text-emerald";
    } else if (dec.status === "REROUTED_OVERFLOW") {
      decisionStatusTag.textContent = "Rerouted (Overflow)";
      decisionStatusTag.className = "tag";
      decisionStatusTag.style.background = "rgba(139, 92, 246, 0.25)";
      decisionStatusTag.style.color = "#c084fc";
      decPurity.textContent = "Overflow Bypass Active";
      decPurity.className = "spec-val text-amber";
    } else {
      decisionStatusTag.textContent = "Inspection Flag";
      decisionStatusTag.className = "tag";
      decisionStatusTag.style.background = "rgba(239, 68, 68, 0.2)";
      decisionStatusTag.style.color = "#f87171";
      decPurity.textContent = "Purity Risk - Review";
      decPurity.className = "spec-val text-amber";
    }

    // Reasoning Audit Trail
    dssReasoningList.innerHTML = "";
    dec.decision_reasoning.forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      dssReasoningList.appendChild(li);
    });
  }
}

// Refresh KPI and Containers Data with background throttling & health indicator
async function refreshAllData() {
  if (document.hidden) return; // Skip polling when tab is not active
  try {
    const [statsRes, containersRes] = await Promise.all([
      fetch("/api/statistics"),
      fetch("/api/containers"),
    ]);

    const statusText = document.getElementById("system-status-text");
    const statusDot = document.querySelector(".status-dot");

    if (statsRes.ok) {
      const stats = await statsRes.json();
      state.statistics = stats;
      renderKPIs(stats);
      if (statusText) statusText.textContent = "DSS Engine Online";
      if (statusDot) {
        statusDot.className = "status-dot pulse";
        statusDot.style.background = "#10b981";
      }
    }

    if (containersRes.ok) {
      const containers = await containersRes.json();
      state.containers = containers;
      renderContainers(containers);
      checkAlerts(containers);
    }
  } catch (err) {
    console.error("Data refresh failed:", err);
    const statusText = document.getElementById("system-status-text");
    const statusDot = document.querySelector(".status-dot");
    if (statusText) statusText.textContent = "Connecting to Backend...";
    if (statusDot) {
      statusDot.className = "status-dot";
      statusDot.style.background = "#f59e0b";
    }
  }
}

function renderKPIs(s) {
  kpiTotalItems.textContent = s.total_items_processed || 0;
  kpiAccuracy.textContent = (s.sorting_accuracy_rate_pct || 97.5) + "%";
  kpiWeight.textContent = (s.total_weight_kg || 0).toFixed(1) + " kg";
  kpiCarbon.textContent = (s.carbon_offset_kg || 0).toFixed(1) + " kg";
}

function renderContainers(containers) {
  containersGrid.innerHTML = "";

  containers.forEach((c) => {
    const card = document.createElement("div");
    card.className = "card bin-card";

    const pct = c.fill_percentage;
    const status = c.status;

    card.innerHTML = `
      <div class="bin-header">
        <div>
          <h3 class="bin-title">${c.name}</h3>
          <span class="bin-loc">${c.location_name} • ${c.waste_type}</span>
        </div>
        <span class="status-pill ${status}">${status}</span>
      </div>

      <div class="gauge-section">
        <div class="gauge-meta">
          <span>Fill Level</span>
          <span class="gauge-pct">${pct}% (${c.current_volume_liters.toFixed(1)} / ${c.capacity_liters} L)</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill ${status}" style="width: ${Math.min(100, pct)}%"></div>
        </div>
      </div>

      <div class="bin-telemetry-row">
        <div class="telemetry-item">
          <span>Weight</span>
          <span>${c.current_weight_kg.toFixed(1)} kg</span>
        </div>
        <div class="telemetry-item">
          <span>Est. Full</span>
          <span>${c.estimated_hours_to_full} hrs</span>
        </div>
        <div class="telemetry-item">
          <span>Battery</span>
          <span>${c.battery_level_pct || 92}%</span>
        </div>
      </div>

      <div class="bin-actions">
        <button class="btn btn-secondary btn-sm" onclick="depositToBin('${c.id}')">+5L Deposit</button>
        <button class="btn btn-outline btn-sm" onclick="emptyBin('${c.id}')">Empty Bin</button>
      </div>
    `;
    containersGrid.appendChild(card);
  });
}

function checkAlerts(containers) {
  const criticals = containers.filter((c) => c.status === "CRITICAL" || c.status === "OVERFLOW");
  const warnings = containers.filter((c) => c.status === "WARNING");

  if (criticals.length > 0 || warnings.length > 0) {
    alertsContainer.classList.remove("hidden");
    alertsContent.innerHTML = "";

    criticals.forEach((c) => {
      const p = document.createElement("div");
      p.innerHTML = `<strong>CRITICAL ALERT:</strong> ${c.name} reached <strong>${c.fill_percentage}%</strong>! Immediate truck dispatch required.`;
      alertsContent.appendChild(p);
    });

    warnings.forEach((c) => {
      const p = document.createElement("div");
      p.innerHTML = `<strong>WARNING:</strong> ${c.name} at <strong>${c.fill_percentage}%</strong> capacity.`;
      alertsContent.appendChild(p);
    });
  } else {
    alertsContainer.classList.add("hidden");
  }
}

// Action Button Listeners
function setupActionButtons() {
  btnSimulateTick.addEventListener("click", async () => {
    btnSimulateTick.disabled = true;
    btnSimulateTick.textContent = "Simulating...";
    try {
      await fetch("/api/simulation/tick", { method: "POST" });
      await refreshAllData();
    } catch (err) {
      console.error(err);
    } finally {
      btnSimulateTick.disabled = false;
      btnSimulateTick.innerHTML = `
        <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="23 4 23 10 17 10"></polyline>
          <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
        </svg>
        Simulate 1-Hr City Traffic
      `;
    }
  });

  btnEmptyAll.addEventListener("click", async () => {
    if (confirm("Reset and empty all smart containers in the city?")) {
      await fetch("/api/containers/empty-all", { method: "POST" });
      refreshAllData();
    }
  });

  btnOptimizeRoute.addEventListener("click", runRouteOptimization);
}

// Route Optimization
async function runRouteOptimization() {
  btnOptimizeRoute.disabled = true;
  btnOptimizeRoute.textContent = "Computing TSP Shortest Path...";

  try {
    const res = await fetch("/api/dss/route-optimization?min_fill_pct=70.0");
    if (!res.ok) throw new Error("Optimization failed");
    const data = await res.json();
    renderRouteOptimization(data);
  } catch (err) {
    console.error(err);
  } finally {
    btnOptimizeRoute.disabled = false;
    btnOptimizeRoute.innerHTML = `
      <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="3 11 22 2 13 21 11 13 3 11"></polygon>
      </svg>
      Compute Optimal Route
    `;
  }
}

function renderRouteOptimization(data) {
  document.getElementById("route-stops-count").textContent = data.total_bins_visited;
  document.getElementById("route-distance").textContent = data.total_distance_km.toFixed(1) + " km";
  document.getElementById("route-time").textContent = data.estimated_time_minutes.toFixed(0) + " mins";
  document.getElementById("route-volume").textContent = data.total_volume_collected_liters.toFixed(0) + " L";
  document.getElementById("route-fuel").textContent = data.fuel_saved_liters.toFixed(1) + " L";
  document.getElementById("route-co2").textContent = data.co2_reduction_kg.toFixed(1) + " kg";

  const timeline = document.getElementById("waypoints-timeline");
  timeline.innerHTML = "";

  if (!data.waypoints || data.waypoints.length === 0) {
    timeline.innerHTML = `<p class="text-muted" style="padding: 1.5rem; text-align: center;">All containers are within normal levels (&lt;70%). No truck dispatch needed!</p>`;
    return;
  }

  data.waypoints.forEach((wp) => {
    const node = document.createElement("div");
    node.className = `waypoint-node ${wp.urgency_level}`;
    node.innerHTML = `
      <div class="waypoint-left">
        <div class="waypoint-index">${wp.stop_number}</div>
        <div>
          <div class="waypoint-name">${wp.bin_name}</div>
          <div class="waypoint-type">${wp.waste_type} • ${wp.urgency_level} • Current Fill: ${wp.fill_percentage}%</div>
        </div>
      </div>
      <div class="waypoint-right">
        <div>Collect: <strong>${wp.volume_to_collect_liters} L</strong> (${wp.weight_to_collect_kg} kg)</div>
        <div>Cumulative: <strong>${wp.cumulative_distance_km} km</strong></div>
        <div>ETA: <strong>+${wp.estimated_arrival_minutes} min</strong></div>
      </div>
    `;
    timeline.appendChild(node);
  });
}

// Global actions attached to window
window.depositToBin = async function (binId) {
  const container = state.containers.find((c) => c.id === binId);
  if (!container) return;
  const newVol = container.current_volume_liters + 5.0;
  await fetch(`/api/containers/${binId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ current_volume_liters: newVol }),
  });
  refreshAllData();
};

window.emptyBin = async function (binId) {
  await fetch(`/api/containers/${binId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reset_to_empty: true }),
  });
  refreshAllData();
};
