# Generation view queue count per model ID
_Exported on 12/13/2025 at 10:57:40 EST from Cursor (2.1.39)_

---

**User**

please update the generation view logic in @static to show the queue count for each model id, not total

---

**Cursor**

```javascript
// Show per-model queue counts
if (serverStatus.queue_by_model) {
  const modelQueues = Object.entries(serverStatus.queue_by_model);
  if (modelQueues.length > 0) {
    const queueParts = modelQueues.map(([modelId, info]) => {
      const name = getModelDisplayName(modelId) || modelId;
      const count = info.pending_count + (info.is_processing ? 1 : 0);
      return `${name}: ${count}`;
    });
    statusParts.push(`📋 ${queueParts.join(", ")}`);
  }
} else if (serverStatus.queue_length > 0) {
  // Fallback for backwards compatibility
  statusParts.push(`📋 Queued: ${serverStatus.queue_length}`);
}
```

```javascript
// Build per-model queue message
let queueMsg = `${status.queue_length} more item(s) in queue...`;
if (status.queue_by_model) {
  const parts = Object.entries(status.queue_by_model)
    .filter(([, info]) => info.pending_count > 0)
    .map(([modelId, info]) => {
      const name = getModelDisplayName(modelId) || modelId;
      return `${name}: ${info.pending_count}`;
    });
  if (parts.length > 0) {
    queueMsg = parts.join(", ");
  }
}
showToast("info", "Processing queue", queueMsg);
```

Let me examine the current logic for displaying queue counts in `viewer.js`. I can see the relevant section is around lines 880-895 in the `updateSelectionStatus` function. Let me make the change to show per-model queue counts instead of total.

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";

// Save selected model ID to localStorage
function saveSelectedModel(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY_MODEL, modelId);
  } catch (e) {
    console.warn("Could not save model to localStorage:", e);
  }
}

// Get saved model ID from localStorage
function getSavedModel() {
  try {
    return localStorage.getItem(STORAGE_KEY_MODEL);
  } catch (e) {
    return null;
  }
}

// Save selected tool to localStorage
function saveSelectedTool(toolName) {
  try {
    localStorage.setItem(STORAGE_KEY_TOOL, toolName || "");
  } catch (e) {
    console.warn("Could not save tool to localStorage:", e);
  }
}

// Get saved tool from localStorage
function getSavedTool() {
  try {
    return localStorage.getItem(STORAGE_KEY_TOOL) || "";
  } catch (e) {
    return "";
  }
}

// Save selected quadrants to localStorage
function saveSelectedQuadrants() {
  try {
    const quadrantsArray = Array.from(selectedQuadrants);
    localStorage.setItem(STORAGE_KEY_SELECTION, JSON.stringify(quadrantsArray));
  } catch (e) {
    console.warn("Could not save selection to localStorage:", e);
  }
}

// Get saved quadrants from localStorage
function getSavedQuadrants() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTION);
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    return [];
  }
}

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Check if saved model ID exists in available models
  const savedModelId = getSavedModel();
  const savedModelExists =
    savedModelId && config.models.some((m) => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;

    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (
      !savedModelExists &&
      model.model_id === config.default_model_id
    ) {
      option.selected = true;
    } else if (!savedModelExists && !config.default_model_id && index === 0) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection and save to localStorage
  select.addEventListener("change", () => {
    saveSelectedModel(select.value);
    select.blur();
  });
}

// Get the currently selected model ID
function getSelectedModelId() {
  const select = document.getElementById("modelSelect");
  return select ? select.value : null;
}

// Get display name for a model ID
function getModelDisplayName(modelId) {
  if (!modelId) return null;
  const configEl = document.getElementById("app-config");
  if (!configEl) return modelId;
  try {
    const config = JSON.parse(configEl.dataset.config);
    const models = config.models || [];
    const model = models.find((m) => m.model_id === modelId);
    return model ? model.name : modelId;
  } catch {
    return modelId;
  }
}

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to ALL currently processing quadrants (from all models)
  const processingQuadrants =
    status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || status.active_model_count > 0;

  if (isProcessing && processingQuadrants.length > 0) {
    document.body.classList.add("generating");
    processingQuadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else if (!isProcessing) {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items AND create overlays
  // Also create overlays for processing items
  const processingItems = getProcessingItems(status);
  updateQueueOverlays(status.queue || [], processingItems);
}

// Extract currently processing items from status
function getProcessingItems(status) {
  const processingItems = [];
  if (status.queue_by_model) {
    Object.entries(status.queue_by_model).forEach(([modelId, info]) => {
      if (info.is_processing && info.current_item) {
        processingItems.push({
          ...info.current_item,
          model_id: modelId,
          _isProcessing: true,
        });
      }
    });
  }
  return processingItems;
}

// Create/update overlays for pending queue items and processing items
function updateQueueOverlays(queueItems, processingItems = []) {
  // Remove existing overlays
  document.querySelectorAll(".queue-overlay").forEach((el) => el.remove());

  // Also clear queued class from all tiles
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    tile.classList.remove("queued");
  });

  const hasItems =
    (queueItems && queueItems.length > 0) ||
    (processingItems && processingItems.length > 0);
  if (!hasItems) return;

  const grid = document.querySelector(".grid");
  if (!grid) return;

  // Get grid dimensions from config
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Helper function to create an overlay for an item
  function createOverlay(item, options = {}) {
    const { isProcessing = false, queuePosition = null } = options;

    if (!item.quadrants || item.quadrants.length === 0) return null;

    // Calculate bounding box for this generation
    const quadrants = item.quadrants;
    let minCol = Infinity,
      maxCol = -Infinity;
    let minRow = Infinity,
      maxRow = -Infinity;

    // Track which tiles are visible in the current view
    const visibleQuadrants = [];

    quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        visibleQuadrants.push([qx, qy]);
        const col = qx - gridX;
        const row = qy - gridY;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);

        // Add appropriate class to tile
        if (isProcessing) {
          tile.classList.add("locked");
        } else if (!tile.classList.contains("locked")) {
          tile.classList.add("queued");
        }
      }
    });

    if (visibleQuadrants.length === 0) return null;

    // Create overlay element
    const overlay = document.createElement("div");
    overlay.className = isProcessing
      ? "queue-overlay processing"
      : "queue-overlay";
    overlay.dataset.itemId = item.id;

    // Calculate position and size
    const left = minCol * (sizePx + gap);
    const top = minRow * (sizePx + gap);
    const width = (maxCol - minCol + 1) * sizePx + (maxCol - minCol) * gap;
    const height = (maxRow - minRow + 1) * sizePx + (maxRow - minRow) * gap;

    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    // Badge: spinner for processing, number for queued
    const badge = document.createElement("div");
    badge.className = isProcessing ? "queue-badge processing" : "queue-badge";
    if (isProcessing) {
      badge.innerHTML = `<svg class="processing-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>`;
    } else {
      badge.textContent = queuePosition;
    }
    overlay.appendChild(badge);

    // Model name label
    if (item.model_id) {
      const modelLabel = document.createElement("div");
      modelLabel.className = "queue-model-label";
      modelLabel.textContent =
        getModelDisplayName(item.model_id) || item.model_id;
      overlay.appendChild(modelLabel);
    }

    // Cancel button (visible on hover) - for both processing and queued
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "queue-cancel-btn";
    cancelBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>`;
    cancelBtn.title = isProcessing
      ? "Cancel this generation (in progress)"
      : "Cancel this generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelQueueItem(item.id);
    };
    overlay.appendChild(cancelBtn);

    return overlay;
  }

  // First, create overlays for processing items (with spinner)
  processingItems.forEach((item) => {
    const overlay = createOverlay(item, { isProcessing: true });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });

  // Then, create overlays for queued items (with position numbers)
  queueItems.forEach((item, index) => {
    const overlay = createOverlay(item, { queuePosition: index + 1 });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });
}

// Cancel a specific queue item
async function cancelQueueItem(itemId) {
  try {
    const response = await fetch(`/api/queue/cancel/${itemId}`, {
      method: "POST",
    });
    const result = await response.json();

    if (result.success && result.cancelled) {
      showToast("success", "Cancelled", result.message);
      // Trigger immediate status update
      checkGenerationStatus();
    } else if (result.success && !result.cancelled) {
      showToast("info", "Not found", result.message);
    } else {
      showToast("error", "Error", result.error || "Failed to cancel");
    }
  } catch (error) {
    console.error("Cancel queue item failed:", error);
    showToast("error", "Error", "Failed to cancel queue item");
  }
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "d":
    case "D":
      document.getElementById("showRender").click();
      break;
    case "g":
    case "G":
      generateSelected();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate other tools if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
    saveSelectedTool("select");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
    saveSelectedTool("");
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate other tools if active
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
    saveSelectedTool("fixwater");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast(
    "loading",
    "Filling with water...",
    `Processing quadrant (${qx}, ${qy})`
  );

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus(serverStatus = null) {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");
  const generateRectBtn = document.getElementById("generateRectBtn");

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    // Show all active models generating
    const activeModels = serverStatus.active_models || [];
    const processingQuadrants =
      serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

    if (activeModels.length > 0 && processingQuadrants.length > 0) {
      // Show which models are actively generating
      const modelNames = activeModels.map(
        (modelId) => getModelDisplayName(modelId) || modelId || "default"
      );

      if (activeModels.length === 1) {
        const coordsStr = processingQuadrants
          .map(([x, y]) => `(${x},${y})`)
          .join(" ");
        statusParts.push(`🔄 ${modelNames[0]}: ${coordsStr}`);
      } else {
        // Multiple models generating in parallel
        statusParts.push(`🔄 Generating: ${modelNames.join(", ")}`);
      }
    } else if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      // Fallback to old behavior
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsStr = serverStatus.quadrants
        .map(([x, y]) => `(${x},${y})`)
        .join(" ");
      statusParts.push(`${action} ${coordsStr}`);
    }

    // Show per-model queue counts (only models with items)
    if (serverStatus.queue_by_model) {
      const modelQueues = Object.entries(serverStatus.queue_by_model);
      const queueParts = modelQueues
        .map(([modelId, info]) => {
          const name = getModelDisplayName(modelId) || modelId;
          const count = info.pending_count + (info.is_processing ? 1 : 0);
          return { name, count };
        })
        .filter(({ count }) => count > 0)
        .map(({ name, count }) => `${name}: ${count}`);

      if (queueParts.length > 0) {
        statusParts.push(`📋 ${queueParts.join(", ")}`);
      }
    }
  }

  // Update selection display in toolbar-info (separate from status)
  const selectedDisplay = document.getElementById("selectedQuadrantsDisplay");
  if (selectedDisplay) {
    if (count > 0) {
      const coordsStr = Array.from(selectedQuadrants)
        .map((key) => {
          const [x, y] = key.split(",");
          return `(${x},${y})`;
        })
        .join(" ");
      selectedDisplay.textContent = `✓ ${coordsStr}`;
      selectedDisplay.style.display = "";
    } else {
      selectedDisplay.textContent = "";
      selectedDisplay.style.display = "none";
    }
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = `${count} selected`;
  }

  countEl.textContent = statusText;

  // Update status bar styling
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (isProcessing) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate Rectangle requires exactly 2 selected
  generateRectBtn.disabled = count !== 2;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state (tracked from server)
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function clearQueue() {
  // Get current queue info first
  try {
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();

    const queueLength = status.queue_length || 0;
    const isGenerating = status.is_generating || false;

    if (queueLength === 0 && !isGenerating) {
      showToast(
        "info",
        "Nothing to clear",
        "There are no pending items or active generations."
      );
      return;
    }

    // Build confirm message based on what will be cancelled
    let confirmMessage = "Are you sure you want to clear the queue?";
    const parts = [];
    if (queueLength > 0) {
      parts.push(`${queueLength} pending item(s)`);
    }
    if (isGenerating) {
      parts.push("the current generation in progress");
    }
    if (parts.length > 0) {
      confirmMessage = `Are you sure you want to cancel ${parts.join(
        " and "
      )}?`;
    }

    if (!confirm(confirmMessage)) {
      return;
    }

    // Clear the queue
    const response = await fetch("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        "Queue cleared",
        result.message || `Cleared ${result.cleared_count} item(s)`
      );
      // Refresh status
      await checkGenerationStatus();
    } else {
      showToast(
        "error",
        "Failed to clear queue",
        result.error || "Unknown error"
      );
    }
  } catch (error) {
    console.error("Clear queue error:", error);
    showToast("error", "Request failed", error.message);
  }
}

// Prompt dialog functions
function showPromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("promptDialog");
  const input = document.getElementById("promptInput");
  if (dialog && input) {
    input.value = "";
    dialog.style.display = "flex";
    input.focus();
  }
}

function hidePromptDialog() {
  const dialog = document.getElementById("promptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitPromptGeneration() {
  const input = document.getElementById("promptInput");
  const prompt = input ? input.value.trim() : "";
  hidePromptDialog();
  await generateSelected(prompt);
}

async function generateSelected(prompt = null) {
  if (selectedQuadrants.size === 0) return;

  // Separate selected quadrants into those that need generation vs those that already have it
  const toGenerate = [];
  const contextQuadrants = [];

  Array.from(selectedQuadrants).forEach((s) => {
    const [x, y] = s.split(",").map(Number);
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

    if (tile && !tile.classList.contains("placeholder")) {
      // This quadrant already has a generation - use as context
      contextQuadrants.push([x, y]);
    } else {
      // This quadrant needs generation
      toGenerate.push([x, y]);
    }
  });

  // If nothing needs generation, inform the user
  if (toGenerate.length === 0) {
    showToast(
      "info",
      "Already generated",
      "All selected quadrants already have generations. Select at least one empty quadrant."
    );
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback with context info
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${toGenerate.length} quadrant(s) for generation${promptMsg}${contextMsg}`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const requestBody = {
      quadrants: toGenerate,
      model_id: modelId,
    };

    // Include context quadrants if any
    if (contextQuadrants.length > 0) {
      requestBody.context = contextQuadrants;
    }

    // Include prompt if provided
    if (prompt) {
      requestBody.prompt = prompt;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.queued) {
      console.log(
        "Generation queued at position:",
        result.position,
        "model:",
        result.model_id
      );
      // Get model name for display
      const modelName = getModelDisplayName(result.model_id);
      const modelInfo = modelName ? ` (${modelName})` : "";
      showToast(
        "success",
        "Added to queue",
        `Queued at position ${result.position}${modelInfo}`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for render`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Render queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function generateRectangle() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the rectangle corners."
    );
    return;
  }

  // Get the two selected coordinates
  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return { x, y };
  });

  // Calculate rectangle bounds (top-left and bottom-right)
  const minX = Math.min(coords[0].x, coords[1].x);
  const maxX = Math.max(coords[0].x, coords[1].x);
  const minY = Math.min(coords[0].y, coords[1].y);
  const maxY = Math.max(coords[0].y, coords[1].y);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const totalQuadrants = width * height;

  // Build confirmation message
  const confirmMessage =
    `Generate rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY})?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `This will create a generation plan and queue all steps.\n` +
    `Pre-existing generations will be skipped.`;

  if (!confirm(confirmMessage)) {
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate rectangle requested:",
    { tl: [minX, minY], br: [maxX, maxY] },
    "with model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("generateRectBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Queueing...<span class="spinner"></span>';

  showToast(
    "loading",
    "Creating generation plan...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY})`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        model_id: modelId,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";

    if (result.success) {
      if (result.queued_count === 0) {
        showToast(
          "info",
          "Nothing to generate",
          result.message || "All quadrants already generated."
        );
      } else {
        console.log("Rectangle generation queued:", result);
        const summary = result.plan_summary || {};
        const stepTypes = summary.steps_by_type || {};
        const typeInfo = Object.entries(stepTypes)
          .map(([type, count]) => `${count}× ${type}`)
          .join(", ");

        showToast(
          "success",
          "Rectangle queued!",
          `${result.queued_count} step(s) for ${
            summary.total_quadrants || "?"
          } quadrant(s)` + (typeInfo ? ` (${typeInfo})` : "")
        );
      }
    } else {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    clearLoadingToasts();
    console.error("Generate rectangle error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  saveSelectedQuadrants();
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  if (
    tileEl.classList.contains("locked") ||
    tileEl.classList.contains("queued")
  ) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  saveSelectedQuadrants();
  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;
let lastStatus = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Track state changes - consider active_model_count for parallel processing
    const wasGenerating = isGenerating || isRendering;
    const hasActiveModels = (status.active_model_count || 0) > 0;
    isGenerating =
      (status.is_generating || hasActiveModels) &&
      status.status !== "rendering";
    isRendering = status.is_generating && status.status === "rendering";
    const nowProcessing = isGenerating || isRendering;

    // Apply visual styles based on server status
    applyStatusStyles(status);

    // Update render button based on state
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on render button only (generate state is shown in toolbar)
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      } else {
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
      }

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        const opName = isRendering ? "Render" : "Generation";
        showToast(
          "loading",
          `${opName} in progress...`,
          status.message || "Please wait..."
        );
      }

      // Update the loading toast message
      const loadingToast = document.querySelector(
        ".toast.loading .toast-message"
      );
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset render button
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        // Build per-model queue message
        let queueMsg = `${status.queue_length} more item(s) in queue...`;
        if (status.queue_by_model) {
          const parts = Object.entries(status.queue_by_model)
            .filter(([, info]) => info.pending_count > 0)
            .map(([modelId, info]) => {
              const name = getModelDisplayName(modelId) || modelId;
              return `${name}: ${info.pending_count}`;
            });
          if (parts.length > 0) {
            queueMsg = parts.join(", ");
          }
        }
        showToast("info", "Processing queue", queueMsg);
      } else {
        // No more items, reload after delay
        stopStatusPolling();
        setTimeout(() => window.location.reload(), 1500);
      }
    } else if (status.status === "error" && status.error) {
      clearLoadingToasts();
      showToast("error", "Error", status.error);

      // Continue polling if there are more items in queue
      if (status.queue_length === 0) {
        stopStatusPolling();
      }
    } else if (
      status.status === "idle" &&
      status.queue_length === 0 &&
      !nowProcessing
    ) {
      // Idle with no queue - stop polling
      stopStatusPolling();
    }

    // Update selection status with server info
    updateSelectionStatus(status);
    lastStatus = status;
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

// Restore saved tool on page load
function restoreSavedTool() {
  const savedTool = getSavedTool();
  if (!savedTool) return;

  // Check if the tool button exists before activating
  switch (savedTool) {
    case "select":
      if (document.getElementById("selectTool")) {
        toggleSelectTool();
      }
      break;
    case "fixwater":
      if (document.getElementById("fixWaterTool")) {
        toggleFixWaterTool();
      }
      break;
    case "waterfill":
      if (document.getElementById("waterFillTool")) {
        toggleWaterFillTool();
      }
      break;
    default:
      // Unknown tool, clear saved state
      saveSelectedTool("");
      break;
  }
}

// Restore saved quadrant selections on page load
function restoreSavedQuadrants() {
  const savedQuadrants = getSavedQuadrants();
  if (!savedQuadrants || savedQuadrants.length === 0) return;

  let restoredCount = 0;

  savedQuadrants.forEach((key) => {
    // Check if this quadrant tile exists on the current page
    const tile = document.querySelector(`.tile[data-coords="${key}"]`);
    if (tile) {
      // Don't restore if tile is locked or queued
      if (
        !tile.classList.contains("locked") &&
        !tile.classList.contains("queued")
      ) {
        selectedQuadrants.add(key);
        tile.classList.add("selected");
        restoredCount++;
      }
    }
  });

  if (restoredCount > 0) {
    console.log(`Restored ${restoredCount} selected quadrant(s)`);
    // Update localStorage to only contain valid selections
    saveSelectedQuadrants();
    updateSelectionStatus();
  } else if (savedQuadrants.length > 0) {
    // Had saved selections but none are on current page - clear storage
    saveSelectedQuadrants();
  }
}

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

  // Restore saved quadrant selections
  restoreSavedQuadrants();

  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Apply initial status styles
    applyStatusStyles(status);

    if (status.is_generating || status.queue_length > 0) {
      console.log(
        "Processing in progress or queue non-empty, starting polling..."
      );
      isGenerating = status.is_generating && status.status !== "rendering";
      isRendering = status.is_generating && status.status === "rendering";
      startStatusPolling();
    }

    updateSelectionStatus(status);
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now let me also update the toast message that appears when processing completes to show per-model counts consistently:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";

// Save selected model ID to localStorage
function saveSelectedModel(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY_MODEL, modelId);
  } catch (e) {
    console.warn("Could not save model to localStorage:", e);
  }
}

// Get saved model ID from localStorage
function getSavedModel() {
  try {
    return localStorage.getItem(STORAGE_KEY_MODEL);
  } catch (e) {
    return null;
  }
}

// Save selected tool to localStorage
function saveSelectedTool(toolName) {
  try {
    localStorage.setItem(STORAGE_KEY_TOOL, toolName || "");
  } catch (e) {
    console.warn("Could not save tool to localStorage:", e);
  }
}

// Get saved tool from localStorage
function getSavedTool() {
  try {
    return localStorage.getItem(STORAGE_KEY_TOOL) || "";
  } catch (e) {
    return "";
  }
}

// Save selected quadrants to localStorage
function saveSelectedQuadrants() {
  try {
    const quadrantsArray = Array.from(selectedQuadrants);
    localStorage.setItem(STORAGE_KEY_SELECTION, JSON.stringify(quadrantsArray));
  } catch (e) {
    console.warn("Could not save selection to localStorage:", e);
  }
}

// Get saved quadrants from localStorage
function getSavedQuadrants() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTION);
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    return [];
  }
}

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Check if saved model ID exists in available models
  const savedModelId = getSavedModel();
  const savedModelExists =
    savedModelId && config.models.some((m) => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;

    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (
      !savedModelExists &&
      model.model_id === config.default_model_id
    ) {
      option.selected = true;
    } else if (!savedModelExists && !config.default_model_id && index === 0) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection and save to localStorage
  select.addEventListener("change", () => {
    saveSelectedModel(select.value);
    select.blur();
  });
}

// Get the currently selected model ID
function getSelectedModelId() {
  const select = document.getElementById("modelSelect");
  return select ? select.value : null;
}

// Get display name for a model ID
function getModelDisplayName(modelId) {
  if (!modelId) return null;
  const configEl = document.getElementById("app-config");
  if (!configEl) return modelId;
  try {
    const config = JSON.parse(configEl.dataset.config);
    const models = config.models || [];
    const model = models.find((m) => m.model_id === modelId);
    return model ? model.name : modelId;
  } catch {
    return modelId;
  }
}

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to ALL currently processing quadrants (from all models)
  const processingQuadrants =
    status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || status.active_model_count > 0;

  if (isProcessing && processingQuadrants.length > 0) {
    document.body.classList.add("generating");
    processingQuadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else if (!isProcessing) {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items AND create overlays
  // Also create overlays for processing items
  const processingItems = getProcessingItems(status);
  updateQueueOverlays(status.queue || [], processingItems);
}

// Extract currently processing items from status
function getProcessingItems(status) {
  const processingItems = [];
  if (status.queue_by_model) {
    Object.entries(status.queue_by_model).forEach(([modelId, info]) => {
      if (info.is_processing && info.current_item) {
        processingItems.push({
          ...info.current_item,
          model_id: modelId,
          _isProcessing: true,
        });
      }
    });
  }
  return processingItems;
}

// Create/update overlays for pending queue items and processing items
function updateQueueOverlays(queueItems, processingItems = []) {
  // Remove existing overlays
  document.querySelectorAll(".queue-overlay").forEach((el) => el.remove());

  // Also clear queued class from all tiles
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    tile.classList.remove("queued");
  });

  const hasItems =
    (queueItems && queueItems.length > 0) ||
    (processingItems && processingItems.length > 0);
  if (!hasItems) return;

  const grid = document.querySelector(".grid");
  if (!grid) return;

  // Get grid dimensions from config
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Helper function to create an overlay for an item
  function createOverlay(item, options = {}) {
    const { isProcessing = false, queuePosition = null } = options;

    if (!item.quadrants || item.quadrants.length === 0) return null;

    // Calculate bounding box for this generation
    const quadrants = item.quadrants;
    let minCol = Infinity,
      maxCol = -Infinity;
    let minRow = Infinity,
      maxRow = -Infinity;

    // Track which tiles are visible in the current view
    const visibleQuadrants = [];

    quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        visibleQuadrants.push([qx, qy]);
        const col = qx - gridX;
        const row = qy - gridY;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);

        // Add appropriate class to tile
        if (isProcessing) {
          tile.classList.add("locked");
        } else if (!tile.classList.contains("locked")) {
          tile.classList.add("queued");
        }
      }
    });

    if (visibleQuadrants.length === 0) return null;

    // Create overlay element
    const overlay = document.createElement("div");
    overlay.className = isProcessing
      ? "queue-overlay processing"
      : "queue-overlay";
    overlay.dataset.itemId = item.id;

    // Calculate position and size
    const left = minCol * (sizePx + gap);
    const top = minRow * (sizePx + gap);
    const width = (maxCol - minCol + 1) * sizePx + (maxCol - minCol) * gap;
    const height = (maxRow - minRow + 1) * sizePx + (maxRow - minRow) * gap;

    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    // Badge: spinner for processing, number for queued
    const badge = document.createElement("div");
    badge.className = isProcessing ? "queue-badge processing" : "queue-badge";
    if (isProcessing) {
      badge.innerHTML = `<svg class="processing-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>`;
    } else {
      badge.textContent = queuePosition;
    }
    overlay.appendChild(badge);

    // Model name label
    if (item.model_id) {
      const modelLabel = document.createElement("div");
      modelLabel.className = "queue-model-label";
      modelLabel.textContent =
        getModelDisplayName(item.model_id) || item.model_id;
      overlay.appendChild(modelLabel);
    }

    // Cancel button (visible on hover) - for both processing and queued
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "queue-cancel-btn";
    cancelBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>`;
    cancelBtn.title = isProcessing
      ? "Cancel this generation (in progress)"
      : "Cancel this generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelQueueItem(item.id);
    };
    overlay.appendChild(cancelBtn);

    return overlay;
  }

  // First, create overlays for processing items (with spinner)
  processingItems.forEach((item) => {
    const overlay = createOverlay(item, { isProcessing: true });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });

  // Then, create overlays for queued items (with position numbers)
  queueItems.forEach((item, index) => {
    const overlay = createOverlay(item, { queuePosition: index + 1 });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });
}

// Cancel a specific queue item
async function cancelQueueItem(itemId) {
  try {
    const response = await fetch(`/api/queue/cancel/${itemId}`, {
      method: "POST",
    });
    const result = await response.json();

    if (result.success && result.cancelled) {
      showToast("success", "Cancelled", result.message);
      // Trigger immediate status update
      checkGenerationStatus();
    } else if (result.success && !result.cancelled) {
      showToast("info", "Not found", result.message);
    } else {
      showToast("error", "Error", result.error || "Failed to cancel");
    }
  } catch (error) {
    console.error("Cancel queue item failed:", error);
    showToast("error", "Error", "Failed to cancel queue item");
  }
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "d":
    case "D":
      document.getElementById("showRender").click();
      break;
    case "g":
    case "G":
      generateSelected();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate other tools if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
    saveSelectedTool("select");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
    saveSelectedTool("");
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate other tools if active
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
    saveSelectedTool("fixwater");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast(
    "loading",
    "Filling with water...",
    `Processing quadrant (${qx}, ${qy})`
  );

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus(serverStatus = null) {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");
  const generateRectBtn = document.getElementById("generateRectBtn");

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    // Show all active models generating
    const activeModels = serverStatus.active_models || [];
    const processingQuadrants =
      serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

    if (activeModels.length > 0 && processingQuadrants.length > 0) {
      // Show which models are actively generating
      const modelNames = activeModels.map(
        (modelId) => getModelDisplayName(modelId) || modelId || "default"
      );

      if (activeModels.length === 1) {
        const coordsStr = processingQuadrants
          .map(([x, y]) => `(${x},${y})`)
          .join(" ");
        statusParts.push(`🔄 ${modelNames[0]}: ${coordsStr}`);
      } else {
        // Multiple models generating in parallel
        statusParts.push(`🔄 Generating: ${modelNames.join(", ")}`);
      }
    } else if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      // Fallback to old behavior
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsStr = serverStatus.quadrants
        .map(([x, y]) => `(${x},${y})`)
        .join(" ");
      statusParts.push(`${action} ${coordsStr}`);
    }

    // Show per-model queue counts (only models with items)
    if (serverStatus.queue_by_model) {
      const modelQueues = Object.entries(serverStatus.queue_by_model);
      const queueParts = modelQueues
        .map(([modelId, info]) => {
          const name = getModelDisplayName(modelId) || modelId;
          const count = info.pending_count + (info.is_processing ? 1 : 0);
          return { name, count };
        })
        .filter(({ count }) => count > 0)
        .map(({ name, count }) => `${name}: ${count}`);

      if (queueParts.length > 0) {
        statusParts.push(`📋 ${queueParts.join(", ")}`);
      }
    }
  }

  // Update selection display in toolbar-info (separate from status)
  const selectedDisplay = document.getElementById("selectedQuadrantsDisplay");
  if (selectedDisplay) {
    if (count > 0) {
      const coordsStr = Array.from(selectedQuadrants)
        .map((key) => {
          const [x, y] = key.split(",");
          return `(${x},${y})`;
        })
        .join(" ");
      selectedDisplay.textContent = `✓ ${coordsStr}`;
      selectedDisplay.style.display = "";
    } else {
      selectedDisplay.textContent = "";
      selectedDisplay.style.display = "none";
    }
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = `${count} selected`;
  }

  countEl.textContent = statusText;

  // Update status bar styling
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (isProcessing) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate Rectangle requires exactly 2 selected
  generateRectBtn.disabled = count !== 2;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state (tracked from server)
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function clearQueue() {
  // Get current queue info first
  try {
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();

    const queueLength = status.queue_length || 0;
    const isGenerating = status.is_generating || false;

    if (queueLength === 0 && !isGenerating) {
      showToast(
        "info",
        "Nothing to clear",
        "There are no pending items or active generations."
      );
      return;
    }

    // Build confirm message based on what will be cancelled
    let confirmMessage = "Are you sure you want to clear the queue?";
    const parts = [];
    if (queueLength > 0) {
      parts.push(`${queueLength} pending item(s)`);
    }
    if (isGenerating) {
      parts.push("the current generation in progress");
    }
    if (parts.length > 0) {
      confirmMessage = `Are you sure you want to cancel ${parts.join(
        " and "
      )}?`;
    }

    if (!confirm(confirmMessage)) {
      return;
    }

    // Clear the queue
    const response = await fetch("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        "Queue cleared",
        result.message || `Cleared ${result.cleared_count} item(s)`
      );
      // Refresh status
      await checkGenerationStatus();
    } else {
      showToast(
        "error",
        "Failed to clear queue",
        result.error || "Unknown error"
      );
    }
  } catch (error) {
    console.error("Clear queue error:", error);
    showToast("error", "Request failed", error.message);
  }
}

// Prompt dialog functions
function showPromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("promptDialog");
  const input = document.getElementById("promptInput");
  if (dialog && input) {
    input.value = "";
    dialog.style.display = "flex";
    input.focus();
  }
}

function hidePromptDialog() {
  const dialog = document.getElementById("promptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitPromptGeneration() {
  const input = document.getElementById("promptInput");
  const prompt = input ? input.value.trim() : "";
  hidePromptDialog();
  await generateSelected(prompt);
}

async function generateSelected(prompt = null) {
  if (selectedQuadrants.size === 0) return;

  // Separate selected quadrants into those that need generation vs those that already have it
  const toGenerate = [];
  const contextQuadrants = [];

  Array.from(selectedQuadrants).forEach((s) => {
    const [x, y] = s.split(",").map(Number);
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

    if (tile && !tile.classList.contains("placeholder")) {
      // This quadrant already has a generation - use as context
      contextQuadrants.push([x, y]);
    } else {
      // This quadrant needs generation
      toGenerate.push([x, y]);
    }
  });

  // If nothing needs generation, inform the user
  if (toGenerate.length === 0) {
    showToast(
      "info",
      "Already generated",
      "All selected quadrants already have generations. Select at least one empty quadrant."
    );
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback with context info
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${toGenerate.length} quadrant(s) for generation${promptMsg}${contextMsg}`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const requestBody = {
      quadrants: toGenerate,
      model_id: modelId,
    };

    // Include context quadrants if any
    if (contextQuadrants.length > 0) {
      requestBody.context = contextQuadrants;
    }

    // Include prompt if provided
    if (prompt) {
      requestBody.prompt = prompt;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.queued) {
      console.log(
        "Generation queued at position:",
        result.position,
        "model:",
        result.model_id
      );
      // Get model name for display
      const modelName = getModelDisplayName(result.model_id);
      const modelInfo = modelName ? ` (${modelName})` : "";
      showToast(
        "success",
        "Added to queue",
        `Queued at position ${result.position}${modelInfo}`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for render`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Render queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function generateRectangle() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the rectangle corners."
    );
    return;
  }

  // Get the two selected coordinates
  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return { x, y };
  });

  // Calculate rectangle bounds (top-left and bottom-right)
  const minX = Math.min(coords[0].x, coords[1].x);
  const maxX = Math.max(coords[0].x, coords[1].x);
  const minY = Math.min(coords[0].y, coords[1].y);
  const maxY = Math.max(coords[0].y, coords[1].y);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const totalQuadrants = width * height;

  // Build confirmation message
  const confirmMessage =
    `Generate rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY})?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `This will create a generation plan and queue all steps.\n` +
    `Pre-existing generations will be skipped.`;

  if (!confirm(confirmMessage)) {
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate rectangle requested:",
    { tl: [minX, minY], br: [maxX, maxY] },
    "with model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("generateRectBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Queueing...<span class="spinner"></span>';

  showToast(
    "loading",
    "Creating generation plan...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY})`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        model_id: modelId,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";

    if (result.success) {
      if (result.queued_count === 0) {
        showToast(
          "info",
          "Nothing to generate",
          result.message || "All quadrants already generated."
        );
      } else {
        console.log("Rectangle generation queued:", result);
        const summary = result.plan_summary || {};
        const stepTypes = summary.steps_by_type || {};
        const typeInfo = Object.entries(stepTypes)
          .map(([type, count]) => `${count}× ${type}`)
          .join(", ");

        showToast(
          "success",
          "Rectangle queued!",
          `${result.queued_count} step(s) for ${
            summary.total_quadrants || "?"
          } quadrant(s)` + (typeInfo ? ` (${typeInfo})` : "")
        );
      }
    } else {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    clearLoadingToasts();
    console.error("Generate rectangle error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  saveSelectedQuadrants();
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  if (
    tileEl.classList.contains("locked") ||
    tileEl.classList.contains("queued")
  ) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  saveSelectedQuadrants();
  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;
let lastStatus = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Track state changes - consider active_model_count for parallel processing
    const wasGenerating = isGenerating || isRendering;
    const hasActiveModels = (status.active_model_count || 0) > 0;
    isGenerating =
      (status.is_generating || hasActiveModels) &&
      status.status !== "rendering";
    isRendering = status.is_generating && status.status === "rendering";
    const nowProcessing = isGenerating || isRendering;

    // Apply visual styles based on server status
    applyStatusStyles(status);

    // Update render button based on state
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on render button only (generate state is shown in toolbar)
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      } else {
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
      }

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        const opName = isRendering ? "Render" : "Generation";
        showToast(
          "loading",
          `${opName} in progress...`,
          status.message || "Please wait..."
        );
      }

      // Update the loading toast message
      const loadingToast = document.querySelector(
        ".toast.loading .toast-message"
      );
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset render button
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        // Build per-model queue message
        let queueMsg = "";
        if (status.queue_by_model) {
          const parts = Object.entries(status.queue_by_model)
            .map(([modelId, info]) => {
              const name = getModelDisplayName(modelId) || modelId;
              const count = info.pending_count + (info.is_processing ? 1 : 0);
              return { name, count };
            })
            .filter(({ count }) => count > 0)
            .map(({ name, count }) => `${name}: ${count}`);
          if (parts.length > 0) {
            queueMsg = parts.join(", ");
          }
        }
        if (queueMsg) {
          showToast("info", "Processing queue", queueMsg);
        }
      } else {
        // No more items, reload after delay
        stopStatusPolling();
        setTimeout(() => window.location.reload(), 1500);
      }
    } else if (status.status === "error" && status.error) {
      clearLoadingToasts();
      showToast("error", "Error", status.error);

      // Continue polling if there are more items in queue
      if (status.queue_length === 0) {
        stopStatusPolling();
      }
    } else if (
      status.status === "idle" &&
      status.queue_length === 0 &&
      !nowProcessing
    ) {
      // Idle with no queue - stop polling
      stopStatusPolling();
    }

    // Update selection status with server info
    updateSelectionStatus(status);
    lastStatus = status;
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

// Restore saved tool on page load
function restoreSavedTool() {
  const savedTool = getSavedTool();
  if (!savedTool) return;

  // Check if the tool button exists before activating
  switch (savedTool) {
    case "select":
      if (document.getElementById("selectTool")) {
        toggleSelectTool();
      }
      break;
    case "fixwater":
      if (document.getElementById("fixWaterTool")) {
        toggleFixWaterTool();
      }
      break;
    case "waterfill":
      if (document.getElementById("waterFillTool")) {
        toggleWaterFillTool();
      }
      break;
    default:
      // Unknown tool, clear saved state
      saveSelectedTool("");
      break;
  }
}

// Restore saved quadrant selections on page load
function restoreSavedQuadrants() {
  const savedQuadrants = getSavedQuadrants();
  if (!savedQuadrants || savedQuadrants.length === 0) return;

  let restoredCount = 0;

  savedQuadrants.forEach((key) => {
    // Check if this quadrant tile exists on the current page
    const tile = document.querySelector(`.tile[data-coords="${key}"]`);
    if (tile) {
      // Don't restore if tile is locked or queued
      if (
        !tile.classList.contains("locked") &&
        !tile.classList.contains("queued")
      ) {
        selectedQuadrants.add(key);
        tile.classList.add("selected");
        restoredCount++;
      }
    }
  });

  if (restoredCount > 0) {
    console.log(`Restored ${restoredCount} selected quadrant(s)`);
    // Update localStorage to only contain valid selections
    saveSelectedQuadrants();
    updateSelectionStatus();
  } else if (savedQuadrants.length > 0) {
    // Had saved selections but none are on current page - clear storage
    saveSelectedQuadrants();
  }
}

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

  // Restore saved quadrant selections
  restoreSavedQuadrants();

  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Apply initial status styles
    applyStatusStyles(status);

    if (status.is_generating || status.queue_length > 0) {
      console.log(
        "Processing in progress or queue non-empty, starting polling..."
      );
      isGenerating = status.is_generating && status.status !== "rendering";
      isRendering = status.is_generating && status.status === "rendering";
      startStatusPolling();
    }

    updateSelectionStatus(status);
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

I've updated the generation view logic in `viewer.js` to show queue counts for each model ID instead of the total. Here's what changed:

1. **In `updateSelectionStatus()`** (status bar display):
   - Now filters out models with 0 items in queue
   - Removed the fallback that showed total queue length (`📋 Queued: X`)
   - Only shows models that actually have queued/processing items: `📋 ModelA: 3, ModelB: 2`

2. **In the completion toast**:
   - Uses consistent logic - includes both pending count and currently processing item per model
   - Only shows models with count > 0
   - Only displays the toast if there are per-model items to show

The display will now show something like `📋 ModelA: 3, ModelB: 2` instead of `📋 Queued: 5`.

---

**User**

make sure its changed in the generation quadrant display

---

**Cursor**

```javascript
// Show current processing status from server
if (serverStatus) {
  // Show all active models generating
  const activeModels = serverStatus.active_models || [];
  const processingQuadrants =
    serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

  if (activeModels.length > 0 && processingQuadrants.length > 0) {
    // Show which models are actively generating
    const modelNames = activeModels.map(
      (modelId) => getModelDisplayName(modelId) || modelId || "default"
    );

    if (activeModels.length === 1) {
      const coordsStr = processingQuadrants
        .map(([x, y]) => `(${x},${y})`)
        .join(" ");
      statusParts.push(`🔄 ${modelNames[0]}: ${coordsStr}`);
    } else {
      // Multiple models generating in parallel
      statusParts.push(`🔄 Generating: ${modelNames.join(", ")}`);
    }
  } else if (
    serverStatus.is_generating &&
    serverStatus.quadrants &&
    serverStatus.quadrants.length > 0
  ) {
    // Fallback to old behavior
    const action =
      serverStatus.status === "rendering" ? "Rendering" : "Generating";
    const coordsStr = serverStatus.quadrants
      .map(([x, y]) => `(${x},${y})`)
      .join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }
```

Let me look at the generation quadrant display section in `updateSelectionStatus` to update that as well:

I see - the generation quadrant display section (lines 851-878) shows which models are generating but doesn't include the queue count. Let me update that to show per-model queue counts alongside the active generation info:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";

// Save selected model ID to localStorage
function saveSelectedModel(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY_MODEL, modelId);
  } catch (e) {
    console.warn("Could not save model to localStorage:", e);
  }
}

// Get saved model ID from localStorage
function getSavedModel() {
  try {
    return localStorage.getItem(STORAGE_KEY_MODEL);
  } catch (e) {
    return null;
  }
}

// Save selected tool to localStorage
function saveSelectedTool(toolName) {
  try {
    localStorage.setItem(STORAGE_KEY_TOOL, toolName || "");
  } catch (e) {
    console.warn("Could not save tool to localStorage:", e);
  }
}

// Get saved tool from localStorage
function getSavedTool() {
  try {
    return localStorage.getItem(STORAGE_KEY_TOOL) || "";
  } catch (e) {
    return "";
  }
}

// Save selected quadrants to localStorage
function saveSelectedQuadrants() {
  try {
    const quadrantsArray = Array.from(selectedQuadrants);
    localStorage.setItem(STORAGE_KEY_SELECTION, JSON.stringify(quadrantsArray));
  } catch (e) {
    console.warn("Could not save selection to localStorage:", e);
  }
}

// Get saved quadrants from localStorage
function getSavedQuadrants() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTION);
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    return [];
  }
}

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Check if saved model ID exists in available models
  const savedModelId = getSavedModel();
  const savedModelExists =
    savedModelId && config.models.some((m) => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;

    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (
      !savedModelExists &&
      model.model_id === config.default_model_id
    ) {
      option.selected = true;
    } else if (!savedModelExists && !config.default_model_id && index === 0) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection and save to localStorage
  select.addEventListener("change", () => {
    saveSelectedModel(select.value);
    select.blur();
  });
}

// Get the currently selected model ID
function getSelectedModelId() {
  const select = document.getElementById("modelSelect");
  return select ? select.value : null;
}

// Get display name for a model ID
function getModelDisplayName(modelId) {
  if (!modelId) return null;
  const configEl = document.getElementById("app-config");
  if (!configEl) return modelId;
  try {
    const config = JSON.parse(configEl.dataset.config);
    const models = config.models || [];
    const model = models.find((m) => m.model_id === modelId);
    return model ? model.name : modelId;
  } catch {
    return modelId;
  }
}

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to ALL currently processing quadrants (from all models)
  const processingQuadrants =
    status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || status.active_model_count > 0;

  if (isProcessing && processingQuadrants.length > 0) {
    document.body.classList.add("generating");
    processingQuadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else if (!isProcessing) {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items AND create overlays
  // Also create overlays for processing items
  const processingItems = getProcessingItems(status);
  updateQueueOverlays(status.queue || [], processingItems);
}

// Extract currently processing items from status
function getProcessingItems(status) {
  const processingItems = [];
  if (status.queue_by_model) {
    Object.entries(status.queue_by_model).forEach(([modelId, info]) => {
      if (info.is_processing && info.current_item) {
        processingItems.push({
          ...info.current_item,
          model_id: modelId,
          _isProcessing: true,
        });
      }
    });
  }
  return processingItems;
}

// Create/update overlays for pending queue items and processing items
function updateQueueOverlays(queueItems, processingItems = []) {
  // Remove existing overlays
  document.querySelectorAll(".queue-overlay").forEach((el) => el.remove());

  // Also clear queued class from all tiles
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    tile.classList.remove("queued");
  });

  const hasItems =
    (queueItems && queueItems.length > 0) ||
    (processingItems && processingItems.length > 0);
  if (!hasItems) return;

  const grid = document.querySelector(".grid");
  if (!grid) return;

  // Get grid dimensions from config
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Helper function to create an overlay for an item
  function createOverlay(item, options = {}) {
    const { isProcessing = false, queuePosition = null } = options;

    if (!item.quadrants || item.quadrants.length === 0) return null;

    // Calculate bounding box for this generation
    const quadrants = item.quadrants;
    let minCol = Infinity,
      maxCol = -Infinity;
    let minRow = Infinity,
      maxRow = -Infinity;

    // Track which tiles are visible in the current view
    const visibleQuadrants = [];

    quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        visibleQuadrants.push([qx, qy]);
        const col = qx - gridX;
        const row = qy - gridY;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);

        // Add appropriate class to tile
        if (isProcessing) {
          tile.classList.add("locked");
        } else if (!tile.classList.contains("locked")) {
          tile.classList.add("queued");
        }
      }
    });

    if (visibleQuadrants.length === 0) return null;

    // Create overlay element
    const overlay = document.createElement("div");
    overlay.className = isProcessing
      ? "queue-overlay processing"
      : "queue-overlay";
    overlay.dataset.itemId = item.id;

    // Calculate position and size
    const left = minCol * (sizePx + gap);
    const top = minRow * (sizePx + gap);
    const width = (maxCol - minCol + 1) * sizePx + (maxCol - minCol) * gap;
    const height = (maxRow - minRow + 1) * sizePx + (maxRow - minRow) * gap;

    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    // Badge: spinner for processing, number for queued
    const badge = document.createElement("div");
    badge.className = isProcessing ? "queue-badge processing" : "queue-badge";
    if (isProcessing) {
      badge.innerHTML = `<svg class="processing-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>`;
    } else {
      badge.textContent = queuePosition;
    }
    overlay.appendChild(badge);

    // Model name label
    if (item.model_id) {
      const modelLabel = document.createElement("div");
      modelLabel.className = "queue-model-label";
      modelLabel.textContent =
        getModelDisplayName(item.model_id) || item.model_id;
      overlay.appendChild(modelLabel);
    }

    // Cancel button (visible on hover) - for both processing and queued
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "queue-cancel-btn";
    cancelBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>`;
    cancelBtn.title = isProcessing
      ? "Cancel this generation (in progress)"
      : "Cancel this generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelQueueItem(item.id);
    };
    overlay.appendChild(cancelBtn);

    return overlay;
  }

  // First, create overlays for processing items (with spinner)
  processingItems.forEach((item) => {
    const overlay = createOverlay(item, { isProcessing: true });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });

  // Then, create overlays for queued items (with position numbers)
  queueItems.forEach((item, index) => {
    const overlay = createOverlay(item, { queuePosition: index + 1 });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });
}

// Cancel a specific queue item
async function cancelQueueItem(itemId) {
  try {
    const response = await fetch(`/api/queue/cancel/${itemId}`, {
      method: "POST",
    });
    const result = await response.json();

    if (result.success && result.cancelled) {
      showToast("success", "Cancelled", result.message);
      // Trigger immediate status update
      checkGenerationStatus();
    } else if (result.success && !result.cancelled) {
      showToast("info", "Not found", result.message);
    } else {
      showToast("error", "Error", result.error || "Failed to cancel");
    }
  } catch (error) {
    console.error("Cancel queue item failed:", error);
    showToast("error", "Error", "Failed to cancel queue item");
  }
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "d":
    case "D":
      document.getElementById("showRender").click();
      break;
    case "g":
    case "G":
      generateSelected();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate other tools if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
    saveSelectedTool("select");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
    saveSelectedTool("");
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate other tools if active
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
    saveSelectedTool("fixwater");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast(
    "loading",
    "Filling with water...",
    `Processing quadrant (${qx}, ${qy})`
  );

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus(serverStatus = null) {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");
  const generateRectBtn = document.getElementById("generateRectBtn");

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    // Show all active models generating
    const activeModels = serverStatus.active_models || [];
    const processingQuadrants =
      serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

    if (activeModels.length > 0 && processingQuadrants.length > 0) {
      // Show which models are actively generating with their queue counts
      const queueByModel = serverStatus.queue_by_model || {};

      if (activeModels.length === 1) {
        const modelId = activeModels[0];
        const modelName = getModelDisplayName(modelId) || modelId || "default";
        const modelInfo = queueByModel[modelId];
        const queueCount = modelInfo
          ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
          : 0;
        const coordsStr = processingQuadrants
          .map(([x, y]) => `(${x},${y})`)
          .join(" ");
        const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
        statusParts.push(`🔄 ${modelName}${countStr}: ${coordsStr}`);
      } else {
        // Multiple models generating in parallel - show each with queue count
        const modelParts = activeModels.map((modelId) => {
          const name = getModelDisplayName(modelId) || modelId || "default";
          const modelInfo = queueByModel[modelId];
          const queueCount = modelInfo
            ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
            : 0;
          return queueCount > 0 ? `${name} [${queueCount}]` : name;
        });
        statusParts.push(`🔄 ${modelParts.join(", ")}`);
      }
    } else if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      // Fallback to old behavior
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsStr = serverStatus.quadrants
        .map(([x, y]) => `(${x},${y})`)
        .join(" ");
      statusParts.push(`${action} ${coordsStr}`);
    }

    // Show per-model queue counts (only models with items)
    if (serverStatus.queue_by_model) {
      const modelQueues = Object.entries(serverStatus.queue_by_model);
      const queueParts = modelQueues
        .map(([modelId, info]) => {
          const name = getModelDisplayName(modelId) || modelId;
          const count = info.pending_count + (info.is_processing ? 1 : 0);
          return { name, count };
        })
        .filter(({ count }) => count > 0)
        .map(({ name, count }) => `${name}: ${count}`);

      if (queueParts.length > 0) {
        statusParts.push(`📋 ${queueParts.join(", ")}`);
      }
    }
  }

  // Update selection display in toolbar-info (separate from status)
  const selectedDisplay = document.getElementById("selectedQuadrantsDisplay");
  if (selectedDisplay) {
    if (count > 0) {
      const coordsStr = Array.from(selectedQuadrants)
        .map((key) => {
          const [x, y] = key.split(",");
          return `(${x},${y})`;
        })
        .join(" ");
      selectedDisplay.textContent = `✓ ${coordsStr}`;
      selectedDisplay.style.display = "";
    } else {
      selectedDisplay.textContent = "";
      selectedDisplay.style.display = "none";
    }
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = `${count} selected`;
  }

  countEl.textContent = statusText;

  // Update status bar styling
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (isProcessing) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate Rectangle requires exactly 2 selected
  generateRectBtn.disabled = count !== 2;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state (tracked from server)
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function clearQueue() {
  // Get current queue info first
  try {
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();

    const queueLength = status.queue_length || 0;
    const isGenerating = status.is_generating || false;

    if (queueLength === 0 && !isGenerating) {
      showToast(
        "info",
        "Nothing to clear",
        "There are no pending items or active generations."
      );
      return;
    }

    // Build confirm message based on what will be cancelled
    let confirmMessage = "Are you sure you want to clear the queue?";
    const parts = [];
    if (queueLength > 0) {
      parts.push(`${queueLength} pending item(s)`);
    }
    if (isGenerating) {
      parts.push("the current generation in progress");
    }
    if (parts.length > 0) {
      confirmMessage = `Are you sure you want to cancel ${parts.join(
        " and "
      )}?`;
    }

    if (!confirm(confirmMessage)) {
      return;
    }

    // Clear the queue
    const response = await fetch("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        "Queue cleared",
        result.message || `Cleared ${result.cleared_count} item(s)`
      );
      // Refresh status
      await checkGenerationStatus();
    } else {
      showToast(
        "error",
        "Failed to clear queue",
        result.error || "Unknown error"
      );
    }
  } catch (error) {
    console.error("Clear queue error:", error);
    showToast("error", "Request failed", error.message);
  }
}

// Prompt dialog functions
function showPromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("promptDialog");
  const input = document.getElementById("promptInput");
  if (dialog && input) {
    input.value = "";
    dialog.style.display = "flex";
    input.focus();
  }
}

function hidePromptDialog() {
  const dialog = document.getElementById("promptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitPromptGeneration() {
  const input = document.getElementById("promptInput");
  const prompt = input ? input.value.trim() : "";
  hidePromptDialog();
  await generateSelected(prompt);
}

async function generateSelected(prompt = null) {
  if (selectedQuadrants.size === 0) return;

  // Separate selected quadrants into those that need generation vs those that already have it
  const toGenerate = [];
  const contextQuadrants = [];

  Array.from(selectedQuadrants).forEach((s) => {
    const [x, y] = s.split(",").map(Number);
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

    if (tile && !tile.classList.contains("placeholder")) {
      // This quadrant already has a generation - use as context
      contextQuadrants.push([x, y]);
    } else {
      // This quadrant needs generation
      toGenerate.push([x, y]);
    }
  });

  // If nothing needs generation, inform the user
  if (toGenerate.length === 0) {
    showToast(
      "info",
      "Already generated",
      "All selected quadrants already have generations. Select at least one empty quadrant."
    );
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback with context info
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${toGenerate.length} quadrant(s) for generation${promptMsg}${contextMsg}`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const requestBody = {
      quadrants: toGenerate,
      model_id: modelId,
    };

    // Include context quadrants if any
    if (contextQuadrants.length > 0) {
      requestBody.context = contextQuadrants;
    }

    // Include prompt if provided
    if (prompt) {
      requestBody.prompt = prompt;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.queued) {
      console.log(
        "Generation queued at position:",
        result.position,
        "model:",
        result.model_id
      );
      // Get model name for display
      const modelName = getModelDisplayName(result.model_id);
      const modelInfo = modelName ? ` (${modelName})` : "";
      showToast(
        "success",
        "Added to queue",
        `Queued at position ${result.position}${modelInfo}`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for render`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Render queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function generateRectangle() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the rectangle corners."
    );
    return;
  }

  // Get the two selected coordinates
  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return { x, y };
  });

  // Calculate rectangle bounds (top-left and bottom-right)
  const minX = Math.min(coords[0].x, coords[1].x);
  const maxX = Math.max(coords[0].x, coords[1].x);
  const minY = Math.min(coords[0].y, coords[1].y);
  const maxY = Math.max(coords[0].y, coords[1].y);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const totalQuadrants = width * height;

  // Build confirmation message
  const confirmMessage =
    `Generate rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY})?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `This will create a generation plan and queue all steps.\n` +
    `Pre-existing generations will be skipped.`;

  if (!confirm(confirmMessage)) {
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate rectangle requested:",
    { tl: [minX, minY], br: [maxX, maxY] },
    "with model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("generateRectBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Queueing...<span class="spinner"></span>';

  showToast(
    "loading",
    "Creating generation plan...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY})`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        model_id: modelId,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";

    if (result.success) {
      if (result.queued_count === 0) {
        showToast(
          "info",
          "Nothing to generate",
          result.message || "All quadrants already generated."
        );
      } else {
        console.log("Rectangle generation queued:", result);
        const summary = result.plan_summary || {};
        const stepTypes = summary.steps_by_type || {};
        const typeInfo = Object.entries(stepTypes)
          .map(([type, count]) => `${count}× ${type}`)
          .join(", ");

        showToast(
          "success",
          "Rectangle queued!",
          `${result.queued_count} step(s) for ${
            summary.total_quadrants || "?"
          } quadrant(s)` + (typeInfo ? ` (${typeInfo})` : "")
        );
      }
    } else {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    clearLoadingToasts();
    console.error("Generate rectangle error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  saveSelectedQuadrants();
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  if (
    tileEl.classList.contains("locked") ||
    tileEl.classList.contains("queued")
  ) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  saveSelectedQuadrants();
  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;
let lastStatus = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Track state changes - consider active_model_count for parallel processing
    const wasGenerating = isGenerating || isRendering;
    const hasActiveModels = (status.active_model_count || 0) > 0;
    isGenerating =
      (status.is_generating || hasActiveModels) &&
      status.status !== "rendering";
    isRendering = status.is_generating && status.status === "rendering";
    const nowProcessing = isGenerating || isRendering;

    // Apply visual styles based on server status
    applyStatusStyles(status);

    // Update render button based on state
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on render button only (generate state is shown in toolbar)
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      } else {
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
      }

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        const opName = isRendering ? "Render" : "Generation";
        showToast(
          "loading",
          `${opName} in progress...`,
          status.message || "Please wait..."
        );
      }

      // Update the loading toast message
      const loadingToast = document.querySelector(
        ".toast.loading .toast-message"
      );
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset render button
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        // Build per-model queue message
        let queueMsg = "";
        if (status.queue_by_model) {
          const parts = Object.entries(status.queue_by_model)
            .map(([modelId, info]) => {
              const name = getModelDisplayName(modelId) || modelId;
              const count = info.pending_count + (info.is_processing ? 1 : 0);
              return { name, count };
            })
            .filter(({ count }) => count > 0)
            .map(({ name, count }) => `${name}: ${count}`);
          if (parts.length > 0) {
            queueMsg = parts.join(", ");
          }
        }
        if (queueMsg) {
          showToast("info", "Processing queue", queueMsg);
        }
      } else {
        // No more items, reload after delay
        stopStatusPolling();
        setTimeout(() => window.location.reload(), 1500);
      }
    } else if (status.status === "error" && status.error) {
      clearLoadingToasts();
      showToast("error", "Error", status.error);

      // Continue polling if there are more items in queue
      if (status.queue_length === 0) {
        stopStatusPolling();
      }
    } else if (
      status.status === "idle" &&
      status.queue_length === 0 &&
      !nowProcessing
    ) {
      // Idle with no queue - stop polling
      stopStatusPolling();
    }

    // Update selection status with server info
    updateSelectionStatus(status);
    lastStatus = status;
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

// Restore saved tool on page load
function restoreSavedTool() {
  const savedTool = getSavedTool();
  if (!savedTool) return;

  // Check if the tool button exists before activating
  switch (savedTool) {
    case "select":
      if (document.getElementById("selectTool")) {
        toggleSelectTool();
      }
      break;
    case "fixwater":
      if (document.getElementById("fixWaterTool")) {
        toggleFixWaterTool();
      }
      break;
    case "waterfill":
      if (document.getElementById("waterFillTool")) {
        toggleWaterFillTool();
      }
      break;
    default:
      // Unknown tool, clear saved state
      saveSelectedTool("");
      break;
  }
}

// Restore saved quadrant selections on page load
function restoreSavedQuadrants() {
  const savedQuadrants = getSavedQuadrants();
  if (!savedQuadrants || savedQuadrants.length === 0) return;

  let restoredCount = 0;

  savedQuadrants.forEach((key) => {
    // Check if this quadrant tile exists on the current page
    const tile = document.querySelector(`.tile[data-coords="${key}"]`);
    if (tile) {
      // Don't restore if tile is locked or queued
      if (
        !tile.classList.contains("locked") &&
        !tile.classList.contains("queued")
      ) {
        selectedQuadrants.add(key);
        tile.classList.add("selected");
        restoredCount++;
      }
    }
  });

  if (restoredCount > 0) {
    console.log(`Restored ${restoredCount} selected quadrant(s)`);
    // Update localStorage to only contain valid selections
    saveSelectedQuadrants();
    updateSelectionStatus();
  } else if (savedQuadrants.length > 0) {
    // Had saved selections but none are on current page - clear storage
    saveSelectedQuadrants();
  }
}

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

  // Restore saved quadrant selections
  restoreSavedQuadrants();

  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Apply initial status styles
    applyStatusStyles(status);

    if (status.is_generating || status.queue_length > 0) {
      console.log(
        "Processing in progress or queue non-empty, starting polling..."
      );
      isGenerating = status.is_generating && status.status !== "rendering";
      isRendering = status.is_generating && status.status === "rendering";
      startStatusPolling();
    }

    updateSelectionStatus(status);
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Done! Now the generation quadrant display will show per-model queue counts directly:

- **Single model generating**: `🔄 ModelName [3]: (x,y) (x2,y2)` - where `[3]` is the queue count for that model
- **Multiple models generating**: `🔄 ModelA [3], ModelB [2]` - with per-model queue counts in brackets

The queue count includes both pending items and the currently processing item for each model. If a model has 0 items, the count is omitted.

---

**User**

no i mean in the purple outlined tiles - they need to show which queue position the generation is for each model not in total

---

**Cursor**

Ah, I understand now - you want the purple queue overlay badges to show the position within each model's queue, not the overall queue position. Let me look at the `updateQueueOverlays` function:

I see the issue. The current code at line 306-307 uses the overall index as the queue position. I need to change it to show the position within each model's queue:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";

// Save selected model ID to localStorage
function saveSelectedModel(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY_MODEL, modelId);
  } catch (e) {
    console.warn("Could not save model to localStorage:", e);
  }
}

// Get saved model ID from localStorage
function getSavedModel() {
  try {
    return localStorage.getItem(STORAGE_KEY_MODEL);
  } catch (e) {
    return null;
  }
}

// Save selected tool to localStorage
function saveSelectedTool(toolName) {
  try {
    localStorage.setItem(STORAGE_KEY_TOOL, toolName || "");
  } catch (e) {
    console.warn("Could not save tool to localStorage:", e);
  }
}

// Get saved tool from localStorage
function getSavedTool() {
  try {
    return localStorage.getItem(STORAGE_KEY_TOOL) || "";
  } catch (e) {
    return "";
  }
}

// Save selected quadrants to localStorage
function saveSelectedQuadrants() {
  try {
    const quadrantsArray = Array.from(selectedQuadrants);
    localStorage.setItem(STORAGE_KEY_SELECTION, JSON.stringify(quadrantsArray));
  } catch (e) {
    console.warn("Could not save selection to localStorage:", e);
  }
}

// Get saved quadrants from localStorage
function getSavedQuadrants() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTION);
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    return [];
  }
}

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Check if saved model ID exists in available models
  const savedModelId = getSavedModel();
  const savedModelExists =
    savedModelId && config.models.some((m) => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;

    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (
      !savedModelExists &&
      model.model_id === config.default_model_id
    ) {
      option.selected = true;
    } else if (!savedModelExists && !config.default_model_id && index === 0) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection and save to localStorage
  select.addEventListener("change", () => {
    saveSelectedModel(select.value);
    select.blur();
  });
}

// Get the currently selected model ID
function getSelectedModelId() {
  const select = document.getElementById("modelSelect");
  return select ? select.value : null;
}

// Get display name for a model ID
function getModelDisplayName(modelId) {
  if (!modelId) return null;
  const configEl = document.getElementById("app-config");
  if (!configEl) return modelId;
  try {
    const config = JSON.parse(configEl.dataset.config);
    const models = config.models || [];
    const model = models.find((m) => m.model_id === modelId);
    return model ? model.name : modelId;
  } catch {
    return modelId;
  }
}

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to ALL currently processing quadrants (from all models)
  const processingQuadrants =
    status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || status.active_model_count > 0;

  if (isProcessing && processingQuadrants.length > 0) {
    document.body.classList.add("generating");
    processingQuadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else if (!isProcessing) {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items AND create overlays
  // Also create overlays for processing items
  const processingItems = getProcessingItems(status);
  updateQueueOverlays(status.queue || [], processingItems);
}

// Extract currently processing items from status
function getProcessingItems(status) {
  const processingItems = [];
  if (status.queue_by_model) {
    Object.entries(status.queue_by_model).forEach(([modelId, info]) => {
      if (info.is_processing && info.current_item) {
        processingItems.push({
          ...info.current_item,
          model_id: modelId,
          _isProcessing: true,
        });
      }
    });
  }
  return processingItems;
}

// Create/update overlays for pending queue items and processing items
function updateQueueOverlays(queueItems, processingItems = []) {
  // Remove existing overlays
  document.querySelectorAll(".queue-overlay").forEach((el) => el.remove());

  // Also clear queued class from all tiles
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    tile.classList.remove("queued");
  });

  const hasItems =
    (queueItems && queueItems.length > 0) ||
    (processingItems && processingItems.length > 0);
  if (!hasItems) return;

  const grid = document.querySelector(".grid");
  if (!grid) return;

  // Get grid dimensions from config
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Helper function to create an overlay for an item
  function createOverlay(item, options = {}) {
    const { isProcessing = false, queuePosition = null } = options;

    if (!item.quadrants || item.quadrants.length === 0) return null;

    // Calculate bounding box for this generation
    const quadrants = item.quadrants;
    let minCol = Infinity,
      maxCol = -Infinity;
    let minRow = Infinity,
      maxRow = -Infinity;

    // Track which tiles are visible in the current view
    const visibleQuadrants = [];

    quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        visibleQuadrants.push([qx, qy]);
        const col = qx - gridX;
        const row = qy - gridY;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);

        // Add appropriate class to tile
        if (isProcessing) {
          tile.classList.add("locked");
        } else if (!tile.classList.contains("locked")) {
          tile.classList.add("queued");
        }
      }
    });

    if (visibleQuadrants.length === 0) return null;

    // Create overlay element
    const overlay = document.createElement("div");
    overlay.className = isProcessing
      ? "queue-overlay processing"
      : "queue-overlay";
    overlay.dataset.itemId = item.id;

    // Calculate position and size
    const left = minCol * (sizePx + gap);
    const top = minRow * (sizePx + gap);
    const width = (maxCol - minCol + 1) * sizePx + (maxCol - minCol) * gap;
    const height = (maxRow - minRow + 1) * sizePx + (maxRow - minRow) * gap;

    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    // Badge: spinner for processing, number for queued
    const badge = document.createElement("div");
    badge.className = isProcessing ? "queue-badge processing" : "queue-badge";
    if (isProcessing) {
      badge.innerHTML = `<svg class="processing-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>`;
    } else {
      badge.textContent = queuePosition;
    }
    overlay.appendChild(badge);

    // Model name label
    if (item.model_id) {
      const modelLabel = document.createElement("div");
      modelLabel.className = "queue-model-label";
      modelLabel.textContent =
        getModelDisplayName(item.model_id) || item.model_id;
      overlay.appendChild(modelLabel);
    }

    // Cancel button (visible on hover) - for both processing and queued
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "queue-cancel-btn";
    cancelBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>`;
    cancelBtn.title = isProcessing
      ? "Cancel this generation (in progress)"
      : "Cancel this generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelQueueItem(item.id);
    };
    overlay.appendChild(cancelBtn);

    return overlay;
  }

  // First, create overlays for processing items (with spinner)
  processingItems.forEach((item) => {
    const overlay = createOverlay(item, { isProcessing: true });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });

  // Then, create overlays for queued items (with per-model position numbers)
  // Track position within each model's queue
  const modelPositionCounters = {};

  queueItems.forEach((item) => {
    const modelId = item.model_id || "default";
    // Initialize or increment the counter for this model
    if (!modelPositionCounters[modelId]) {
      modelPositionCounters[modelId] = 1;
    }
    const positionInModelQueue = modelPositionCounters[modelId];
    modelPositionCounters[modelId]++;

    const overlay = createOverlay(item, { queuePosition: positionInModelQueue });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });
}

// Cancel a specific queue item
async function cancelQueueItem(itemId) {
  try {
    const response = await fetch(`/api/queue/cancel/${itemId}`, {
      method: "POST",
    });
    const result = await response.json();

    if (result.success && result.cancelled) {
      showToast("success", "Cancelled", result.message);
      // Trigger immediate status update
      checkGenerationStatus();
    } else if (result.success && !result.cancelled) {
      showToast("info", "Not found", result.message);
    } else {
      showToast("error", "Error", result.error || "Failed to cancel");
    }
  } catch (error) {
    console.error("Cancel queue item failed:", error);
    showToast("error", "Error", "Failed to cancel queue item");
  }
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "d":
    case "D":
      document.getElementById("showRender").click();
      break;
    case "g":
    case "G":
      generateSelected();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate other tools if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
    saveSelectedTool("select");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
    saveSelectedTool("");
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate other tools if active
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
    saveSelectedTool("fixwater");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast(
    "loading",
    "Filling with water...",
    `Processing quadrant (${qx}, ${qy})`
  );

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus(serverStatus = null) {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");
  const generateRectBtn = document.getElementById("generateRectBtn");

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    // Show all active models generating
    const activeModels = serverStatus.active_models || [];
    const processingQuadrants =
      serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

    if (activeModels.length > 0 && processingQuadrants.length > 0) {
      // Show which models are actively generating with their queue counts
      const queueByModel = serverStatus.queue_by_model || {};

      if (activeModels.length === 1) {
        const modelId = activeModels[0];
        const modelName = getModelDisplayName(modelId) || modelId || "default";
        const modelInfo = queueByModel[modelId];
        const queueCount = modelInfo
          ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
          : 0;
        const coordsStr = processingQuadrants
          .map(([x, y]) => `(${x},${y})`)
          .join(" ");
        const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
        statusParts.push(`🔄 ${modelName}${countStr}: ${coordsStr}`);
      } else {
        // Multiple models generating in parallel - show each with queue count
        const modelParts = activeModels.map((modelId) => {
          const name = getModelDisplayName(modelId) || modelId || "default";
          const modelInfo = queueByModel[modelId];
          const queueCount = modelInfo
            ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
            : 0;
          return queueCount > 0 ? `${name} [${queueCount}]` : name;
        });
        statusParts.push(`🔄 ${modelParts.join(", ")}`);
      }
    } else if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      // Fallback to old behavior
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsStr = serverStatus.quadrants
        .map(([x, y]) => `(${x},${y})`)
        .join(" ");
      statusParts.push(`${action} ${coordsStr}`);
    }

    // Show per-model queue counts (only models with items)
    if (serverStatus.queue_by_model) {
      const modelQueues = Object.entries(serverStatus.queue_by_model);
      const queueParts = modelQueues
        .map(([modelId, info]) => {
          const name = getModelDisplayName(modelId) || modelId;
          const count = info.pending_count + (info.is_processing ? 1 : 0);
          return { name, count };
        })
        .filter(({ count }) => count > 0)
        .map(({ name, count }) => `${name}: ${count}`);

      if (queueParts.length > 0) {
        statusParts.push(`📋 ${queueParts.join(", ")}`);
      }
    }
  }

  // Update selection display in toolbar-info (separate from status)
  const selectedDisplay = document.getElementById("selectedQuadrantsDisplay");
  if (selectedDisplay) {
    if (count > 0) {
      const coordsStr = Array.from(selectedQuadrants)
        .map((key) => {
          const [x, y] = key.split(",");
          return `(${x},${y})`;
        })
        .join(" ");
      selectedDisplay.textContent = `✓ ${coordsStr}`;
      selectedDisplay.style.display = "";
    } else {
      selectedDisplay.textContent = "";
      selectedDisplay.style.display = "none";
    }
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = `${count} selected`;
  }

  countEl.textContent = statusText;

  // Update status bar styling
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (isProcessing) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate Rectangle requires exactly 2 selected
  generateRectBtn.disabled = count !== 2;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state (tracked from server)
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function clearQueue() {
  // Get current queue info first
  try {
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();

    const queueLength = status.queue_length || 0;
    const isGenerating = status.is_generating || false;

    if (queueLength === 0 && !isGenerating) {
      showToast(
        "info",
        "Nothing to clear",
        "There are no pending items or active generations."
      );
      return;
    }

    // Build confirm message based on what will be cancelled
    let confirmMessage = "Are you sure you want to clear the queue?";
    const parts = [];
    if (queueLength > 0) {
      parts.push(`${queueLength} pending item(s)`);
    }
    if (isGenerating) {
      parts.push("the current generation in progress");
    }
    if (parts.length > 0) {
      confirmMessage = `Are you sure you want to cancel ${parts.join(
        " and "
      )}?`;
    }

    if (!confirm(confirmMessage)) {
      return;
    }

    // Clear the queue
    const response = await fetch("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        "Queue cleared",
        result.message || `Cleared ${result.cleared_count} item(s)`
      );
      // Refresh status
      await checkGenerationStatus();
    } else {
      showToast(
        "error",
        "Failed to clear queue",
        result.error || "Unknown error"
      );
    }
  } catch (error) {
    console.error("Clear queue error:", error);
    showToast("error", "Request failed", error.message);
  }
}

// Prompt dialog functions
function showPromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("promptDialog");
  const input = document.getElementById("promptInput");
  if (dialog && input) {
    input.value = "";
    dialog.style.display = "flex";
    input.focus();
  }
}

function hidePromptDialog() {
  const dialog = document.getElementById("promptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitPromptGeneration() {
  const input = document.getElementById("promptInput");
  const prompt = input ? input.value.trim() : "";
  hidePromptDialog();
  await generateSelected(prompt);
}

async function generateSelected(prompt = null) {
  if (selectedQuadrants.size === 0) return;

  // Separate selected quadrants into those that need generation vs those that already have it
  const toGenerate = [];
  const contextQuadrants = [];

  Array.from(selectedQuadrants).forEach((s) => {
    const [x, y] = s.split(",").map(Number);
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

    if (tile && !tile.classList.contains("placeholder")) {
      // This quadrant already has a generation - use as context
      contextQuadrants.push([x, y]);
    } else {
      // This quadrant needs generation
      toGenerate.push([x, y]);
    }
  });

  // If nothing needs generation, inform the user
  if (toGenerate.length === 0) {
    showToast(
      "info",
      "Already generated",
      "All selected quadrants already have generations. Select at least one empty quadrant."
    );
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback with context info
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${toGenerate.length} quadrant(s) for generation${promptMsg}${contextMsg}`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const requestBody = {
      quadrants: toGenerate,
      model_id: modelId,
    };

    // Include context quadrants if any
    if (contextQuadrants.length > 0) {
      requestBody.context = contextQuadrants;
    }

    // Include prompt if provided
    if (prompt) {
      requestBody.prompt = prompt;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.queued) {
      console.log(
        "Generation queued at position:",
        result.position,
        "model:",
        result.model_id
      );
      // Get model name for display
      const modelName = getModelDisplayName(result.model_id);
      const modelInfo = modelName ? ` (${modelName})` : "";
      showToast(
        "success",
        "Added to queue",
        `Queued at position ${result.position}${modelInfo}`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for render`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Render queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function generateRectangle() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the rectangle corners."
    );
    return;
  }

  // Get the two selected coordinates
  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return { x, y };
  });

  // Calculate rectangle bounds (top-left and bottom-right)
  const minX = Math.min(coords[0].x, coords[1].x);
  const maxX = Math.max(coords[0].x, coords[1].x);
  const minY = Math.min(coords[0].y, coords[1].y);
  const maxY = Math.max(coords[0].y, coords[1].y);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const totalQuadrants = width * height;

  // Build confirmation message
  const confirmMessage =
    `Generate rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY})?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `This will create a generation plan and queue all steps.\n` +
    `Pre-existing generations will be skipped.`;

  if (!confirm(confirmMessage)) {
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate rectangle requested:",
    { tl: [minX, minY], br: [maxX, maxY] },
    "with model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("generateRectBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Queueing...<span class="spinner"></span>';

  showToast(
    "loading",
    "Creating generation plan...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY})`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        model_id: modelId,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";

    if (result.success) {
      if (result.queued_count === 0) {
        showToast(
          "info",
          "Nothing to generate",
          result.message || "All quadrants already generated."
        );
      } else {
        console.log("Rectangle generation queued:", result);
        const summary = result.plan_summary || {};
        const stepTypes = summary.steps_by_type || {};
        const typeInfo = Object.entries(stepTypes)
          .map(([type, count]) => `${count}× ${type}`)
          .join(", ");

        showToast(
          "success",
          "Rectangle queued!",
          `${result.queued_count} step(s) for ${
            summary.total_quadrants || "?"
          } quadrant(s)` + (typeInfo ? ` (${typeInfo})` : "")
        );
      }
    } else {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    clearLoadingToasts();
    console.error("Generate rectangle error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  saveSelectedQuadrants();
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  if (
    tileEl.classList.contains("locked") ||
    tileEl.classList.contains("queued")
  ) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  saveSelectedQuadrants();
  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;
let lastStatus = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Track state changes - consider active_model_count for parallel processing
    const wasGenerating = isGenerating || isRendering;
    const hasActiveModels = (status.active_model_count || 0) > 0;
    isGenerating =
      (status.is_generating || hasActiveModels) &&
      status.status !== "rendering";
    isRendering = status.is_generating && status.status === "rendering";
    const nowProcessing = isGenerating || isRendering;

    // Apply visual styles based on server status
    applyStatusStyles(status);

    // Update render button based on state
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on render button only (generate state is shown in toolbar)
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      } else {
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
      }

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        const opName = isRendering ? "Render" : "Generation";
        showToast(
          "loading",
          `${opName} in progress...`,
          status.message || "Please wait..."
        );
      }

      // Update the loading toast message
      const loadingToast = document.querySelector(
        ".toast.loading .toast-message"
      );
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset render button
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        // Build per-model queue message
        let queueMsg = "";
        if (status.queue_by_model) {
          const parts = Object.entries(status.queue_by_model)
            .map(([modelId, info]) => {
              const name = getModelDisplayName(modelId) || modelId;
              const count = info.pending_count + (info.is_processing ? 1 : 0);
              return { name, count };
            })
            .filter(({ count }) => count > 0)
            .map(({ name, count }) => `${name}: ${count}`);
          if (parts.length > 0) {
            queueMsg = parts.join(", ");
          }
        }
        if (queueMsg) {
          showToast("info", "Processing queue", queueMsg);
        }
      } else {
        // No more items, reload after delay
        stopStatusPolling();
        setTimeout(() => window.location.reload(), 1500);
      }
    } else if (status.status === "error" && status.error) {
      clearLoadingToasts();
      showToast("error", "Error", status.error);

      // Continue polling if there are more items in queue
      if (status.queue_length === 0) {
        stopStatusPolling();
      }
    } else if (
      status.status === "idle" &&
      status.queue_length === 0 &&
      !nowProcessing
    ) {
      // Idle with no queue - stop polling
      stopStatusPolling();
    }

    // Update selection status with server info
    updateSelectionStatus(status);
    lastStatus = status;
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

// Restore saved tool on page load
function restoreSavedTool() {
  const savedTool = getSavedTool();
  if (!savedTool) return;

  // Check if the tool button exists before activating
  switch (savedTool) {
    case "select":
      if (document.getElementById("selectTool")) {
        toggleSelectTool();
      }
      break;
    case "fixwater":
      if (document.getElementById("fixWaterTool")) {
        toggleFixWaterTool();
      }
      break;
    case "waterfill":
      if (document.getElementById("waterFillTool")) {
        toggleWaterFillTool();
      }
      break;
    default:
      // Unknown tool, clear saved state
      saveSelectedTool("");
      break;
  }
}

// Restore saved quadrant selections on page load
function restoreSavedQuadrants() {
  const savedQuadrants = getSavedQuadrants();
  if (!savedQuadrants || savedQuadrants.length === 0) return;

  let restoredCount = 0;

  savedQuadrants.forEach((key) => {
    // Check if this quadrant tile exists on the current page
    const tile = document.querySelector(`.tile[data-coords="${key}"]`);
    if (tile) {
      // Don't restore if tile is locked or queued
      if (
        !tile.classList.contains("locked") &&
        !tile.classList.contains("queued")
      ) {
        selectedQuadrants.add(key);
        tile.classList.add("selected");
        restoredCount++;
      }
    }
  });

  if (restoredCount > 0) {
    console.log(`Restored ${restoredCount} selected quadrant(s)`);
    // Update localStorage to only contain valid selections
    saveSelectedQuadrants();
    updateSelectionStatus();
  } else if (savedQuadrants.length > 0) {
    // Had saved selections but none are on current page - clear storage
    saveSelectedQuadrants();
  }
}

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

  // Restore saved quadrant selections
  restoreSavedQuadrants();

  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Apply initial status styles
    applyStatusStyles(status);

    if (status.is_generating || status.queue_length > 0) {
      console.log(
        "Processing in progress or queue non-empty, starting polling..."
      );
      isGenerating = status.is_generating && status.status !== "rendering";
      isRendering = status.is_generating && status.status === "rendering";
      startStatusPolling();
    }

    updateSelectionStatus(status);
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Done! Now the purple outlined queue overlays will show the position within each model's queue, not the overall total position.

For example, if you have items queued for different models:
- ModelA item → shows **1** (first in ModelA's queue)
- ModelB item → shows **1** (first in ModelB's queue)
- ModelA item → shows **2** (second in ModelA's queue)
- ModelB item → shows **2** (second in ModelB's queue)

Previously these would have shown 1, 2, 3, 4 (total position). Now each model's queue positions are tracked independently.

---

**User**

ok great now if i click on the staus bar indicator for the current model generation, link to that xy coordinate

---

**Cursor**

I'll update the status bar to make the coordinate display clickable, linking to that location. Let me look at how the status is rendered and add click handling:

I need to change the status display to use HTML with clickable coordinate links. Let me update the `updateSelectionStatus` function:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";

// Save selected model ID to localStorage
function saveSelectedModel(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY_MODEL, modelId);
  } catch (e) {
    console.warn("Could not save model to localStorage:", e);
  }
}

// Get saved model ID from localStorage
function getSavedModel() {
  try {
    return localStorage.getItem(STORAGE_KEY_MODEL);
  } catch (e) {
    return null;
  }
}

// Save selected tool to localStorage
function saveSelectedTool(toolName) {
  try {
    localStorage.setItem(STORAGE_KEY_TOOL, toolName || "");
  } catch (e) {
    console.warn("Could not save tool to localStorage:", e);
  }
}

// Get saved tool from localStorage
function getSavedTool() {
  try {
    return localStorage.getItem(STORAGE_KEY_TOOL) || "";
  } catch (e) {
    return "";
  }
}

// Save selected quadrants to localStorage
function saveSelectedQuadrants() {
  try {
    const quadrantsArray = Array.from(selectedQuadrants);
    localStorage.setItem(STORAGE_KEY_SELECTION, JSON.stringify(quadrantsArray));
  } catch (e) {
    console.warn("Could not save selection to localStorage:", e);
  }
}

// Get saved quadrants from localStorage
function getSavedQuadrants() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTION);
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    return [];
  }
}

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Check if saved model ID exists in available models
  const savedModelId = getSavedModel();
  const savedModelExists =
    savedModelId && config.models.some((m) => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;

    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (
      !savedModelExists &&
      model.model_id === config.default_model_id
    ) {
      option.selected = true;
    } else if (!savedModelExists && !config.default_model_id && index === 0) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection and save to localStorage
  select.addEventListener("change", () => {
    saveSelectedModel(select.value);
    select.blur();
  });
}

// Get the currently selected model ID
function getSelectedModelId() {
  const select = document.getElementById("modelSelect");
  return select ? select.value : null;
}

// Get display name for a model ID
function getModelDisplayName(modelId) {
  if (!modelId) return null;
  const configEl = document.getElementById("app-config");
  if (!configEl) return modelId;
  try {
    const config = JSON.parse(configEl.dataset.config);
    const models = config.models || [];
    const model = models.find((m) => m.model_id === modelId);
    return model ? model.name : modelId;
  } catch {
    return modelId;
  }
}

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to ALL currently processing quadrants (from all models)
  const processingQuadrants =
    status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || status.active_model_count > 0;

  if (isProcessing && processingQuadrants.length > 0) {
    document.body.classList.add("generating");
    processingQuadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else if (!isProcessing) {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items AND create overlays
  // Also create overlays for processing items
  const processingItems = getProcessingItems(status);
  updateQueueOverlays(status.queue || [], processingItems);
}

// Extract currently processing items from status
function getProcessingItems(status) {
  const processingItems = [];
  if (status.queue_by_model) {
    Object.entries(status.queue_by_model).forEach(([modelId, info]) => {
      if (info.is_processing && info.current_item) {
        processingItems.push({
          ...info.current_item,
          model_id: modelId,
          _isProcessing: true,
        });
      }
    });
  }
  return processingItems;
}

// Create/update overlays for pending queue items and processing items
function updateQueueOverlays(queueItems, processingItems = []) {
  // Remove existing overlays
  document.querySelectorAll(".queue-overlay").forEach((el) => el.remove());

  // Also clear queued class from all tiles
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    tile.classList.remove("queued");
  });

  const hasItems =
    (queueItems && queueItems.length > 0) ||
    (processingItems && processingItems.length > 0);
  if (!hasItems) return;

  const grid = document.querySelector(".grid");
  if (!grid) return;

  // Get grid dimensions from config
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Helper function to create an overlay for an item
  function createOverlay(item, options = {}) {
    const { isProcessing = false, queuePosition = null } = options;

    if (!item.quadrants || item.quadrants.length === 0) return null;

    // Calculate bounding box for this generation
    const quadrants = item.quadrants;
    let minCol = Infinity,
      maxCol = -Infinity;
    let minRow = Infinity,
      maxRow = -Infinity;

    // Track which tiles are visible in the current view
    const visibleQuadrants = [];

    quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        visibleQuadrants.push([qx, qy]);
        const col = qx - gridX;
        const row = qy - gridY;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);

        // Add appropriate class to tile
        if (isProcessing) {
          tile.classList.add("locked");
        } else if (!tile.classList.contains("locked")) {
          tile.classList.add("queued");
        }
      }
    });

    if (visibleQuadrants.length === 0) return null;

    // Create overlay element
    const overlay = document.createElement("div");
    overlay.className = isProcessing
      ? "queue-overlay processing"
      : "queue-overlay";
    overlay.dataset.itemId = item.id;

    // Calculate position and size
    const left = minCol * (sizePx + gap);
    const top = minRow * (sizePx + gap);
    const width = (maxCol - minCol + 1) * sizePx + (maxCol - minCol) * gap;
    const height = (maxRow - minRow + 1) * sizePx + (maxRow - minRow) * gap;

    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    // Badge: spinner for processing, number for queued
    const badge = document.createElement("div");
    badge.className = isProcessing ? "queue-badge processing" : "queue-badge";
    if (isProcessing) {
      badge.innerHTML = `<svg class="processing-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>`;
    } else {
      badge.textContent = queuePosition;
    }
    overlay.appendChild(badge);

    // Model name label
    if (item.model_id) {
      const modelLabel = document.createElement("div");
      modelLabel.className = "queue-model-label";
      modelLabel.textContent =
        getModelDisplayName(item.model_id) || item.model_id;
      overlay.appendChild(modelLabel);
    }

    // Cancel button (visible on hover) - for both processing and queued
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "queue-cancel-btn";
    cancelBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>`;
    cancelBtn.title = isProcessing
      ? "Cancel this generation (in progress)"
      : "Cancel this generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelQueueItem(item.id);
    };
    overlay.appendChild(cancelBtn);

    return overlay;
  }

  // First, create overlays for processing items (with spinner)
  processingItems.forEach((item) => {
    const overlay = createOverlay(item, { isProcessing: true });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });

  // Then, create overlays for queued items (with per-model position numbers)
  // Track position within each model's queue
  const modelPositionCounters = {};

  queueItems.forEach((item) => {
    const modelId = item.model_id || "default";
    // Initialize or increment the counter for this model
    if (!modelPositionCounters[modelId]) {
      modelPositionCounters[modelId] = 1;
    }
    const positionInModelQueue = modelPositionCounters[modelId];
    modelPositionCounters[modelId]++;

    const overlay = createOverlay(item, {
      queuePosition: positionInModelQueue,
    });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });
}

// Cancel a specific queue item
async function cancelQueueItem(itemId) {
  try {
    const response = await fetch(`/api/queue/cancel/${itemId}`, {
      method: "POST",
    });
    const result = await response.json();

    if (result.success && result.cancelled) {
      showToast("success", "Cancelled", result.message);
      // Trigger immediate status update
      checkGenerationStatus();
    } else if (result.success && !result.cancelled) {
      showToast("info", "Not found", result.message);
    } else {
      showToast("error", "Error", result.error || "Failed to cancel");
    }
  } catch (error) {
    console.error("Cancel queue item failed:", error);
    showToast("error", "Error", "Failed to cancel queue item");
  }
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "d":
    case "D":
      document.getElementById("showRender").click();
      break;
    case "g":
    case "G":
      generateSelected();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate other tools if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
    saveSelectedTool("select");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
    saveSelectedTool("");
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate other tools if active
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
    saveSelectedTool("fixwater");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast(
    "loading",
    "Filling with water...",
    `Processing quadrant (${qx}, ${qy})`
  );

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus(serverStatus = null) {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");
  const generateRectBtn = document.getElementById("generateRectBtn");

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    // Show all active models generating
    const activeModels = serverStatus.active_models || [];
    const processingQuadrants =
      serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

    if (activeModels.length > 0 && processingQuadrants.length > 0) {
      // Show which models are actively generating with their queue counts
      const queueByModel = serverStatus.queue_by_model || {};

      if (activeModels.length === 1) {
        const modelId = activeModels[0];
        const modelName = getModelDisplayName(modelId) || modelId || "default";
        const modelInfo = queueByModel[modelId];
        const queueCount = modelInfo
          ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
          : 0;
        // Create clickable coordinate links
        const coordsHtml = processingQuadrants
          .map(
            ([x, y]) =>
              `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
          )
          .join(" ");
        const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
        statusParts.push({
          html: `🔄 ${modelName}${countStr}: ${coordsHtml}`,
        });
      } else {
        // Multiple models generating in parallel - show each with queue count and coords
        const queueByModel = serverStatus.queue_by_model || {};
        const modelPartsHtml = activeModels.map((modelId) => {
          const name = getModelDisplayName(modelId) || modelId || "default";
          const modelInfo = queueByModel[modelId];
          const queueCount = modelInfo
            ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
            : 0;
          // Get quadrants for this specific model
          const modelQuadrants =
            modelInfo && modelInfo.current_item
              ? modelInfo.current_item.quadrants || []
              : [];
          const coordsHtml =
            modelQuadrants.length > 0
              ? " " +
                modelQuadrants
                  .map(
                    ([x, y]) =>
                      `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
                  )
                  .join(" ")
              : "";
          const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
          return `${name}${countStr}${coordsHtml}`;
        });
        statusParts.push({ html: `🔄 ${modelPartsHtml.join(", ")}` });
      }
    } else if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      // Fallback to old behavior with clickable coords
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsHtml = serverStatus.quadrants
        .map(
          ([x, y]) =>
            `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
        )
        .join(" ");
      statusParts.push({ html: `${action} ${coordsHtml}` });
    }

    // Show per-model queue counts (only models with items)
    if (serverStatus.queue_by_model) {
      const modelQueues = Object.entries(serverStatus.queue_by_model);
      const queueParts = modelQueues
        .map(([modelId, info]) => {
          const name = getModelDisplayName(modelId) || modelId;
          const count = info.pending_count + (info.is_processing ? 1 : 0);
          return { name, count };
        })
        .filter(({ count }) => count > 0)
        .map(({ name, count }) => `${name}: ${count}`);

      if (queueParts.length > 0) {
        statusParts.push(`📋 ${queueParts.join(", ")}`);
      }
    }
  }

  // Update selection display in toolbar-info (separate from status)
  const selectedDisplay = document.getElementById("selectedQuadrantsDisplay");
  if (selectedDisplay) {
    if (count > 0) {
      const coordsStr = Array.from(selectedQuadrants)
        .map((key) => {
          const [x, y] = key.split(",");
          return `(${x},${y})`;
        })
        .join(" ");
      selectedDisplay.textContent = `✓ ${coordsStr}`;
      selectedDisplay.style.display = "";
    } else {
      selectedDisplay.textContent = "";
      selectedDisplay.style.display = "none";
    }
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = `${count} selected`;
  }

  countEl.textContent = statusText;

  // Update status bar styling
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (isProcessing) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate Rectangle requires exactly 2 selected
  generateRectBtn.disabled = count !== 2;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state (tracked from server)
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function clearQueue() {
  // Get current queue info first
  try {
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();

    const queueLength = status.queue_length || 0;
    const isGenerating = status.is_generating || false;

    if (queueLength === 0 && !isGenerating) {
      showToast(
        "info",
        "Nothing to clear",
        "There are no pending items or active generations."
      );
      return;
    }

    // Build confirm message based on what will be cancelled
    let confirmMessage = "Are you sure you want to clear the queue?";
    const parts = [];
    if (queueLength > 0) {
      parts.push(`${queueLength} pending item(s)`);
    }
    if (isGenerating) {
      parts.push("the current generation in progress");
    }
    if (parts.length > 0) {
      confirmMessage = `Are you sure you want to cancel ${parts.join(
        " and "
      )}?`;
    }

    if (!confirm(confirmMessage)) {
      return;
    }

    // Clear the queue
    const response = await fetch("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        "Queue cleared",
        result.message || `Cleared ${result.cleared_count} item(s)`
      );
      // Refresh status
      await checkGenerationStatus();
    } else {
      showToast(
        "error",
        "Failed to clear queue",
        result.error || "Unknown error"
      );
    }
  } catch (error) {
    console.error("Clear queue error:", error);
    showToast("error", "Request failed", error.message);
  }
}

// Prompt dialog functions
function showPromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("promptDialog");
  const input = document.getElementById("promptInput");
  if (dialog && input) {
    input.value = "";
    dialog.style.display = "flex";
    input.focus();
  }
}

function hidePromptDialog() {
  const dialog = document.getElementById("promptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitPromptGeneration() {
  const input = document.getElementById("promptInput");
  const prompt = input ? input.value.trim() : "";
  hidePromptDialog();
  await generateSelected(prompt);
}

async function generateSelected(prompt = null) {
  if (selectedQuadrants.size === 0) return;

  // Separate selected quadrants into those that need generation vs those that already have it
  const toGenerate = [];
  const contextQuadrants = [];

  Array.from(selectedQuadrants).forEach((s) => {
    const [x, y] = s.split(",").map(Number);
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

    if (tile && !tile.classList.contains("placeholder")) {
      // This quadrant already has a generation - use as context
      contextQuadrants.push([x, y]);
    } else {
      // This quadrant needs generation
      toGenerate.push([x, y]);
    }
  });

  // If nothing needs generation, inform the user
  if (toGenerate.length === 0) {
    showToast(
      "info",
      "Already generated",
      "All selected quadrants already have generations. Select at least one empty quadrant."
    );
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback with context info
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${toGenerate.length} quadrant(s) for generation${promptMsg}${contextMsg}`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const requestBody = {
      quadrants: toGenerate,
      model_id: modelId,
    };

    // Include context quadrants if any
    if (contextQuadrants.length > 0) {
      requestBody.context = contextQuadrants;
    }

    // Include prompt if provided
    if (prompt) {
      requestBody.prompt = prompt;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.queued) {
      console.log(
        "Generation queued at position:",
        result.position,
        "model:",
        result.model_id
      );
      // Get model name for display
      const modelName = getModelDisplayName(result.model_id);
      const modelInfo = modelName ? ` (${modelName})` : "";
      showToast(
        "success",
        "Added to queue",
        `Queued at position ${result.position}${modelInfo}`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for render`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Render queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function generateRectangle() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the rectangle corners."
    );
    return;
  }

  // Get the two selected coordinates
  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return { x, y };
  });

  // Calculate rectangle bounds (top-left and bottom-right)
  const minX = Math.min(coords[0].x, coords[1].x);
  const maxX = Math.max(coords[0].x, coords[1].x);
  const minY = Math.min(coords[0].y, coords[1].y);
  const maxY = Math.max(coords[0].y, coords[1].y);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const totalQuadrants = width * height;

  // Build confirmation message
  const confirmMessage =
    `Generate rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY})?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `This will create a generation plan and queue all steps.\n` +
    `Pre-existing generations will be skipped.`;

  if (!confirm(confirmMessage)) {
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate rectangle requested:",
    { tl: [minX, minY], br: [maxX, maxY] },
    "with model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("generateRectBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Queueing...<span class="spinner"></span>';

  showToast(
    "loading",
    "Creating generation plan...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY})`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        model_id: modelId,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";

    if (result.success) {
      if (result.queued_count === 0) {
        showToast(
          "info",
          "Nothing to generate",
          result.message || "All quadrants already generated."
        );
      } else {
        console.log("Rectangle generation queued:", result);
        const summary = result.plan_summary || {};
        const stepTypes = summary.steps_by_type || {};
        const typeInfo = Object.entries(stepTypes)
          .map(([type, count]) => `${count}× ${type}`)
          .join(", ");

        showToast(
          "success",
          "Rectangle queued!",
          `${result.queued_count} step(s) for ${
            summary.total_quadrants || "?"
          } quadrant(s)` + (typeInfo ? ` (${typeInfo})` : "")
        );
      }
    } else {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    clearLoadingToasts();
    console.error("Generate rectangle error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  saveSelectedQuadrants();
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  if (
    tileEl.classList.contains("locked") ||
    tileEl.classList.contains("queued")
  ) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  saveSelectedQuadrants();
  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;
let lastStatus = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Track state changes - consider active_model_count for parallel processing
    const wasGenerating = isGenerating || isRendering;
    const hasActiveModels = (status.active_model_count || 0) > 0;
    isGenerating =
      (status.is_generating || hasActiveModels) &&
      status.status !== "rendering";
    isRendering = status.is_generating && status.status === "rendering";
    const nowProcessing = isGenerating || isRendering;

    // Apply visual styles based on server status
    applyStatusStyles(status);

    // Update render button based on state
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on render button only (generate state is shown in toolbar)
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      } else {
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
      }

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        const opName = isRendering ? "Render" : "Generation";
        showToast(
          "loading",
          `${opName} in progress...`,
          status.message || "Please wait..."
        );
      }

      // Update the loading toast message
      const loadingToast = document.querySelector(
        ".toast.loading .toast-message"
      );
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset render button
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        // Build per-model queue message
        let queueMsg = "";
        if (status.queue_by_model) {
          const parts = Object.entries(status.queue_by_model)
            .map(([modelId, info]) => {
              const name = getModelDisplayName(modelId) || modelId;
              const count = info.pending_count + (info.is_processing ? 1 : 0);
              return { name, count };
            })
            .filter(({ count }) => count > 0)
            .map(({ name, count }) => `${name}: ${count}`);
          if (parts.length > 0) {
            queueMsg = parts.join(", ");
          }
        }
        if (queueMsg) {
          showToast("info", "Processing queue", queueMsg);
        }
      } else {
        // No more items, reload after delay
        stopStatusPolling();
        setTimeout(() => window.location.reload(), 1500);
      }
    } else if (status.status === "error" && status.error) {
      clearLoadingToasts();
      showToast("error", "Error", status.error);

      // Continue polling if there are more items in queue
      if (status.queue_length === 0) {
        stopStatusPolling();
      }
    } else if (
      status.status === "idle" &&
      status.queue_length === 0 &&
      !nowProcessing
    ) {
      // Idle with no queue - stop polling
      stopStatusPolling();
    }

    // Update selection status with server info
    updateSelectionStatus(status);
    lastStatus = status;
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

// Restore saved tool on page load
function restoreSavedTool() {
  const savedTool = getSavedTool();
  if (!savedTool) return;

  // Check if the tool button exists before activating
  switch (savedTool) {
    case "select":
      if (document.getElementById("selectTool")) {
        toggleSelectTool();
      }
      break;
    case "fixwater":
      if (document.getElementById("fixWaterTool")) {
        toggleFixWaterTool();
      }
      break;
    case "waterfill":
      if (document.getElementById("waterFillTool")) {
        toggleWaterFillTool();
      }
      break;
    default:
      // Unknown tool, clear saved state
      saveSelectedTool("");
      break;
  }
}

// Restore saved quadrant selections on page load
function restoreSavedQuadrants() {
  const savedQuadrants = getSavedQuadrants();
  if (!savedQuadrants || savedQuadrants.length === 0) return;

  let restoredCount = 0;

  savedQuadrants.forEach((key) => {
    // Check if this quadrant tile exists on the current page
    const tile = document.querySelector(`.tile[data-coords="${key}"]`);
    if (tile) {
      // Don't restore if tile is locked or queued
      if (
        !tile.classList.contains("locked") &&
        !tile.classList.contains("queued")
      ) {
        selectedQuadrants.add(key);
        tile.classList.add("selected");
        restoredCount++;
      }
    }
  });

  if (restoredCount > 0) {
    console.log(`Restored ${restoredCount} selected quadrant(s)`);
    // Update localStorage to only contain valid selections
    saveSelectedQuadrants();
    updateSelectionStatus();
  } else if (savedQuadrants.length > 0) {
    // Had saved selections but none are on current page - clear storage
    saveSelectedQuadrants();
  }
}

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

  // Restore saved quadrant selections
  restoreSavedQuadrants();

  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Apply initial status styles
    applyStatusStyles(status);

    if (status.is_generating || status.queue_length > 0) {
      console.log(
        "Processing in progress or queue non-empty, starting polling..."
      );
      isGenerating = status.is_generating && status.status !== "rendering";
      isRendering = status.is_generating && status.status === "rendering";
      startStatusPolling();
    }

    updateSelectionStatus(status);
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now I need to update the code that sets the status text to handle HTML and add click handlers:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";

// Save selected model ID to localStorage
function saveSelectedModel(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY_MODEL, modelId);
  } catch (e) {
    console.warn("Could not save model to localStorage:", e);
  }
}

// Get saved model ID from localStorage
function getSavedModel() {
  try {
    return localStorage.getItem(STORAGE_KEY_MODEL);
  } catch (e) {
    return null;
  }
}

// Save selected tool to localStorage
function saveSelectedTool(toolName) {
  try {
    localStorage.setItem(STORAGE_KEY_TOOL, toolName || "");
  } catch (e) {
    console.warn("Could not save tool to localStorage:", e);
  }
}

// Get saved tool from localStorage
function getSavedTool() {
  try {
    return localStorage.getItem(STORAGE_KEY_TOOL) || "";
  } catch (e) {
    return "";
  }
}

// Save selected quadrants to localStorage
function saveSelectedQuadrants() {
  try {
    const quadrantsArray = Array.from(selectedQuadrants);
    localStorage.setItem(STORAGE_KEY_SELECTION, JSON.stringify(quadrantsArray));
  } catch (e) {
    console.warn("Could not save selection to localStorage:", e);
  }
}

// Get saved quadrants from localStorage
function getSavedQuadrants() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTION);
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    return [];
  }
}

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Check if saved model ID exists in available models
  const savedModelId = getSavedModel();
  const savedModelExists =
    savedModelId && config.models.some((m) => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;

    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (
      !savedModelExists &&
      model.model_id === config.default_model_id
    ) {
      option.selected = true;
    } else if (!savedModelExists && !config.default_model_id && index === 0) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection and save to localStorage
  select.addEventListener("change", () => {
    saveSelectedModel(select.value);
    select.blur();
  });
}

// Get the currently selected model ID
function getSelectedModelId() {
  const select = document.getElementById("modelSelect");
  return select ? select.value : null;
}

// Get display name for a model ID
function getModelDisplayName(modelId) {
  if (!modelId) return null;
  const configEl = document.getElementById("app-config");
  if (!configEl) return modelId;
  try {
    const config = JSON.parse(configEl.dataset.config);
    const models = config.models || [];
    const model = models.find((m) => m.model_id === modelId);
    return model ? model.name : modelId;
  } catch {
    return modelId;
  }
}

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to ALL currently processing quadrants (from all models)
  const processingQuadrants =
    status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || status.active_model_count > 0;

  if (isProcessing && processingQuadrants.length > 0) {
    document.body.classList.add("generating");
    processingQuadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else if (!isProcessing) {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items AND create overlays
  // Also create overlays for processing items
  const processingItems = getProcessingItems(status);
  updateQueueOverlays(status.queue || [], processingItems);
}

// Extract currently processing items from status
function getProcessingItems(status) {
  const processingItems = [];
  if (status.queue_by_model) {
    Object.entries(status.queue_by_model).forEach(([modelId, info]) => {
      if (info.is_processing && info.current_item) {
        processingItems.push({
          ...info.current_item,
          model_id: modelId,
          _isProcessing: true,
        });
      }
    });
  }
  return processingItems;
}

// Create/update overlays for pending queue items and processing items
function updateQueueOverlays(queueItems, processingItems = []) {
  // Remove existing overlays
  document.querySelectorAll(".queue-overlay").forEach((el) => el.remove());

  // Also clear queued class from all tiles
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    tile.classList.remove("queued");
  });

  const hasItems =
    (queueItems && queueItems.length > 0) ||
    (processingItems && processingItems.length > 0);
  if (!hasItems) return;

  const grid = document.querySelector(".grid");
  if (!grid) return;

  // Get grid dimensions from config
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Helper function to create an overlay for an item
  function createOverlay(item, options = {}) {
    const { isProcessing = false, queuePosition = null } = options;

    if (!item.quadrants || item.quadrants.length === 0) return null;

    // Calculate bounding box for this generation
    const quadrants = item.quadrants;
    let minCol = Infinity,
      maxCol = -Infinity;
    let minRow = Infinity,
      maxRow = -Infinity;

    // Track which tiles are visible in the current view
    const visibleQuadrants = [];

    quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        visibleQuadrants.push([qx, qy]);
        const col = qx - gridX;
        const row = qy - gridY;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);

        // Add appropriate class to tile
        if (isProcessing) {
          tile.classList.add("locked");
        } else if (!tile.classList.contains("locked")) {
          tile.classList.add("queued");
        }
      }
    });

    if (visibleQuadrants.length === 0) return null;

    // Create overlay element
    const overlay = document.createElement("div");
    overlay.className = isProcessing
      ? "queue-overlay processing"
      : "queue-overlay";
    overlay.dataset.itemId = item.id;

    // Calculate position and size
    const left = minCol * (sizePx + gap);
    const top = minRow * (sizePx + gap);
    const width = (maxCol - minCol + 1) * sizePx + (maxCol - minCol) * gap;
    const height = (maxRow - minRow + 1) * sizePx + (maxRow - minRow) * gap;

    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    // Badge: spinner for processing, number for queued
    const badge = document.createElement("div");
    badge.className = isProcessing ? "queue-badge processing" : "queue-badge";
    if (isProcessing) {
      badge.innerHTML = `<svg class="processing-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>`;
    } else {
      badge.textContent = queuePosition;
    }
    overlay.appendChild(badge);

    // Model name label
    if (item.model_id) {
      const modelLabel = document.createElement("div");
      modelLabel.className = "queue-model-label";
      modelLabel.textContent =
        getModelDisplayName(item.model_id) || item.model_id;
      overlay.appendChild(modelLabel);
    }

    // Cancel button (visible on hover) - for both processing and queued
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "queue-cancel-btn";
    cancelBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>`;
    cancelBtn.title = isProcessing
      ? "Cancel this generation (in progress)"
      : "Cancel this generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelQueueItem(item.id);
    };
    overlay.appendChild(cancelBtn);

    return overlay;
  }

  // First, create overlays for processing items (with spinner)
  processingItems.forEach((item) => {
    const overlay = createOverlay(item, { isProcessing: true });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });

  // Then, create overlays for queued items (with per-model position numbers)
  // Track position within each model's queue
  const modelPositionCounters = {};

  queueItems.forEach((item) => {
    const modelId = item.model_id || "default";
    // Initialize or increment the counter for this model
    if (!modelPositionCounters[modelId]) {
      modelPositionCounters[modelId] = 1;
    }
    const positionInModelQueue = modelPositionCounters[modelId];
    modelPositionCounters[modelId]++;

    const overlay = createOverlay(item, {
      queuePosition: positionInModelQueue,
    });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });
}

// Cancel a specific queue item
async function cancelQueueItem(itemId) {
  try {
    const response = await fetch(`/api/queue/cancel/${itemId}`, {
      method: "POST",
    });
    const result = await response.json();

    if (result.success && result.cancelled) {
      showToast("success", "Cancelled", result.message);
      // Trigger immediate status update
      checkGenerationStatus();
    } else if (result.success && !result.cancelled) {
      showToast("info", "Not found", result.message);
    } else {
      showToast("error", "Error", result.error || "Failed to cancel");
    }
  } catch (error) {
    console.error("Cancel queue item failed:", error);
    showToast("error", "Error", "Failed to cancel queue item");
  }
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "d":
    case "D":
      document.getElementById("showRender").click();
      break;
    case "g":
    case "G":
      generateSelected();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate other tools if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
    saveSelectedTool("select");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
    saveSelectedTool("");
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate other tools if active
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
    saveSelectedTool("fixwater");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast(
    "loading",
    "Filling with water...",
    `Processing quadrant (${qx}, ${qy})`
  );

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus(serverStatus = null) {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");
  const generateRectBtn = document.getElementById("generateRectBtn");

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    // Show all active models generating
    const activeModels = serverStatus.active_models || [];
    const processingQuadrants =
      serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

    if (activeModels.length > 0 && processingQuadrants.length > 0) {
      // Show which models are actively generating with their queue counts
      const queueByModel = serverStatus.queue_by_model || {};

      if (activeModels.length === 1) {
        const modelId = activeModels[0];
        const modelName = getModelDisplayName(modelId) || modelId || "default";
        const modelInfo = queueByModel[modelId];
        const queueCount = modelInfo
          ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
          : 0;
        // Create clickable coordinate links
        const coordsHtml = processingQuadrants
          .map(
            ([x, y]) =>
              `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
          )
          .join(" ");
        const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
        statusParts.push({
          html: `🔄 ${modelName}${countStr}: ${coordsHtml}`,
        });
      } else {
        // Multiple models generating in parallel - show each with queue count and coords
        const queueByModel = serverStatus.queue_by_model || {};
        const modelPartsHtml = activeModels.map((modelId) => {
          const name = getModelDisplayName(modelId) || modelId || "default";
          const modelInfo = queueByModel[modelId];
          const queueCount = modelInfo
            ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
            : 0;
          // Get quadrants for this specific model
          const modelQuadrants =
            modelInfo && modelInfo.current_item
              ? modelInfo.current_item.quadrants || []
              : [];
          const coordsHtml =
            modelQuadrants.length > 0
              ? " " +
                modelQuadrants
                  .map(
                    ([x, y]) =>
                      `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
                  )
                  .join(" ")
              : "";
          const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
          return `${name}${countStr}${coordsHtml}`;
        });
        statusParts.push({ html: `🔄 ${modelPartsHtml.join(", ")}` });
      }
    } else if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      // Fallback to old behavior with clickable coords
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsHtml = serverStatus.quadrants
        .map(
          ([x, y]) =>
            `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
        )
        .join(" ");
      statusParts.push({ html: `${action} ${coordsHtml}` });
    }

    // Show per-model queue counts (only models with items)
    if (serverStatus.queue_by_model) {
      const modelQueues = Object.entries(serverStatus.queue_by_model);
      const queueParts = modelQueues
        .map(([modelId, info]) => {
          const name = getModelDisplayName(modelId) || modelId;
          const count = info.pending_count + (info.is_processing ? 1 : 0);
          return { name, count };
        })
        .filter(({ count }) => count > 0)
        .map(({ name, count }) => `${name}: ${count}`);

      if (queueParts.length > 0) {
        statusParts.push(`📋 ${queueParts.join(", ")}`);
      }
    }
  }

  // Update selection display in toolbar-info (separate from status)
  const selectedDisplay = document.getElementById("selectedQuadrantsDisplay");
  if (selectedDisplay) {
    if (count > 0) {
      const coordsStr = Array.from(selectedQuadrants)
        .map((key) => {
          const [x, y] = key.split(",");
          return `(${x},${y})`;
        })
        .join(" ");
      selectedDisplay.textContent = `✓ ${coordsStr}`;
      selectedDisplay.style.display = "";
    } else {
      selectedDisplay.textContent = "";
      selectedDisplay.style.display = "none";
    }
  }

  // Build status display - some parts may be HTML objects, others plain strings
  if (statusParts.length > 0) {
    const statusHtml = statusParts
      .map((part) => (typeof part === "object" && part.html ? part.html : part))
      .join(" • ");
    countEl.innerHTML = statusHtml;

    // Add click handlers for coordinate links
    countEl.querySelectorAll(".coord-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const x = parseInt(link.dataset.x, 10);
        const y = parseInt(link.dataset.y, 10);
        navigateToCoord(x, y);
      });
    });
  } else {
    countEl.textContent = `${count} selected`;
  }

  // Update status bar styling
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (isProcessing) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate Rectangle requires exactly 2 selected
  generateRectBtn.disabled = count !== 2;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state (tracked from server)
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function clearQueue() {
  // Get current queue info first
  try {
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();

    const queueLength = status.queue_length || 0;
    const isGenerating = status.is_generating || false;

    if (queueLength === 0 && !isGenerating) {
      showToast(
        "info",
        "Nothing to clear",
        "There are no pending items or active generations."
      );
      return;
    }

    // Build confirm message based on what will be cancelled
    let confirmMessage = "Are you sure you want to clear the queue?";
    const parts = [];
    if (queueLength > 0) {
      parts.push(`${queueLength} pending item(s)`);
    }
    if (isGenerating) {
      parts.push("the current generation in progress");
    }
    if (parts.length > 0) {
      confirmMessage = `Are you sure you want to cancel ${parts.join(
        " and "
      )}?`;
    }

    if (!confirm(confirmMessage)) {
      return;
    }

    // Clear the queue
    const response = await fetch("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        "Queue cleared",
        result.message || `Cleared ${result.cleared_count} item(s)`
      );
      // Refresh status
      await checkGenerationStatus();
    } else {
      showToast(
        "error",
        "Failed to clear queue",
        result.error || "Unknown error"
      );
    }
  } catch (error) {
    console.error("Clear queue error:", error);
    showToast("error", "Request failed", error.message);
  }
}

// Prompt dialog functions
function showPromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("promptDialog");
  const input = document.getElementById("promptInput");
  if (dialog && input) {
    input.value = "";
    dialog.style.display = "flex";
    input.focus();
  }
}

function hidePromptDialog() {
  const dialog = document.getElementById("promptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitPromptGeneration() {
  const input = document.getElementById("promptInput");
  const prompt = input ? input.value.trim() : "";
  hidePromptDialog();
  await generateSelected(prompt);
}

async function generateSelected(prompt = null) {
  if (selectedQuadrants.size === 0) return;

  // Separate selected quadrants into those that need generation vs those that already have it
  const toGenerate = [];
  const contextQuadrants = [];

  Array.from(selectedQuadrants).forEach((s) => {
    const [x, y] = s.split(",").map(Number);
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

    if (tile && !tile.classList.contains("placeholder")) {
      // This quadrant already has a generation - use as context
      contextQuadrants.push([x, y]);
    } else {
      // This quadrant needs generation
      toGenerate.push([x, y]);
    }
  });

  // If nothing needs generation, inform the user
  if (toGenerate.length === 0) {
    showToast(
      "info",
      "Already generated",
      "All selected quadrants already have generations. Select at least one empty quadrant."
    );
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback with context info
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${toGenerate.length} quadrant(s) for generation${promptMsg}${contextMsg}`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const requestBody = {
      quadrants: toGenerate,
      model_id: modelId,
    };

    // Include context quadrants if any
    if (contextQuadrants.length > 0) {
      requestBody.context = contextQuadrants;
    }

    // Include prompt if provided
    if (prompt) {
      requestBody.prompt = prompt;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.queued) {
      console.log(
        "Generation queued at position:",
        result.position,
        "model:",
        result.model_id
      );
      // Get model name for display
      const modelName = getModelDisplayName(result.model_id);
      const modelInfo = modelName ? ` (${modelName})` : "";
      showToast(
        "success",
        "Added to queue",
        `Queued at position ${result.position}${modelInfo}`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for render`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Render queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function generateRectangle() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the rectangle corners."
    );
    return;
  }

  // Get the two selected coordinates
  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return { x, y };
  });

  // Calculate rectangle bounds (top-left and bottom-right)
  const minX = Math.min(coords[0].x, coords[1].x);
  const maxX = Math.max(coords[0].x, coords[1].x);
  const minY = Math.min(coords[0].y, coords[1].y);
  const maxY = Math.max(coords[0].y, coords[1].y);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const totalQuadrants = width * height;

  // Build confirmation message
  const confirmMessage =
    `Generate rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY})?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `This will create a generation plan and queue all steps.\n` +
    `Pre-existing generations will be skipped.`;

  if (!confirm(confirmMessage)) {
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate rectangle requested:",
    { tl: [minX, minY], br: [maxX, maxY] },
    "with model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("generateRectBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Queueing...<span class="spinner"></span>';

  showToast(
    "loading",
    "Creating generation plan...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY})`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        model_id: modelId,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";

    if (result.success) {
      if (result.queued_count === 0) {
        showToast(
          "info",
          "Nothing to generate",
          result.message || "All quadrants already generated."
        );
      } else {
        console.log("Rectangle generation queued:", result);
        const summary = result.plan_summary || {};
        const stepTypes = summary.steps_by_type || {};
        const typeInfo = Object.entries(stepTypes)
          .map(([type, count]) => `${count}× ${type}`)
          .join(", ");

        showToast(
          "success",
          "Rectangle queued!",
          `${result.queued_count} step(s) for ${
            summary.total_quadrants || "?"
          } quadrant(s)` + (typeInfo ? ` (${typeInfo})` : "")
        );
      }
    } else {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    clearLoadingToasts();
    console.error("Generate rectangle error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  saveSelectedQuadrants();
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  if (
    tileEl.classList.contains("locked") ||
    tileEl.classList.contains("queued")
  ) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  saveSelectedQuadrants();
  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;
let lastStatus = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Track state changes - consider active_model_count for parallel processing
    const wasGenerating = isGenerating || isRendering;
    const hasActiveModels = (status.active_model_count || 0) > 0;
    isGenerating =
      (status.is_generating || hasActiveModels) &&
      status.status !== "rendering";
    isRendering = status.is_generating && status.status === "rendering";
    const nowProcessing = isGenerating || isRendering;

    // Apply visual styles based on server status
    applyStatusStyles(status);

    // Update render button based on state
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on render button only (generate state is shown in toolbar)
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      } else {
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
      }

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        const opName = isRendering ? "Render" : "Generation";
        showToast(
          "loading",
          `${opName} in progress...`,
          status.message || "Please wait..."
        );
      }

      // Update the loading toast message
      const loadingToast = document.querySelector(
        ".toast.loading .toast-message"
      );
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset render button
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        // Build per-model queue message
        let queueMsg = "";
        if (status.queue_by_model) {
          const parts = Object.entries(status.queue_by_model)
            .map(([modelId, info]) => {
              const name = getModelDisplayName(modelId) || modelId;
              const count = info.pending_count + (info.is_processing ? 1 : 0);
              return { name, count };
            })
            .filter(({ count }) => count > 0)
            .map(({ name, count }) => `${name}: ${count}`);
          if (parts.length > 0) {
            queueMsg = parts.join(", ");
          }
        }
        if (queueMsg) {
          showToast("info", "Processing queue", queueMsg);
        }
      } else {
        // No more items, reload after delay
        stopStatusPolling();
        setTimeout(() => window.location.reload(), 1500);
      }
    } else if (status.status === "error" && status.error) {
      clearLoadingToasts();
      showToast("error", "Error", status.error);

      // Continue polling if there are more items in queue
      if (status.queue_length === 0) {
        stopStatusPolling();
      }
    } else if (
      status.status === "idle" &&
      status.queue_length === 0 &&
      !nowProcessing
    ) {
      // Idle with no queue - stop polling
      stopStatusPolling();
    }

    // Update selection status with server info
    updateSelectionStatus(status);
    lastStatus = status;
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

// Restore saved tool on page load
function restoreSavedTool() {
  const savedTool = getSavedTool();
  if (!savedTool) return;

  // Check if the tool button exists before activating
  switch (savedTool) {
    case "select":
      if (document.getElementById("selectTool")) {
        toggleSelectTool();
      }
      break;
    case "fixwater":
      if (document.getElementById("fixWaterTool")) {
        toggleFixWaterTool();
      }
      break;
    case "waterfill":
      if (document.getElementById("waterFillTool")) {
        toggleWaterFillTool();
      }
      break;
    default:
      // Unknown tool, clear saved state
      saveSelectedTool("");
      break;
  }
}

// Restore saved quadrant selections on page load
function restoreSavedQuadrants() {
  const savedQuadrants = getSavedQuadrants();
  if (!savedQuadrants || savedQuadrants.length === 0) return;

  let restoredCount = 0;

  savedQuadrants.forEach((key) => {
    // Check if this quadrant tile exists on the current page
    const tile = document.querySelector(`.tile[data-coords="${key}"]`);
    if (tile) {
      // Don't restore if tile is locked or queued
      if (
        !tile.classList.contains("locked") &&
        !tile.classList.contains("queued")
      ) {
        selectedQuadrants.add(key);
        tile.classList.add("selected");
        restoredCount++;
      }
    }
  });

  if (restoredCount > 0) {
    console.log(`Restored ${restoredCount} selected quadrant(s)`);
    // Update localStorage to only contain valid selections
    saveSelectedQuadrants();
    updateSelectionStatus();
  } else if (savedQuadrants.length > 0) {
    // Had saved selections but none are on current page - clear storage
    saveSelectedQuadrants();
  }
}

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

  // Restore saved quadrant selections
  restoreSavedQuadrants();

  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Apply initial status styles
    applyStatusStyles(status);

    if (status.is_generating || status.queue_length > 0) {
      console.log(
        "Processing in progress or queue non-empty, starting polling..."
      );
      isGenerating = status.is_generating && status.status !== "rendering";
      isRendering = status.is_generating && status.status === "rendering";
      startStatusPolling();
    }

    updateSelectionStatus(status);
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now I need to add the `navigateToCoord` function. Let me add it near the other navigation functions:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";

// Save selected model ID to localStorage
function saveSelectedModel(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY_MODEL, modelId);
  } catch (e) {
    console.warn("Could not save model to localStorage:", e);
  }
}

// Get saved model ID from localStorage
function getSavedModel() {
  try {
    return localStorage.getItem(STORAGE_KEY_MODEL);
  } catch (e) {
    return null;
  }
}

// Save selected tool to localStorage
function saveSelectedTool(toolName) {
  try {
    localStorage.setItem(STORAGE_KEY_TOOL, toolName || "");
  } catch (e) {
    console.warn("Could not save tool to localStorage:", e);
  }
}

// Get saved tool from localStorage
function getSavedTool() {
  try {
    return localStorage.getItem(STORAGE_KEY_TOOL) || "";
  } catch (e) {
    return "";
  }
}

// Save selected quadrants to localStorage
function saveSelectedQuadrants() {
  try {
    const quadrantsArray = Array.from(selectedQuadrants);
    localStorage.setItem(STORAGE_KEY_SELECTION, JSON.stringify(quadrantsArray));
  } catch (e) {
    console.warn("Could not save selection to localStorage:", e);
  }
}

// Get saved quadrants from localStorage
function getSavedQuadrants() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_SELECTION);
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    return [];
  }
}

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Check if saved model ID exists in available models
  const savedModelId = getSavedModel();
  const savedModelExists =
    savedModelId && config.models.some((m) => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;

    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (
      !savedModelExists &&
      model.model_id === config.default_model_id
    ) {
      option.selected = true;
    } else if (!savedModelExists && !config.default_model_id && index === 0) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection and save to localStorage
  select.addEventListener("change", () => {
    saveSelectedModel(select.value);
    select.blur();
  });
}

// Get the currently selected model ID
function getSelectedModelId() {
  const select = document.getElementById("modelSelect");
  return select ? select.value : null;
}

// Get display name for a model ID
function getModelDisplayName(modelId) {
  if (!modelId) return null;
  const configEl = document.getElementById("app-config");
  if (!configEl) return modelId;
  try {
    const config = JSON.parse(configEl.dataset.config);
    const models = config.models || [];
    const model = models.find((m) => m.model_id === modelId);
    return model ? model.name : modelId;
  } catch {
    return modelId;
  }
}

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to ALL currently processing quadrants (from all models)
  const processingQuadrants =
    status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || status.active_model_count > 0;

  if (isProcessing && processingQuadrants.length > 0) {
    document.body.classList.add("generating");
    processingQuadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else if (!isProcessing) {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items AND create overlays
  // Also create overlays for processing items
  const processingItems = getProcessingItems(status);
  updateQueueOverlays(status.queue || [], processingItems);
}

// Extract currently processing items from status
function getProcessingItems(status) {
  const processingItems = [];
  if (status.queue_by_model) {
    Object.entries(status.queue_by_model).forEach(([modelId, info]) => {
      if (info.is_processing && info.current_item) {
        processingItems.push({
          ...info.current_item,
          model_id: modelId,
          _isProcessing: true,
        });
      }
    });
  }
  return processingItems;
}

// Create/update overlays for pending queue items and processing items
function updateQueueOverlays(queueItems, processingItems = []) {
  // Remove existing overlays
  document.querySelectorAll(".queue-overlay").forEach((el) => el.remove());

  // Also clear queued class from all tiles
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    tile.classList.remove("queued");
  });

  const hasItems =
    (queueItems && queueItems.length > 0) ||
    (processingItems && processingItems.length > 0);
  if (!hasItems) return;

  const grid = document.querySelector(".grid");
  if (!grid) return;

  // Get grid dimensions from config
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Helper function to create an overlay for an item
  function createOverlay(item, options = {}) {
    const { isProcessing = false, queuePosition = null } = options;

    if (!item.quadrants || item.quadrants.length === 0) return null;

    // Calculate bounding box for this generation
    const quadrants = item.quadrants;
    let minCol = Infinity,
      maxCol = -Infinity;
    let minRow = Infinity,
      maxRow = -Infinity;

    // Track which tiles are visible in the current view
    const visibleQuadrants = [];

    quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        visibleQuadrants.push([qx, qy]);
        const col = qx - gridX;
        const row = qy - gridY;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);

        // Add appropriate class to tile
        if (isProcessing) {
          tile.classList.add("locked");
        } else if (!tile.classList.contains("locked")) {
          tile.classList.add("queued");
        }
      }
    });

    if (visibleQuadrants.length === 0) return null;

    // Create overlay element
    const overlay = document.createElement("div");
    overlay.className = isProcessing
      ? "queue-overlay processing"
      : "queue-overlay";
    overlay.dataset.itemId = item.id;

    // Calculate position and size
    const left = minCol * (sizePx + gap);
    const top = minRow * (sizePx + gap);
    const width = (maxCol - minCol + 1) * sizePx + (maxCol - minCol) * gap;
    const height = (maxRow - minRow + 1) * sizePx + (maxRow - minRow) * gap;

    overlay.style.left = `${left}px`;
    overlay.style.top = `${top}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;

    // Badge: spinner for processing, number for queued
    const badge = document.createElement("div");
    badge.className = isProcessing ? "queue-badge processing" : "queue-badge";
    if (isProcessing) {
      badge.innerHTML = `<svg class="processing-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
      </svg>`;
    } else {
      badge.textContent = queuePosition;
    }
    overlay.appendChild(badge);

    // Model name label
    if (item.model_id) {
      const modelLabel = document.createElement("div");
      modelLabel.className = "queue-model-label";
      modelLabel.textContent =
        getModelDisplayName(item.model_id) || item.model_id;
      overlay.appendChild(modelLabel);
    }

    // Cancel button (visible on hover) - for both processing and queued
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "queue-cancel-btn";
    cancelBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>`;
    cancelBtn.title = isProcessing
      ? "Cancel this generation (in progress)"
      : "Cancel this generation";
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelQueueItem(item.id);
    };
    overlay.appendChild(cancelBtn);

    return overlay;
  }

  // First, create overlays for processing items (with spinner)
  processingItems.forEach((item) => {
    const overlay = createOverlay(item, { isProcessing: true });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });

  // Then, create overlays for queued items (with per-model position numbers)
  // Track position within each model's queue
  const modelPositionCounters = {};

  queueItems.forEach((item) => {
    const modelId = item.model_id || "default";
    // Initialize or increment the counter for this model
    if (!modelPositionCounters[modelId]) {
      modelPositionCounters[modelId] = 1;
    }
    const positionInModelQueue = modelPositionCounters[modelId];
    modelPositionCounters[modelId]++;

    const overlay = createOverlay(item, {
      queuePosition: positionInModelQueue,
    });
    if (overlay) {
      grid.appendChild(overlay);
    }
  });
}

// Cancel a specific queue item
async function cancelQueueItem(itemId) {
  try {
    const response = await fetch(`/api/queue/cancel/${itemId}`, {
      method: "POST",
    });
    const result = await response.json();

    if (result.success && result.cancelled) {
      showToast("success", "Cancelled", result.message);
      // Trigger immediate status update
      checkGenerationStatus();
    } else if (result.success && !result.cancelled) {
      showToast("info", "Not found", result.message);
    } else {
      showToast("error", "Error", result.error || "Failed to cancel");
    }
  } catch (error) {
    console.error("Cancel queue item failed:", error);
    showToast("error", "Error", "Failed to cancel queue item");
  }
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

// Navigate to center the view on a specific coordinate
function navigateToCoord(targetX, targetY) {
  const params = getParams();
  const nx = parseInt(params.nx);
  const ny = parseInt(params.ny);
  // Center the target coordinate in the view
  const x = targetX - Math.floor(nx / 2);
  const y = targetY - Math.floor(ny / 2);
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "d":
    case "D":
      document.getElementById("showRender").click();
      break;
    case "g":
    case "G":
      generateSelected();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate other tools if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
    saveSelectedTool("select");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
    saveSelectedTool("");
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate other tools if active
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
    saveSelectedTool("fixwater");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast(
    "loading",
    "Filling with water...",
    `Processing quadrant (${qx}, ${qy})`
  );

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus(serverStatus = null) {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");
  const generateRectBtn = document.getElementById("generateRectBtn");

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    // Show all active models generating
    const activeModels = serverStatus.active_models || [];
    const processingQuadrants =
      serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

    if (activeModels.length > 0 && processingQuadrants.length > 0) {
      // Show which models are actively generating with their queue counts
      const queueByModel = serverStatus.queue_by_model || {};

      if (activeModels.length === 1) {
        const modelId = activeModels[0];
        const modelName = getModelDisplayName(modelId) || modelId || "default";
        const modelInfo = queueByModel[modelId];
        const queueCount = modelInfo
          ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
          : 0;
        // Create clickable coordinate links
        const coordsHtml = processingQuadrants
          .map(
            ([x, y]) =>
              `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
          )
          .join(" ");
        const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
        statusParts.push({
          html: `🔄 ${modelName}${countStr}: ${coordsHtml}`,
        });
      } else {
        // Multiple models generating in parallel - show each with queue count and coords
        const queueByModel = serverStatus.queue_by_model || {};
        const modelPartsHtml = activeModels.map((modelId) => {
          const name = getModelDisplayName(modelId) || modelId || "default";
          const modelInfo = queueByModel[modelId];
          const queueCount = modelInfo
            ? modelInfo.pending_count + (modelInfo.is_processing ? 1 : 0)
            : 0;
          // Get quadrants for this specific model
          const modelQuadrants =
            modelInfo && modelInfo.current_item
              ? modelInfo.current_item.quadrants || []
              : [];
          const coordsHtml =
            modelQuadrants.length > 0
              ? " " +
                modelQuadrants
                  .map(
                    ([x, y]) =>
                      `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
                  )
                  .join(" ")
              : "";
          const countStr = queueCount > 0 ? ` [${queueCount}]` : "";
          return `${name}${countStr}${coordsHtml}`;
        });
        statusParts.push({ html: `🔄 ${modelPartsHtml.join(", ")}` });
      }
    } else if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      // Fallback to old behavior with clickable coords
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsHtml = serverStatus.quadrants
        .map(
          ([x, y]) =>
            `<a href="#" class="coord-link" data-x="${x}" data-y="${y}">(${x},${y})</a>`
        )
        .join(" ");
      statusParts.push({ html: `${action} ${coordsHtml}` });
    }

    // Show per-model queue counts (only models with items)
    if (serverStatus.queue_by_model) {
      const modelQueues = Object.entries(serverStatus.queue_by_model);
      const queueParts = modelQueues
        .map(([modelId, info]) => {
          const name = getModelDisplayName(modelId) || modelId;
          const count = info.pending_count + (info.is_processing ? 1 : 0);
          return { name, count };
        })
        .filter(({ count }) => count > 0)
        .map(({ name, count }) => `${name}: ${count}`);

      if (queueParts.length > 0) {
        statusParts.push(`📋 ${queueParts.join(", ")}`);
      }
    }
  }

  // Update selection display in toolbar-info (separate from status)
  const selectedDisplay = document.getElementById("selectedQuadrantsDisplay");
  if (selectedDisplay) {
    if (count > 0) {
      const coordsStr = Array.from(selectedQuadrants)
        .map((key) => {
          const [x, y] = key.split(",");
          return `(${x},${y})`;
        })
        .join(" ");
      selectedDisplay.textContent = `✓ ${coordsStr}`;
      selectedDisplay.style.display = "";
    } else {
      selectedDisplay.textContent = "";
      selectedDisplay.style.display = "none";
    }
  }

  // Build status display - some parts may be HTML objects, others plain strings
  if (statusParts.length > 0) {
    const statusHtml = statusParts
      .map((part) => (typeof part === "object" && part.html ? part.html : part))
      .join(" • ");
    countEl.innerHTML = statusHtml;

    // Add click handlers for coordinate links
    countEl.querySelectorAll(".coord-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const x = parseInt(link.dataset.x, 10);
        const y = parseInt(link.dataset.y, 10);
        navigateToCoord(x, y);
      });
    });
  } else {
    countEl.textContent = `${count} selected`;
  }

  // Update status bar styling
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (isProcessing) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate Rectangle requires exactly 2 selected
  generateRectBtn.disabled = count !== 2;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state (tracked from server)
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function clearQueue() {
  // Get current queue info first
  try {
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();

    const queueLength = status.queue_length || 0;
    const isGenerating = status.is_generating || false;

    if (queueLength === 0 && !isGenerating) {
      showToast(
        "info",
        "Nothing to clear",
        "There are no pending items or active generations."
      );
      return;
    }

    // Build confirm message based on what will be cancelled
    let confirmMessage = "Are you sure you want to clear the queue?";
    const parts = [];
    if (queueLength > 0) {
      parts.push(`${queueLength} pending item(s)`);
    }
    if (isGenerating) {
      parts.push("the current generation in progress");
    }
    if (parts.length > 0) {
      confirmMessage = `Are you sure you want to cancel ${parts.join(
        " and "
      )}?`;
    }

    if (!confirm(confirmMessage)) {
      return;
    }

    // Clear the queue
    const response = await fetch("/api/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        "Queue cleared",
        result.message || `Cleared ${result.cleared_count} item(s)`
      );
      // Refresh status
      await checkGenerationStatus();
    } else {
      showToast(
        "error",
        "Failed to clear queue",
        result.error || "Unknown error"
      );
    }
  } catch (error) {
    console.error("Clear queue error:", error);
    showToast("error", "Request failed", error.message);
  }
}

// Prompt dialog functions
function showPromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("promptDialog");
  const input = document.getElementById("promptInput");
  if (dialog && input) {
    input.value = "";
    dialog.style.display = "flex";
    input.focus();
  }
}

function hidePromptDialog() {
  const dialog = document.getElementById("promptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitPromptGeneration() {
  const input = document.getElementById("promptInput");
  const prompt = input ? input.value.trim() : "";
  hidePromptDialog();
  await generateSelected(prompt);
}

async function generateSelected(prompt = null) {
  if (selectedQuadrants.size === 0) return;

  // Separate selected quadrants into those that need generation vs those that already have it
  const toGenerate = [];
  const contextQuadrants = [];

  Array.from(selectedQuadrants).forEach((s) => {
    const [x, y] = s.split(",").map(Number);
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

    if (tile && !tile.classList.contains("placeholder")) {
      // This quadrant already has a generation - use as context
      contextQuadrants.push([x, y]);
    } else {
      // This quadrant needs generation
      toGenerate.push([x, y]);
    }
  });

  // If nothing needs generation, inform the user
  if (toGenerate.length === 0) {
    showToast(
      "info",
      "Already generated",
      "All selected quadrants already have generations. Select at least one empty quadrant."
    );
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback with context info
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${toGenerate.length} quadrant(s) for generation${promptMsg}${contextMsg}`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const requestBody = {
      quadrants: toGenerate,
      model_id: modelId,
    };

    // Include context quadrants if any
    if (contextQuadrants.length > 0) {
      requestBody.context = contextQuadrants;
    }

    // Include prompt if provided
    if (prompt) {
      requestBody.prompt = prompt;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.queued) {
      console.log(
        "Generation queued at position:",
        result.position,
        "model:",
        result.model_id
      );
      // Get model name for display
      const modelName = getModelDisplayName(result.model_id);
      const modelInfo = modelName ? ` (${modelName})` : "";
      showToast(
        "success",
        "Added to queue",
        `Queued at position ${result.position}${modelInfo}`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for render`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Render queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
    } else if (!result.success) {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
  }
}

async function generateRectangle() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the rectangle corners."
    );
    return;
  }

  // Get the two selected coordinates
  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return { x, y };
  });

  // Calculate rectangle bounds (top-left and bottom-right)
  const minX = Math.min(coords[0].x, coords[1].x);
  const maxX = Math.max(coords[0].x, coords[1].x);
  const minY = Math.min(coords[0].y, coords[1].y);
  const maxY = Math.max(coords[0].y, coords[1].y);

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const totalQuadrants = width * height;

  // Build confirmation message
  const confirmMessage =
    `Generate rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY})?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `This will create a generation plan and queue all steps.\n` +
    `Pre-existing generations will be skipped.`;

  if (!confirm(confirmMessage)) {
    return;
  }

  const modelId = getSelectedModelId();

  console.log(
    "Generate rectangle requested:",
    { tl: [minX, minY], br: [maxX, maxY] },
    "with model:",
    modelId
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("generateRectBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Queueing...<span class="spinner"></span>';

  showToast(
    "loading",
    "Creating generation plan...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY})`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        model_id: modelId,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";

    if (result.success) {
      if (result.queued_count === 0) {
        showToast(
          "info",
          "Nothing to generate",
          result.message || "All quadrants already generated."
        );
      } else {
        console.log("Rectangle generation queued:", result);
        const summary = result.plan_summary || {};
        const stepTypes = summary.steps_by_type || {};
        const typeInfo = Object.entries(stepTypes)
          .map(([type, count]) => `${count}× ${type}`)
          .join(", ");

        showToast(
          "success",
          "Rectangle queued!",
          `${result.queued_count} step(s) for ${
            summary.total_quadrants || "?"
          } quadrant(s)` + (typeInfo ? ` (${typeInfo})` : "")
        );
      }
    } else {
      showToast("error", "Failed to queue", result.error || "Unknown error");
    }

    // Fetch latest status to update UI
    await checkGenerationStatus();
  } catch (error) {
    clearLoadingToasts();
    console.error("Generate rectangle error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Generate Rectangle";
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  saveSelectedQuadrants();
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  if (
    tileEl.classList.contains("locked") ||
    tileEl.classList.contains("queued")
  ) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  saveSelectedQuadrants();
  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;
let lastStatus = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Track state changes - consider active_model_count for parallel processing
    const wasGenerating = isGenerating || isRendering;
    const hasActiveModels = (status.active_model_count || 0) > 0;
    isGenerating =
      (status.is_generating || hasActiveModels) &&
      status.status !== "rendering";
    isRendering = status.is_generating && status.status === "rendering";
    const nowProcessing = isGenerating || isRendering;

    // Apply visual styles based on server status
    applyStatusStyles(status);

    // Update render button based on state
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on render button only (generate state is shown in toolbar)
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      } else {
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
      }

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        const opName = isRendering ? "Render" : "Generation";
        showToast(
          "loading",
          `${opName} in progress...`,
          status.message || "Please wait..."
        );
      }

      // Update the loading toast message
      const loadingToast = document.querySelector(
        ".toast.loading .toast-message"
      );
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset render button
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        // Build per-model queue message
        let queueMsg = "";
        if (status.queue_by_model) {
          const parts = Object.entries(status.queue_by_model)
            .map(([modelId, info]) => {
              const name = getModelDisplayName(modelId) || modelId;
              const count = info.pending_count + (info.is_processing ? 1 : 0);
              return { name, count };
            })
            .filter(({ count }) => count > 0)
            .map(({ name, count }) => `${name}: ${count}`);
          if (parts.length > 0) {
            queueMsg = parts.join(", ");
          }
        }
        if (queueMsg) {
          showToast("info", "Processing queue", queueMsg);
        }
      } else {
        // No more items, reload after delay
        stopStatusPolling();
        setTimeout(() => window.location.reload(), 1500);
      }
    } else if (status.status === "error" && status.error) {
      clearLoadingToasts();
      showToast("error", "Error", status.error);

      // Continue polling if there are more items in queue
      if (status.queue_length === 0) {
        stopStatusPolling();
      }
    } else if (
      status.status === "idle" &&
      status.queue_length === 0 &&
      !nowProcessing
    ) {
      // Idle with no queue - stop polling
      stopStatusPolling();
    }

    // Update selection status with server info
    updateSelectionStatus(status);
    lastStatus = status;
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

// Restore saved tool on page load
function restoreSavedTool() {
  const savedTool = getSavedTool();
  if (!savedTool) return;

  // Check if the tool button exists before activating
  switch (savedTool) {
    case "select":
      if (document.getElementById("selectTool")) {
        toggleSelectTool();
      }
      break;
    case "fixwater":
      if (document.getElementById("fixWaterTool")) {
        toggleFixWaterTool();
      }
      break;
    case "waterfill":
      if (document.getElementById("waterFillTool")) {
        toggleWaterFillTool();
      }
      break;
    default:
      // Unknown tool, clear saved state
      saveSelectedTool("");
      break;
  }
}

// Restore saved quadrant selections on page load
function restoreSavedQuadrants() {
  const savedQuadrants = getSavedQuadrants();
  if (!savedQuadrants || savedQuadrants.length === 0) return;

  let restoredCount = 0;

  savedQuadrants.forEach((key) => {
    // Check if this quadrant tile exists on the current page
    const tile = document.querySelector(`.tile[data-coords="${key}"]`);
    if (tile) {
      // Don't restore if tile is locked or queued
      if (
        !tile.classList.contains("locked") &&
        !tile.classList.contains("queued")
      ) {
        selectedQuadrants.add(key);
        tile.classList.add("selected");
        restoredCount++;
      }
    }
  });

  if (restoredCount > 0) {
    console.log(`Restored ${restoredCount} selected quadrant(s)`);
    // Update localStorage to only contain valid selections
    saveSelectedQuadrants();
    updateSelectionStatus();
  } else if (savedQuadrants.length > 0) {
    // Had saved selections but none are on current page - clear storage
    saveSelectedQuadrants();
  }
}

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

  // Restore saved quadrant selections
  restoreSavedQuadrants();

  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Apply initial status styles
    applyStatusStyles(status);

    if (status.is_generating || status.queue_length > 0) {
      console.log(
        "Processing in progress or queue non-empty, starting polling..."
      );
      isGenerating = status.is_generating && status.status !== "rendering";
      isRendering = status.is_generating && status.status === "rendering";
      startStatusPolling();
    }

    updateSelectionStatus(status);
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now let me add some CSS styling for the coordinate links to make them look clickable:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
  font-size: 12px;
  background: #1a1a2e;
  color: #eee;
  min-height: 100vh;
  padding: 15px;
  padding-top: 95px; /* Space for fixed toolbar */
}

.toolbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: #1a1a2e;
  padding: 10px 15px;
  z-index: 100;
  border-bottom: 1px solid #333;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.toolbar-row {
  display: flex;
  align-items: center;
  gap: 15px;
  flex-wrap: wrap;
}

.toolbar-info {
  color: #666;
  font-size: 0.75rem;
}

.toolbar-info span + span {
  margin-left: 12px;
}

.toolbar-info .selected-quadrants {
  color: #00d9ff;
  font-weight: bold;
}

.controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.controls label {
  color: #888;
  font-size: 0.8rem;
}

.controls input[type="number"] {
  width: 50px;
  padding: 5px 6px;
  border: 1px solid #333;
  border-radius: 3px;
  background: #16213e;
  color: #fff;
  font-family: inherit;
  font-size: 0.8rem;
}

.controls button {
  padding: 5px 12px;
  background: #00d9ff;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.75rem;
  font-weight: bold;
}

.controls button:hover {
  background: #00b8d4;
}

.toggle-group {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: 8px;
  padding-left: 12px;
  border-left: 1px solid #333;
}

.toggle-group label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  user-select: none;
  font-size: 0.75rem;
}

.toggle-group input[type="checkbox"] {
  width: 14px;
  height: 14px;
  accent-color: #00d9ff;
  cursor: pointer;
}

.grid-container {
  display: inline-block;
  border-radius: 8px;
  overflow: hidden;
}

.grid-container.show-lines {
  border: 2px solid #333;
}

.grid {
  display: grid;
  background: #333;
}

.grid-container.show-lines .grid {
  gap: 2px;
}

.grid-container:not(.show-lines) .grid {
  gap: 0;
  background: transparent;
}

.grid-container:not(.show-lines) {
  border: none;
}

.tile {
  position: relative;
  background: #2a2a4a;
  display: flex;
  align-items: center;
  justify-content: center;
}

.tile img {
  display: block;
  max-width: 100%;
  height: auto;
}

.tile.placeholder {
  background: #3a3a5a;
}

.tile .coords {
  position: absolute;
  top: 6px;
  left: 6px;
  background: rgba(0, 0, 0, 0.7);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.65rem;
  color: #00d9ff;
  transition: opacity 0.2s;
}

.tile.placeholder .coords {
  color: #666;
}

.grid-container:not(.show-coords) .tile .coords {
  opacity: 0;
}

/* Tool button styles */
.tools-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.tools-label {
  color: #666;
  font-size: 0.7rem;
}

.tool-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: #2a2a3e;
  color: #666;
  border: 1px solid #444;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  transition: all 0.15s ease;
}

.tool-btn:hover {
  background: #3a3a4e;
  color: #999;
  border-color: #555;
}

.tool-btn.active {
  background: #10b981;
  color: #fff;
  border-color: #10b981;
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.tool-btn.active:hover {
  background: #059669;
  border-color: #059669;
}

.tool-btn svg {
  width: 12px;
  height: 12px;
}

/* Selection styles */
.tile.selected {
  outline: 3px solid #ff3333;
  outline-offset: -3px;
  z-index: 10;
}

.grid-container.show-lines .tile.selected {
  outline-color: #ff3333;
}

.tile.selectable {
  cursor: pointer;
}

.tile.placeholder.selected {
  background: rgba(255, 51, 51, 0.15);
}

/* Locked tiles (during generation) */
.tile.locked {
  outline: 3px solid #a855f7;
  outline-offset: -3px;
  z-index: 10;
}

.grid-container.show-lines .tile.locked {
  outline-color: #a855f7;
}

.tile.placeholder.locked {
  background: rgba(168, 85, 247, 0.2);
}

/* Queued tiles (in queue, waiting to be generated) */
.tile.queued {
  outline: 3px dashed #c084fc !important;
  outline-offset: -3px;
  z-index: 9;
  animation: pulse-queue 1.5s ease-in-out infinite;
  /* Fallback: box-shadow for browsers that don't render dashed outlines well */
  box-shadow: inset 0 0 0 3px rgba(192, 132, 252, 0.5);
}

@keyframes pulse-queue {
  0%,
  100% {
    outline-color: #c084fc;
    box-shadow: inset 0 0 0 3px rgba(192, 132, 252, 0.5);
  }
  50% {
    outline-color: #8b5cf6;
    box-shadow: inset 0 0 0 3px rgba(139, 92, 246, 0.7);
  }
}

.grid-container.show-lines .tile.queued {
  outline-color: #c084fc;
}

.tile.placeholder.queued {
  background: rgba(168, 85, 247, 0.3) !important;
}

/* Allow selection while generating - only disable clicking on locked/queued tiles */
body.generating .tile.locked.selectable,
body.generating .tile.queued.selectable {
  cursor: not-allowed;
}

/* Selection status bar */
.selection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  background: rgba(255, 51, 51, 0.1);
  border: 1px solid rgba(255, 51, 51, 0.3);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #ff6666;
  margin-top: 8px;
}

.selection-status.empty {
  background: transparent;
  border-color: #333;
  color: #666;
}

.selection-status.generating {
  background: rgba(168, 85, 247, 0.1);
  border-color: rgba(168, 85, 247, 0.4);
  color: #a855f7;
}

/* Clickable coordinate links in status bar */
.coord-link {
  color: #00d9ff;
  text-decoration: none;
  cursor: pointer;
  padding: 1px 3px;
  border-radius: 3px;
  transition: background-color 0.15s ease;
}

.coord-link:hover {
  background-color: rgba(0, 217, 255, 0.2);
  text-decoration: underline;
}

.selection-limit {
  color: #888;
  font-size: 0.65rem;
}

.deselect-btn {
  padding: 3px 8px;
  background: #ff3333;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  margin-left: auto;
  transition: all 0.2s;
}

.deselect-btn:hover:not(:disabled) {
  background: #ff5555;
}

.deselect-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
}

.delete-btn {
  padding: 3px 8px;
  background: #dc2626;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  transition: all 0.2s;
}

.delete-btn:hover:not(:disabled) {
  background: #b91c1c;
}

.delete-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
}

.render-btn {
  padding: 4px 12px;
  background: #f59e0b;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
}

.render-btn:hover:not(:disabled) {
  background: #d97706;
}

.render-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

.render-btn.loading {
  background: #666;
  cursor: wait;
}

.render-btn .spinner {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-left: 4px;
  vertical-align: middle;
}

.generate-btn {
  padding: 4px 12px;
  background: #00d9ff;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
}

.generate-btn:hover:not(:disabled) {
  background: #00b8d4;
}

.generate-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

/* Generate Rectangle button */
.generate-rect-btn {
  padding: 4px 12px;
  background: #8b5cf6;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
}

.generate-rect-btn:hover:not(:disabled) {
  background: #7c3aed;
}

.generate-rect-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

.generate-rect-btn.loading {
  background: #666;
  cursor: wait;
}

.generate-rect-btn .spinner {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-left: 4px;
  vertical-align: middle;
}

/* Clear Queue button */
.clear-queue-btn {
  padding: 4px 10px;
  background: #6b7280;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  font-weight: 500;
  transition: all 0.2s;
  margin-left: 8px;
}

.clear-queue-btn:hover {
  background: #ef4444;
}

.clear-queue-btn:active {
  background: #dc2626;
}

/* Toast notification system */
.toast-container {
  position: fixed;
  bottom: 15px;
  right: 15px;
  z-index: 1000;
  display: flex;
  flex-direction: column-reverse;
  gap: 8px;
  max-width: 320px;
}

.toast {
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 0.75rem;
  animation: slideIn 0.3s ease-out;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.toast.success {
  background: linear-gradient(135deg, #065f46 0%, #047857 100%);
  border: 1px solid #10b981;
  color: #d1fae5;
}

.toast.error {
  background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
  border: 1px solid #ef4444;
  color: #fee2e2;
}

.toast.info {
  background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%);
  border: 1px solid #3b82f6;
  color: #dbeafe;
}

.toast.loading {
  background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
  border: 1px solid #6b7280;
  color: #e5e7eb;
}

.toast-icon {
  flex-shrink: 0;
  font-size: 1rem;
}

.toast-content {
  flex: 1;
}

.toast-title {
  font-weight: 600;
  margin-bottom: 2px;
}

.toast-message {
  opacity: 0.9;
  font-size: 0.7rem;
}

.toast-close {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  opacity: 0.7;
  font-size: 1rem;
  padding: 0;
  line-height: 1;
}

.toast-close:hover {
  opacity: 1;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@keyframes slideOut {
  from {
    transform: translateX(0);
    opacity: 1;
  }
  to {
    transform: translateX(100%);
    opacity: 0;
  }
}

.toast.removing {
  animation: slideOut 0.3s ease-in forwards;
}

/* Loading spinner for generate button */
.generate-btn.loading {
  background: #666;
  color: #fff;
  cursor: wait;
}

.generate-btn .spinner {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-left: 4px;
  vertical-align: middle;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* Water fix tool styles */
.water-fix-status {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #60a5fa;
  margin-top: 8px;
}

.water-fix-label {
  color: #93c5fd;
  font-weight: 500;
}

.color-swatch {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.3), inset 0 0 0 1px rgba(0, 0, 0, 0.1);
  background: #333;
}

.color-swatch.has-color {
  cursor: pointer;
}

.color-hex {
  font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
  font-size: 0.75rem;
  color: #93c5fd;
  min-width: 180px;
}

.water-fix-quadrant {
  color: #60a5fa;
  font-size: 0.7rem;
}

.apply-water-fix-btn {
  padding: 4px 12px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
  margin-left: auto;
}

.apply-water-fix-btn:hover:not(:disabled) {
  background: #2563eb;
}

.apply-water-fix-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

.apply-water-fix-btn.loading {
  background: #666;
  cursor: wait;
}

.cancel-water-fix-btn {
  padding: 4px 8px;
  background: #4b5563;
  color: #d1d5db;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  transition: all 0.2s;
}

.cancel-water-fix-btn:hover {
  background: #6b7280;
}

/* Fix water tool active state */
.tile.fix-water-selectable {
  cursor: crosshair;
}

.tile.fix-water-selectable:hover {
  outline: 2px solid #3b82f6;
  outline-offset: -2px;
}

.tile.water-fix-selected {
  outline: 3px solid #3b82f6;
  outline-offset: -3px;
  z-index: 10;
}

/* Water fill tool styles */
.water-fill-status {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  background: rgba(74, 99, 114, 0.15);
  border: 1px solid rgba(74, 99, 114, 0.4);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #7ba3b8;
  margin-top: 8px;
}

.water-fill-label {
  color: #93c5fd;
  font-weight: 500;
}

.water-color-preview {
  background: #4a6372 !important;
}

.water-fill-instruction {
  color: #7ba3b8;
}

.tile.water-fill-selectable {
  cursor: pointer;
}

.tile.water-fill-selectable:hover {
  outline: 2px solid #4a6372;
  outline-offset: -2px;
}

/* Model selector styles */
.model-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.model-label {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #888;
  font-size: 0.75rem;
}

.model-select {
  padding: 4px 8px;
  border: 1px solid #333;
  border-radius: 3px;
  background: #16213e;
  color: #fff;
  font-family: inherit;
  font-size: 0.75rem;
  cursor: pointer;
  min-width: 140px;
}

.model-select:hover {
  border-color: #00d9ff;
}

.model-select:focus {
  outline: none;
  border-color: #00d9ff;
  box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.2);
}

.model-select option {
  background: #16213e;
  color: #fff;
  padding: 4px;
}

/* Generate with Prompt button */
.generate-prompt-btn {
  background: linear-gradient(135deg, #9b59b6, #8e44ad);
  color: #fff;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.generate-prompt-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #a569c0, #9b59b6);
  transform: translateY(-1px);
}

.generate-prompt-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Prompt Dialog */
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.dialog-content {
  background: #1a1a2e;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 24px;
  min-width: 400px;
  max-width: 500px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}

.dialog-content h3 {
  margin: 0 0 12px 0;
  color: #fff;
  font-size: 18px;
}

.dialog-description {
  color: #aaa;
  font-size: 14px;
  margin: 0 0 16px 0;
}

.prompt-input {
  width: 100%;
  background: #0f0f1a;
  border: 1px solid #333;
  border-radius: 4px;
  color: #fff;
  font-size: 14px;
  padding: 12px;
  resize: vertical;
  font-family: inherit;
  box-sizing: border-box;
}

.prompt-input:focus {
  outline: none;
  border-color: #9b59b6;
  box-shadow: 0 0 0 2px rgba(155, 89, 182, 0.2);
}

.prompt-input::placeholder {
  color: #666;
}

.dialog-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 20px;
}

.dialog-cancel-btn {
  background: #333;
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}

.dialog-cancel-btn:hover {
  background: #444;
}

.dialog-submit-btn {
  background: linear-gradient(135deg, #9b59b6, #8e44ad);
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.dialog-submit-btn:hover {
  background: linear-gradient(135deg, #a569c0, #9b59b6);
}

/* Queue Overlays - show pending generations as grouped rectangles */
.queue-overlay {
  position: absolute;
  border: 3px dashed #c084fc;
  border-radius: 4px;
  pointer-events: auto;
  z-index: 20;
  animation: pulse-queue-overlay 1.5s ease-in-out infinite;
  box-sizing: border-box;
}

/* Processing overlay - solid border, different color */
.queue-overlay.processing {
  border-style: solid;
  border-color: #a855f7;
  animation: pulse-processing-overlay 1s ease-in-out infinite;
}

@keyframes pulse-queue-overlay {
  0%,
  100% {
    border-color: #c084fc;
    box-shadow: 0 0 12px rgba(192, 132, 252, 0.3);
  }
  50% {
    border-color: #8b5cf6;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
  }
}

@keyframes pulse-processing-overlay {
  0%,
  100% {
    border-color: #a855f7;
    box-shadow: 0 0 16px rgba(168, 85, 247, 0.4);
  }
  50% {
    border-color: #c084fc;
    box-shadow: 0 0 24px rgba(192, 132, 252, 0.7);
  }
}

.queue-overlay:hover {
  border-color: #a855f7;
  box-shadow: 0 0 16px rgba(168, 85, 247, 0.6);
}

/* Queue position badge */
.queue-badge {
  position: absolute;
  top: 6px;
  left: 6px;
  min-width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #8b5cf6, #a855f7);
  color: white;
  font-size: 12px;
  font-weight: bold;
  font-family: "SF Mono", "Monaco", monospace;
  border-radius: 6px;
  padding: 0 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  z-index: 1;
}

/* Processing badge - with spinner */
.queue-badge.processing {
  background: linear-gradient(135deg, #10b981, #059669);
  min-width: 28px;
  height: 28px;
}

/* Spinner animation for processing badge */
.processing-spinner {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* Model label below badge */
.queue-model-label {
  position: absolute;
  top: 34px;
  left: 6px;
  font-size: 9px;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.9);
  background: rgba(139, 92, 246, 0.8);
  padding: 2px 6px;
  border-radius: 3px;
  max-width: calc(100% - 40px);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
}

/* Cancel button - visible on hover */
.queue-cancel-btn {
  position: absolute;
  top: 6px;
  right: 6px;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(220, 38, 38, 0.9);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  opacity: 0;
  transform: scale(0.8);
  transition: all 0.15s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.queue-overlay:hover .queue-cancel-btn {
  opacity: 1;
  transform: scale(1);
}

.queue-cancel-btn:hover {
  background: #dc2626;
  transform: scale(1.1);
}

.queue-cancel-btn svg {
  width: 14px;
  height: 14px;
}

/* Make grid position relative for overlay positioning */
.grid {
  position: relative;
}

```

Done! Now when you click on a coordinate in the status bar (like `(5,10)` in the "🔄 ModelName [3]: (5,10)" display), it will navigate the view to center on that coordinate.

The changes:
1. **Coordinates are now clickable links** - styled in cyan with a hover effect
2. **Added `navigateToCoord(x, y)` function** - centers the view on the clicked coordinate
3. **CSS styling** - links are cyan colored and highlight on hover

