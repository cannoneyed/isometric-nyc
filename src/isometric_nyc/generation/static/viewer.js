// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";
const STORAGE_KEY_SELECTION = "viewer_selected_quadrants";
const STORAGE_KEY_PROMPT = "viewer_saved_prompt";
const STORAGE_KEY_NEGATIVE_PROMPT = "viewer_saved_negative_prompt";

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

// Save prompt to localStorage
function savePrompt(prompt) {
  try {
    if (prompt && prompt.trim()) {
      localStorage.setItem(STORAGE_KEY_PROMPT, prompt.trim());
    } else {
      localStorage.removeItem(STORAGE_KEY_PROMPT);
    }
    updatePromptButtonIndicator();
  } catch (e) {
    console.warn("Could not save prompt to localStorage:", e);
  }
}

// Get saved prompt from localStorage
function getSavedPrompt() {
  try {
    return localStorage.getItem(STORAGE_KEY_PROMPT) || "";
  } catch (e) {
    return "";
  }
}

// Clear saved prompt
function clearSavedPrompt() {
  try {
    localStorage.removeItem(STORAGE_KEY_PROMPT);
    updatePromptButtonIndicator();
    showToast("info", "Prompt cleared", "Saved prompt has been removed");
  } catch (e) {
    console.warn("Could not clear prompt from localStorage:", e);
  }
}

// Update the prompt button to show indicator when a prompt is saved
function updatePromptButtonIndicator() {
  const btn = document.getElementById("generateWithPromptBtn");
  if (!btn) return;

  const savedPrompt = getSavedPrompt();
  if (savedPrompt) {
    btn.classList.add("has-saved-prompt");
    btn.title = `Generate with custom prompt (overrides default): "${savedPrompt.substring(0, 50)}${savedPrompt.length > 50 ? '...' : ''}"`;
    btn.innerHTML = '+ Prompt <span class="prompt-indicator">●</span>';
  } else {
    btn.classList.remove("has-saved-prompt");
    btn.title = "Generate with custom prompt (overrides default)";
    btn.textContent = "+ Prompt";
  }
}

// Save negative_prompt to localStorage
function saveNegativePrompt(negativePrompt) {
  try {
    if (negativePrompt && negativePrompt.trim()) {
      localStorage.setItem(STORAGE_KEY_NEGATIVE_PROMPT, negativePrompt.trim());
      console.log("💾 Saved negative prompt:", negativePrompt.trim());
    } else {
      localStorage.removeItem(STORAGE_KEY_NEGATIVE_PROMPT);
      console.log("🗑️ Cleared negative prompt");
    }
    updateNegativePromptButtonIndicator();
  } catch (e) {
    console.warn("Could not save negative_prompt to localStorage:", e);
  }
}

// Get saved negative_prompt from localStorage
function getSavedNegativePrompt() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_NEGATIVE_PROMPT) || "";
    console.log("📖 Retrieved negative prompt:", saved || "(empty)");
    return saved;
  } catch (e) {
    console.warn("Could not get negative_prompt from localStorage:", e);
    return "";
  }
}

// Clear saved negative_prompt
function clearSavedNegativePrompt() {
  try {
    localStorage.removeItem(STORAGE_KEY_NEGATIVE_PROMPT);
    updateNegativePromptButtonIndicator();
    showToast("info", "Negative prompt cleared", "Saved negative prompt has been removed");
  } catch (e) {
    console.warn("Could not clear negative_prompt from localStorage:", e);
  }
}

// Update the negative_prompt button to show indicator when a negative_prompt is saved
function updateNegativePromptButtonIndicator() {
  const btn = document.getElementById("generateWithNegativePromptBtn");
  if (!btn) return;

  const savedNegativePrompt = getSavedNegativePrompt();
  if (savedNegativePrompt) {
    btn.classList.add("has-saved-negative-prompt");
    btn.title = `Generate with negative prompt: "${savedNegativePrompt.substring(0, 50)}${savedNegativePrompt.length > 50 ? '...' : ''}"`; btn.innerHTML = '- Neg Prompt <span class="prompt-indicator">●</span>';
  } else {
    btn.classList.remove("has-saved-negative-prompt");
    btn.title = "Generate with negative prompt";
    btn.textContent = "- Neg Prompt";
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
  const tileType = document.getElementById("tileTypeSelect")?.value || "generation";
  return { x, y, nx, ny, sizePx, showLines, showCoords, tileType };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, tileType } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&tile_type=${tileType}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&tile_type=${params.tileType}`;
}

// Navigate to center the view on a specific coordinate
function navigateToCoord(targetX, targetY) {
  const params = getParams();
  const nx = parseInt(params.nx);
  const ny = parseInt(params.ny);
  // Center the target coordinate in the view
  const x = targetX - Math.floor(nx / 2);
  const y = targetY - Math.floor(ny / 2);
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&tile_type=${params.tileType}`;
}

// Hard refresh - clear image cache and reload page
function hardRefresh() {
  // Add cache-busting timestamp to all tile images to force reload
  const timestamp = Date.now();

  // Update all tile image sources to bust cache
  document.querySelectorAll(".tile img").forEach((img) => {
    const url = new URL(img.src);
    url.searchParams.set("_t", timestamp);
    img.src = url.toString();
  });

  // Also reload the page with cache bypass
  // The true parameter forces reload from server, not cache
  window.location.reload(true);
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

function changeTileType() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, tileType } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&tile_type=${tileType}`;
}

function cycleTileType() {
  // Cycle through tile types: generation -> render -> water_mask -> dark_mode -> generation
  const select = document.getElementById("tileTypeSelect");
  if (!select) return;

  const types = ["generation", "render", "water_mask", "dark_mode"];
  const currentIndex = types.indexOf(select.value);
  const nextIndex = (currentIndex + 1) % types.length;
  select.value = types[nextIndex];
  changeTileType();
}

function toggleWaterHighlight() {
  const container = document.getElementById("gridContainer");
  const showWater = document.getElementById("showWater").checked;
  container.classList.toggle("show-water-highlight", showWater);

  // Save preference to localStorage
  try {
    localStorage.setItem("viewer_show_water_highlight", showWater ? "1" : "0");
  } catch (e) {
    console.warn("Could not save water highlight preference:", e);
  }
}

// Initialize water highlight state from localStorage
function initWaterHighlight() {
  try {
    const saved = localStorage.getItem("viewer_show_water_highlight");
    if (saved === "1") {
      const checkbox = document.getElementById("showWater");
      if (checkbox) {
        checkbox.checked = true;
        toggleWaterHighlight();
      }
    }
  } catch (e) {
    // Ignore localStorage errors
  }
}

// Toggle labels (starred/flagged indicators and outlines)
function toggleLabels() {
  const container = document.getElementById("gridContainer");
  const showLabels = document.getElementById("showLabels")?.checked ?? true;
  container.classList.toggle("hide-labels", !showLabels);

  // Save preference to localStorage
  try {
    localStorage.setItem("viewer_show_labels", showLabels ? "1" : "0");
  } catch (e) {
    console.warn("Could not save labels preference:", e);
  }
}

// Initialize labels state from localStorage
function initLabels() {
  try {
    const saved = localStorage.getItem("viewer_show_labels");
    // Default to showing labels if not set (checkbox is checked by default in HTML)
    if (saved === "0") {
      const checkbox = document.getElementById("showLabels");
      if (checkbox) {
        checkbox.checked = false;
        toggleLabels();
      }
    }
  } catch (e) {
    // Ignore localStorage errors
  }
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  // Ignore keyboard shortcuts when typing in input fields
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

  // Ignore keyboard shortcuts when prompt dialog is open
  const promptDialog = document.getElementById("promptDialog");
  if (promptDialog && promptDialog.style.display !== "none") return;

  // Ignore keyboard shortcuts when negative prompt dialog is open
  const negativePromptDialog = document.getElementById("negativePromptDialog");
  if (negativePromptDialog && negativePromptDialog.style.display !== "none") return;

  switch (e.key) {
    case "ArrowLeft":
      e.preventDefault();
      navigate(-1, 0);
      break;
    case "ArrowRight":
      e.preventDefault();
      navigate(1, 0);
      break;
    case "ArrowUp":
      e.preventDefault();
      navigate(0, -1);
      break;
    case "ArrowDown":
      e.preventDefault();
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
      // Cycle through tile types: generation -> render -> water_mask -> generation
      cycleTileType();
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
    case "t":
    case "T":
      toggleWaterSelectTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      if (waterSelectToolActive) cancelWaterSelect();
      break;
    case "[":
      goToPrevStarred();
      break;
    case "]":
      goToNextStarred();
      break;
    case "b":
    case "B":
      referenceSelected();
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
  if (waterSelectToolActive) {
    cancelWaterSelect();
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
  if (waterSelectToolActive) {
    cancelWaterSelect();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar
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
    // Hide water fix status bar
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
  if (waterSelectToolActive) {
    cancelWaterSelect();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar
    waterFillStatus.style.display = "flex";
    saveSelectedTool("waterfill");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar
    waterFillStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

// Water Select tool - marks quadrants as water tiles
let waterSelectToolActive = false;

function toggleWaterSelectTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }
  if (waterFillToolActive) {
    cancelWaterFill();
  }

  waterSelectToolActive = !waterSelectToolActive;
  const btn = document.getElementById("waterSelectTool");
  const tiles = document.querySelectorAll(".tile");
  const waterSelectStatus = document.getElementById("waterSelectStatus");

  if (waterSelectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-select-selectable");
    });
    // Show water select status bar
    waterSelectStatus.style.display = "flex";
    saveSelectedTool("waterselect");
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-select-selectable");
    });
    // Hide water select status bar
    waterSelectStatus.style.display = "none";
    saveSelectedTool("");
  }
}

function cancelWaterSelect() {
  if (waterSelectToolActive) {
    toggleWaterSelectTool();
  }
}

async function handleWaterSelectClick(tileEl) {
  if (!waterSelectToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Three-state cycle: unset (0) → water (1) → explicit not water (-1) → unset (0)
  const isCurrentlyWater = tileEl.dataset.water === "true";
  const isExplicitNotWater = tileEl.dataset.explicitNotWater === "true";

  let requestBody;
  let expectedState;

  if (isExplicitNotWater) {
    // Currently explicit not water → go to unset (0)
    requestBody = { quadrants: [[qx, qy]], is_water: false };
    expectedState = "unset";
  } else if (isCurrentlyWater) {
    // Currently water → go to explicit not water (-1)
    requestBody = { quadrants: [[qx, qy]], explicit_not_water: true };
    expectedState = "explicit_not_water";
  } else {
    // Currently unset → go to water (1)
    requestBody = { quadrants: [[qx, qy]], is_water: true };
    expectedState = "water";
  }

  const instruction = document.getElementById("waterSelectInstruction");
  instruction.textContent = `Updating (${qx}, ${qy})...`;

  try {
    const response = await fetch("/api/water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    const result = await response.json();

    if (result.success) {
      // Update the tile's visual state based on the new status
      updateTileWaterState(tileEl, result.water_status);

      const messages = {
        water: "Marked as water 💧",
        explicit_not_water: "Protected from auto-detection 🛡️",
        unset: "Reset to auto-detect",
      };
      showToast(
        expectedState === "water" ? "success" : "info",
        messages[expectedState],
        `Quadrant (${qx}, ${qy})`
      );

      instruction.textContent =
        "Click to cycle: unset → water → protected → unset";
    } else {
      showToast("error", "Failed to update", result.error || "Unknown error");
      instruction.textContent =
        "Click to cycle: unset → water → protected → unset";
    }
  } catch (error) {
    console.error("Water select error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent =
      "Click to cycle: unset → water → protected → unset";
  }
}

// Update a tile's visual state based on water_status value
function updateTileWaterState(tileEl, waterStatus) {
  // Remove all water-related classes and indicators
  tileEl.classList.remove("water", "explicit-not-water");
  tileEl.dataset.water = "false";
  tileEl.dataset.explicitNotWater = "false";

  const waterIndicator = tileEl.querySelector(".water-indicator");
  if (waterIndicator) waterIndicator.remove();

  const notWaterIndicator = tileEl.querySelector(".explicit-not-water-indicator");
  if (notWaterIndicator) notWaterIndicator.remove();

  if (waterStatus === 1) {
    // Water tile
    tileEl.classList.add("water");
    tileEl.dataset.water = "true";
    const indicator = document.createElement("span");
    indicator.className = "water-indicator";
    indicator.title = "Water tile";
    indicator.textContent = "💧";
    tileEl.appendChild(indicator);
  } else if (waterStatus === -1) {
    // Explicit not water (protected)
    tileEl.classList.add("explicit-not-water");
    tileEl.dataset.explicitNotWater = "true";
    const indicator = document.createElement("span");
    indicator.className = "explicit-not-water-indicator";
    indicator.title = "Explicitly NOT water (protected)";
    indicator.textContent = "🛡️";
    tileEl.appendChild(indicator);
  }
  // waterStatus === 0: unset, no visual indicator needed
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
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const flagBtn = document.getElementById("flagBtn");
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
  if (countEl) {
    if (statusParts.length > 0) {
      const statusHtml = statusParts
        .map((part) =>
          typeof part === "object" && part.html ? part.html : part
        )
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
    } else if (count > 0) {
      countEl.textContent = `${count} selected`;
    } else {
      countEl.textContent = "";
    }
  }

  // Update status bar styling and visibility
  const isProcessing =
    serverStatus &&
    (serverStatus.is_generating || serverStatus.queue_length > 0);
  if (statusEl) {
    if (isProcessing) {
      statusEl.classList.add("generating");
      statusEl.style.display = "";
    } else {
      statusEl.classList.remove("generating");
      // Hide the status row if there's no content
      const hasContent = countEl && countEl.textContent.trim() !== "";
      statusEl.style.display = hasContent ? "" : "none";
    }
  }

  // Enable buttons for selection (can add to queue even during processing)
  if (deselectBtn) deselectBtn.disabled = count === 0;
  if (deleteBtn) deleteBtn.disabled = count === 0;
  if (flagBtn) flagBtn.disabled = count === 0;
  if (renderBtn) renderBtn.disabled = count === 0;
  if (generateBtn) generateBtn.disabled = count === 0;
  // Generate with prompt button
  const generateWithPromptBtn = document.getElementById(
    "generateWithPromptBtn"
  );
  if (generateWithPromptBtn) {
    generateWithPromptBtn.disabled = count === 0;
  }
  // Generate with negative prompt button
  const generateWithNegativePromptBtn = document.getElementById(
    "generateWithNegativePromptBtn"
  );
  if (generateWithNegativePromptBtn) {
    generateWithNegativePromptBtn.disabled = count === 0;
  }
  // Star button requires exactly 1 selected
  const starBtn = document.getElementById("starBtn");
  if (starBtn) starBtn.disabled = count !== 1;
  // Reference button requires exactly 1 selected
  const referenceBtn = document.getElementById("referenceBtn");
  if (referenceBtn) referenceBtn.disabled = count !== 1;
  // Generate Rectangle requires exactly 2 selected
  if (generateRectBtn) generateRectBtn.disabled = count !== 2;
  // Fill Rect Water requires exactly 2 selected
  const fillRectWaterBtn = document.getElementById("fillRectWaterBtn");
  if (fillRectWaterBtn) fillRectWaterBtn.disabled = count !== 2;
  // Export Cmd requires exactly 2 selected
  const exportCmdBtn = document.getElementById("exportCmdBtn");
  if (exportCmdBtn) exportCmdBtn.disabled = count !== 2;
  // Export requires exactly 2 selected
  const exportBtn = document.getElementById("exportBtn");
  if (exportBtn) exportBtn.disabled = count !== 2;
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

  // Check which view mode we're in
  const tileType = document.getElementById("tileTypeSelect")?.value || "generation";
  let dataType, apiEndpoint;
  if (tileType === "render") {
    dataType = "render";
    apiEndpoint = "/api/delete-render";
  } else if (tileType === "water_mask") {
    dataType = "water mask";
    apiEndpoint = "/api/delete-water-mask";
  } else if (tileType === "dark_mode") {
    dataType = "dark mode";
    apiEndpoint = "/api/delete-dark-mode";
  } else {
    dataType = "generation";
    apiEndpoint = "/api/delete";
  }

  let quadrantsToDelete = coords;

  // If exactly 2 quadrants selected, offer to delete the full rectangle
  if (coords.length === 2) {
    const minX = Math.min(coords[0][0], coords[1][0]);
    const maxX = Math.max(coords[0][0], coords[1][0]);
    const minY = Math.min(coords[0][1], coords[1][1]);
    const maxY = Math.max(coords[0][1], coords[1][1]);

    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    const totalQuadrants = width * height;

    // Only offer rectangle deletion if it would include more than the 2 selected
    if (totalQuadrants > 2) {
      const rectangleChoice = confirm(
        `You've selected 2 corners defining a ${width}×${height} rectangle.\n\n` +
        `Do you want to delete ALL ${totalQuadrants} quadrant(s) in this rectangle?\n\n` +
        `Click OK to delete the full rectangle.\n` +
        `Click Cancel to delete only the 2 selected quadrants.`
      );

      if (rectangleChoice) {
        // Build array of all quadrants in the rectangle
        quadrantsToDelete = [];
        for (let x = minX; x <= maxX; x++) {
          for (let y = minY; y <= maxY; y++) {
            quadrantsToDelete.push([x, y]);
          }
        }
        console.log(`Deleting rectangle from (${minX},${minY}) to (${maxX},${maxY}): ${quadrantsToDelete.length} quadrants`);
      }
    }
  }

  // Confirm deletion
  const coordsStr = quadrantsToDelete.length <= 4
    ? quadrantsToDelete.map(([x, y]) => `(${x},${y})`).join(", ")
    : `${quadrantsToDelete.length} quadrants`;
  if (!confirm(`Delete ${dataType} data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch(apiEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: quadrantsToDelete }),
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

async function flagSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Check if any selected tiles are already flagged - if so, unflag them
  let anyFlagged = false;
  coords.forEach(([x, y]) => {
    const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
    if (tile && tile.dataset.flagged === "true") {
      anyFlagged = true;
    }
  });

  // Toggle: if any are flagged, unflag all; otherwise flag all
  const shouldFlag = !anyFlagged;

  try {
    const response = await fetch("/api/flag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords, flag: shouldFlag }),
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        result.flagged ? "Flagged" : "Unflagged",
        result.message
      );

      // Update tile visual state
      coords.forEach(([x, y]) => {
        const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
        if (tile) {
          if (shouldFlag) {
            tile.classList.add("flagged");
            tile.dataset.flagged = "true";
          } else {
            tile.classList.remove("flagged");
            tile.dataset.flagged = "false";
          }
        }
      });

      // Deselect after flagging
      deselectAll();
    } else {
      showToast("error", "Flag failed", result.error);
    }
  } catch (error) {
    console.error("Flag error:", error);
    showToast("error", "Flag failed", error.message);
  }
}

async function starSelected() {
  // Only allow starring exactly 1 quadrant
  if (selectedQuadrants.size !== 1) {
    showToast("error", "Invalid selection", "Select exactly 1 quadrant to star");
    return;
  }

  const coordKey = Array.from(selectedQuadrants)[0];
  const [x, y] = coordKey.split(",").map(Number);
  const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
  
  // Toggle: if already starred, unstar; otherwise star
  const isCurrentlyStarred = tile && tile.dataset.starred === "true";
  const shouldStar = !isCurrentlyStarred;

  try {
    const response = await fetch("/api/star", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrant: [x, y], star: shouldStar }),
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        result.starred ? "⭐ Starred" : "Unstarred",
        result.message
      );

      // Invalidate cached starred list
      cachedStarredList = null;
      currentStarredIndex = -1;

      // Update tile visual state
      if (tile) {
        if (shouldStar) {
          tile.classList.add("starred");
          tile.dataset.starred = "true";
          // Add star indicator if not present
          if (!tile.querySelector(".starred-indicator")) {
            const indicator = document.createElement("span");
            indicator.className = "starred-indicator";
            indicator.title = "Starred for dataset";
            indicator.textContent = "⭐";
            tile.appendChild(indicator);
          }
        } else {
          tile.classList.remove("starred");
          tile.dataset.starred = "false";
          // Remove star indicator
          const indicator = tile.querySelector(".starred-indicator");
          if (indicator) indicator.remove();
        }
      }

      // Deselect after starring
      deselectAll();
    } else {
      showToast("error", "Star failed", result.error);
    }
  } catch (error) {
    console.error("Star error:", error);
    showToast("error", "Star failed", error.message);
  }
}

// Reference tile function
async function referenceSelected() {
  // Only allow marking exactly 1 quadrant as reference (top-left of 2x2)
  if (selectedQuadrants.size !== 1) {
    showToast("warning", "Invalid selection", "Select exactly 1 quadrant (top-left of 2x2 reference tile)");
    return;
  }

  const coordKey = Array.from(selectedQuadrants)[0];
  const [x, y] = coordKey.split(",").map(Number);
  const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);

  // Toggle: if already reference, unmark; otherwise mark
  const isCurrentlyReference = tile && tile.dataset.isReference === "true";
  const shouldMark = !isCurrentlyReference;

  try {
    const response = await fetch("/api/reference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrant: [x, y], reference: shouldMark }),
    });

    const result = await response.json();

    if (result.success) {
      showToast(
        "success",
        shouldMark ? "🍌 Marked as reference" : "Unmarked as reference",
        result.message
      );

      // Update tile visual state
      if (tile) {
        if (shouldMark) {
          tile.classList.add("is-reference");
          tile.dataset.isReference = "true";
          // Add reference indicator if not present
          if (!tile.querySelector(".reference-indicator")) {
            const indicator = document.createElement("span");
            indicator.className = "reference-indicator";
            indicator.title = "Reference tile (top-left of 2x2)";
            indicator.textContent = "🍌";
            tile.appendChild(indicator);
          }
        } else {
          tile.classList.remove("is-reference");
          tile.dataset.isReference = "false";
          // Remove reference indicator
          const indicator = tile.querySelector(".reference-indicator");
          if (indicator) indicator.remove();
        }
      }

      // Deselect after marking
      deselectAll();
    } else {
      showToast("error", "Reference failed", result.error);
    }
  } catch (error) {
    console.error("Reference error:", error);
    showToast("error", "Reference failed", error.message);
  }
}

// Clear all references function
async function clearAllReferences() {
  // Confirm with user
  if (!confirm("Clear all reference tiles? This will remove all 🍌 markers.")) {
    return;
  }

  try {
    const response = await fetch("/api/references/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "References cleared", result.message);

      // Remove all reference indicators from tiles
      const referenceTiles = document.querySelectorAll(".tile.is-reference");
      referenceTiles.forEach((tile) => {
        tile.classList.remove("is-reference");
        tile.dataset.isReference = "false";
        const indicator = tile.querySelector(".reference-indicator");
        if (indicator) indicator.remove();
      });
    } else {
      showToast("error", "Clear failed", result.error);
    }
  } catch (error) {
    console.error("Clear references error:", error);
    showToast("error", "Clear failed", error.message);
  }
}

// Starred entries dialog
async function showStarredDialog() {
  const dialog = document.getElementById("starredDialog");
  const listContainer = document.getElementById("starredList");
  const emptyState = document.getElementById("starredEmptyState");
  const countDisplay = document.getElementById("starredCountDisplay");
  const listContainerWrapper = document.getElementById("starredListContainer");

  if (!dialog || !listContainer) return;

  // Show dialog immediately with loading state
  dialog.style.display = "flex";
  listContainer.innerHTML = '<div style="text-align: center; padding: 20px; color: #888;">Loading...</div>';

  try {
    const response = await fetch("/api/starred");
    const result = await response.json();

    if (result.success) {
      const starred = result.starred || [];
      countDisplay.textContent = `${starred.length} starred`;

      if (starred.length === 0) {
        listContainerWrapper.style.display = "none";
        emptyState.style.display = "block";
      } else {
        listContainerWrapper.style.display = "block";
        emptyState.style.display = "none";
        
        // Build list HTML
        listContainer.innerHTML = starred.map(entry => `
          <div class="starred-entry" data-x="${entry.x}" data-y="${entry.y}" onclick="navigateToStarred(${entry.x}, ${entry.y})">
            <div class="starred-entry-coords">
              <span class="star-icon">⭐</span>
              <span class="coords-text">(${entry.x}, ${entry.y})</span>
            </div>
            <div class="starred-entry-status">
              ${entry.has_generation ? '<span class="has-gen">✓ generation</span>' : ''}
              ${entry.has_render ? '<span class="has-render">✓ render</span>' : ''}
            </div>
            <div class="starred-entry-actions">
              <button class="starred-unstar-btn" onclick="event.stopPropagation(); unstarFromDialog(${entry.x}, ${entry.y})">Unstar</button>
            </div>
          </div>
        `).join('');
      }
    } else {
      showToast("error", "Failed to load starred", result.error);
      dialog.style.display = "none";
    }
  } catch (error) {
    console.error("Load starred error:", error);
    showToast("error", "Failed to load starred", error.message);
    dialog.style.display = "none";
  }
}

function hideStarredDialog() {
  const dialog = document.getElementById("starredDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

function navigateToStarred(x, y) {
  hideStarredDialog();
  navigateToCoord(x, y);
}

async function unstarFromDialog(x, y) {
  try {
    const response = await fetch("/api/star", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrant: [x, y], star: false }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Unstarred", result.message);
      
      // Update tile if visible
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        tile.classList.remove("starred");
        tile.dataset.starred = "false";
        const indicator = tile.querySelector(".starred-indicator");
        if (indicator) indicator.remove();
      }
      
      // Invalidate cached starred list
      cachedStarredList = null;
      currentStarredIndex = -1;
      
      // Refresh the dialog
      showStarredDialog();
    } else {
      showToast("error", "Unstar failed", result.error);
    }
  } catch (error) {
    console.error("Unstar error:", error);
    showToast("error", "Unstar failed", error.message);
  }
}

// Starred navigation state
let cachedStarredList = null;
let currentStarredIndex = -1;

// Fetch and cache the starred list
async function fetchStarredList() {
  try {
    const response = await fetch("/api/starred");
    const result = await response.json();
    if (result.success) {
      cachedStarredList = result.starred || [];
      return cachedStarredList;
    }
  } catch (error) {
    console.error("Failed to fetch starred list:", error);
  }
  return [];
}

// Find the index of a coordinate in the starred list
function findStarredIndex(x, y) {
  if (!cachedStarredList) return -1;
  return cachedStarredList.findIndex(entry => entry.x === x && entry.y === y);
}

// Get the current view center coordinates
function getCurrentViewCenter() {
  const params = getParams();
  const centerX = parseInt(params.x) + Math.floor(parseInt(params.nx) / 2);
  const centerY = parseInt(params.y) + Math.floor(parseInt(params.ny) / 2);
  return { x: centerX, y: centerY };
}

// Find the nearest starred quadrant to current view
function findNearestStarredIndex() {
  if (!cachedStarredList || cachedStarredList.length === 0) return -1;
  
  const center = getCurrentViewCenter();
  let nearestIndex = 0;
  let nearestDist = Infinity;
  
  cachedStarredList.forEach((entry, index) => {
    const dist = Math.abs(entry.x - center.x) + Math.abs(entry.y - center.y);
    if (dist < nearestDist) {
      nearestDist = dist;
      nearestIndex = index;
    }
  });
  
  return nearestIndex;
}

// Go to next starred quadrant
async function goToNextStarred() {
  // Fetch list if not cached
  if (!cachedStarredList) {
    await fetchStarredList();
  }
  
  if (!cachedStarredList || cachedStarredList.length === 0) {
    showToast("info", "No starred entries", "Star some quadrants first");
    return;
  }
  
  // If no current index, find nearest
  if (currentStarredIndex < 0) {
    currentStarredIndex = findNearestStarredIndex();
  }
  
  // Move to next (wrap around)
  currentStarredIndex = (currentStarredIndex + 1) % cachedStarredList.length;
  
  const entry = cachedStarredList[currentStarredIndex];
  showToast("info", `⭐ ${currentStarredIndex + 1}/${cachedStarredList.length}`, `Going to (${entry.x}, ${entry.y})`);
  navigateToCoord(entry.x, entry.y);
}

// Go to previous starred quadrant
async function goToPrevStarred() {
  // Fetch list if not cached
  if (!cachedStarredList) {
    await fetchStarredList();
  }
  
  if (!cachedStarredList || cachedStarredList.length === 0) {
    showToast("info", "No starred entries", "Star some quadrants first");
    return;
  }
  
  // If no current index, find nearest
  if (currentStarredIndex < 0) {
    currentStarredIndex = findNearestStarredIndex();
  }
  
  // Move to previous (wrap around)
  currentStarredIndex = (currentStarredIndex - 1 + cachedStarredList.length) % cachedStarredList.length;
  
  const entry = cachedStarredList[currentStarredIndex];
  showToast("info", `⭐ ${currentStarredIndex + 1}/${cachedStarredList.length}`, `Going to (${entry.x}, ${entry.y})`);
  navigateToCoord(entry.x, entry.y);
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
  const savedPromptDisplay = document.getElementById("savedPromptDisplay");
  const clearPromptBtn = document.getElementById("clearPromptBtn");
  
  if (dialog && input) {
    // Pre-fill with saved prompt if one exists
    const savedPrompt = getSavedPrompt();
    input.value = savedPrompt;
    
    // Update saved prompt display
    if (savedPromptDisplay && clearPromptBtn) {
      if (savedPrompt) {
        savedPromptDisplay.textContent = `Saved: "${savedPrompt.substring(0, 60)}${savedPrompt.length > 60 ? '...' : ''}"`;
        savedPromptDisplay.style.display = "";
        clearPromptBtn.style.display = "";
      } else {
        savedPromptDisplay.style.display = "none";
        clearPromptBtn.style.display = "none";
      }
    }
    
    dialog.style.display = "flex";
    input.focus();
    input.select();
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
  
  // Save the prompt for future generations
  if (prompt) {
    savePrompt(prompt);
    showToast("success", "Prompt saved", `"${prompt.substring(0, 40)}${prompt.length > 40 ? '...' : ''}" will be applied to future generations`);
  }
  
  hidePromptDialog();
  await generateSelected(prompt);
}

// Negative Prompt dialog functions
function showNegativePromptDialog() {
  if (selectedQuadrants.size === 0) return;
  const dialog = document.getElementById("negativePromptDialog");
  const input = document.getElementById("negativePromptInput");
  const savedNegativePromptDisplay = document.getElementById("savedNegativePromptDisplay");
  const clearNegativePromptBtn = document.getElementById("clearNegativePromptBtn");

  if (dialog && input) {
    // Pre-fill with saved negative prompt if one exists
    const savedNegativePrompt = getSavedNegativePrompt();
    input.value = savedNegativePrompt;

    // Update saved negative prompt display
    if (savedNegativePromptDisplay && clearNegativePromptBtn) {
      if (savedNegativePrompt) {
        savedNegativePromptDisplay.textContent = `Saved: "${savedNegativePrompt.substring(0, 60)}${savedNegativePrompt.length > 60 ? '...' : ''}"`;
        savedNegativePromptDisplay.style.display = "";
        clearNegativePromptBtn.style.display = "";
      } else {
        savedNegativePromptDisplay.style.display = "none";
        clearNegativePromptBtn.style.display = "none";
      }
    }

    dialog.style.display = "flex";
    input.focus();
    input.select();
  }
}

function hideNegativePromptDialog() {
  const dialog = document.getElementById("negativePromptDialog");
  if (dialog) {
    dialog.style.display = "none";
  }
}

async function submitNegativePromptGeneration() {
  const input = document.getElementById("negativePromptInput");
  const negativePrompt = input ? input.value.trim() : "";

  // Save the negative prompt for future generations
  if (negativePrompt) {
    saveNegativePrompt(negativePrompt);
    showToast("success", "Negative prompt saved", `"${negativePrompt.substring(0, 40)}${negativePrompt.length > 40 ? '...' : ''}" will be applied to future generations`);
  }

  hideNegativePromptDialog();
  await generateSelected();
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

  // Use saved prompt if no explicit prompt provided
  const effectivePrompt = prompt !== null ? prompt : getSavedPrompt();

  // Always use saved negative_prompt
  const effectiveNegativePrompt = getSavedNegativePrompt();

  const modelId = getSelectedModelId();

  console.log(
    "Generate requested for:",
    toGenerate,
    "with context:",
    contextQuadrants,
    "model:",
    modelId,
    "prompt:",
    effectivePrompt || "(none)",
    "negative_prompt:",
    effectiveNegativePrompt || "(none)"
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();

  // Build context info for toast
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = effectivePrompt ? " with prompt" : "";

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

    // Include prompt if provided (either explicit or saved)
    if (effectivePrompt) {
      requestBody.prompt = effectivePrompt;
    }

    // Include negative_prompt if provided (from saved)
    if (effectiveNegativePrompt) {
      requestBody.negative_prompt = effectiveNegativePrompt;
    }

    console.log("🚀 Sending request body:", JSON.stringify(requestBody, null, 2));

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
        `${toGenerate.length} quadrant(s)${promptMsg}${contextMsg} → position ${result.position}${modelInfo}`
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
        `${coords.length} quadrant(s) for render → position ${result.position}`
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

async function fillRectangleWater() {
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

  // Build confirmation message (similar to delete)
  const confirmMessage =
    `Fill rectangle from (${minX}, ${minY}) to (${maxX}, ${maxY}) with water color?\n\n` +
    `Size: ${width} × ${height} = ${totalQuadrants} quadrant(s)\n\n` +
    `⚠️ This will OVERWRITE any existing generations in this area with solid water color (#4A6372).`;

  if (!confirm(confirmMessage)) {
    return;
  }

  console.log(
    `Filling rectangle with water from (${minX},${minY}) to (${maxX},${maxY}): ${totalQuadrants} quadrants`
  );

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();
  saveSelectedQuadrants();
  updateSelectionStatus();

  // Show immediate feedback
  const btn = document.getElementById("fillRectWaterBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  const originalText = btn.innerHTML;
  btn.innerHTML = 'Filling...<span class="spinner"></span>';

  showToast(
    "loading",
    "Filling with water...",
    `Rectangle (${minX}, ${minY}) to (${maxX}, ${maxY}) - ${totalQuadrants} quadrant(s)`
  );

  try {
    const response = await fetch("/api/water-fill-rectangle", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = originalText;

    if (result.success) {
      showToast(
        "success",
        "Water fill complete!",
        `Filled ${result.filled_count} quadrant(s) with water color`
      );

      // Refresh tile images for all quadrants in the rectangle
      for (let dy = 0; dy < height; dy++) {
        for (let dx = 0; dx < width; dx++) {
          const qx = minX + dx;
          const qy = minY + dy;
          refreshTileImage(qx, qy);
        }
      }
    } else {
      showToast("error", "Fill failed", result.error || "Unknown error");
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill rectangle error:", error);
    showToast("error", "Request failed", error.message);

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = originalText;
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

async function copyExportCommand() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the export bounds."
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

  // Build the export command
  const command = `uv run python src/isometric_nyc/generation/export_import_generation_tile.py generations/v01 --tl='${minX},${minY}' --br='${maxX},${maxY}' --overwrite`;

  try {
    await navigator.clipboard.writeText(command);
    showToast(
      "success",
      "Command copied!",
      `Export command for (${minX},${minY}) to (${maxX},${maxY}) copied to clipboard`
    );
    console.log("Copied export command:", command);
  } catch (error) {
    console.error("Failed to copy to clipboard:", error);
    showToast(
      "error",
      "Copy failed",
      "Could not copy to clipboard. Check browser permissions."
    );
  }
}

async function exportSelected() {
  if (selectedQuadrants.size !== 2) {
    showToast(
      "error",
      "Invalid selection",
      "Please select exactly 2 quadrants to define the export bounds."
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

  // Check if we're in render view mode
  const tileType = document.getElementById("tileTypeSelect")?.value || "generation";
  const useRender = tileType === "render";
  const dataType = useRender ? "renders" : "generations";

  console.log(
    `Exporting ${dataType} from (${minX},${minY}) to (${maxX},${maxY}) (${width}x${height} = ${totalQuadrants} quadrants)`
  );

  // Show loading state
  const btn = document.getElementById("exportBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = 'Exporting...<span class="spinner"></span>';

  showToast(
    "loading",
    "Exporting...",
    `Creating ${width}×${height} image from ${totalQuadrants} quadrant(s)`
  );

  try {
    const response = await fetch("/api/export", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tl: [minX, minY],
        br: [maxX, maxY],
        use_render: useRender,
      }),
    });

    clearLoadingToasts();

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Export failed");
    }

    // Get the filename from Content-Disposition header or create one
    const contentDisposition = response.headers.get("Content-Disposition");
    let filename = `export_tl_${minX}_${minY}_br_${maxX}_${maxY}.png`;
    if (contentDisposition) {
      const match = contentDisposition.match(/filename=(.+)/);
      if (match) {
        filename = match[1];
      }
    }

    // Download the file
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    showToast(
      "success",
      "Export complete!",
      `Downloaded ${filename} (${width}×${height} quadrants)`
    );
    console.log("Export downloaded:", filename);

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Export";
    btn.disabled = selectedQuadrants.size !== 2;
  } catch (error) {
    clearLoadingToasts();
    console.error("Export error:", error);
    showToast("error", "Export failed", error.message);

    // Reset button
    btn.classList.remove("loading");
    btn.innerHTML = "Export";
    btn.disabled = selectedQuadrants.size !== 2;
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

  // Check if this tile is currently being generated (locked = actively processing)
  // Note: queued tiles CAN be selected (e.g., to cancel or manage them)
  const key = `${qx},${qy}`;
  if (tileEl.classList.contains("locked")) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently being processed`
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

    // Handle water select tool clicks
    if (waterSelectToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterSelectClick(tile);
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
let lastProcessingQuadrants = new Set(); // Track quadrants that were processing

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

// Refresh a specific tile's image (or add image if it was a placeholder)
function refreshTileImage(qx, qy) {
  const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
  if (!tile) return;

  const tileType = document.getElementById("tileTypeSelect")?.value || "generation";
  const timestamp = Date.now();
  const imgUrl = `/tile/${qx}/${qy}?tile_type=${tileType}&_t=${timestamp}`;

  let img = tile.querySelector("img");
  if (img) {
    // Update existing image
    img.src = imgUrl;
  } else {
    // Create new image for placeholder tile
    img = document.createElement("img");
    img.src = imgUrl;
    img.alt = `Tile ${qx},${qy}`;
    img.onload = () => {
      // Remove placeholder class once image loads
      tile.classList.remove("placeholder");
    };
    tile.appendChild(img);
  }
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

    // Get current processing quadrants
    const currentProcessingQuadrants = new Set(
      (status.all_processing_quadrants || status.quadrants || []).map(
        ([x, y]) => `${x},${y}`
      )
    );

    // Detect quadrants that just finished processing (were processing, now not)
    const completedQuadrants = [];
    lastProcessingQuadrants.forEach((coordKey) => {
      if (!currentProcessingQuadrants.has(coordKey)) {
        completedQuadrants.push(coordKey);
      }
    });

    // Refresh tiles for completed quadrants
    if (completedQuadrants.length > 0) {
      console.log("Refreshing completed quadrants:", completedQuadrants);
      completedQuadrants.forEach((coordKey) => {
        const [qx, qy] = coordKey.split(",").map(Number);
        refreshTileImage(qx, qy);
      });
    }

    // Update tracking for next poll
    lastProcessingQuadrants = currentProcessingQuadrants;

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
        // No more items - tiles already refreshed, just stop polling
        stopStatusPolling();
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
    case "waterselect":
      if (document.getElementById("waterSelectTool")) {
        toggleWaterSelectTool();
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
      // Don't restore if tile is locked (actively processing)
      // Queued tiles CAN be selected
      if (!tile.classList.contains("locked")) {
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

// =============================================================================
// NYC Outline Feature
// =============================================================================

let nycBoundaryData = null;
let nycOutlineVisible = false;

// Toggle NYC outline visibility
function toggleNycOutline() {
  const checkbox = document.getElementById("showNycOutline");
  nycOutlineVisible = checkbox?.checked || false;

  // Save preference to localStorage
  try {
    localStorage.setItem("viewer_show_nyc_outline", nycOutlineVisible ? "1" : "0");
  } catch (e) {
    console.warn("Could not save NYC outline preference:", e);
  }

  if (nycOutlineVisible) {
    if (nycBoundaryData) {
      renderNycOutline();
    } else {
      fetchNycBoundary();
    }
  } else {
    clearNycOutline();
  }
}

// Initialize NYC outline state from localStorage
function initNycOutline() {
  try {
    const saved = localStorage.getItem("viewer_show_nyc_outline");
    if (saved === "1") {
      const checkbox = document.getElementById("showNycOutline");
      if (checkbox) {
        checkbox.checked = true;
        nycOutlineVisible = true;
        fetchNycBoundary();
      }
    }
  } catch (e) {
    // Ignore localStorage errors
  }
}

// Fetch NYC boundary data from API
async function fetchNycBoundary() {
  try {
    const response = await fetch("/api/nyc-boundary");
    const data = await response.json();
    nycBoundaryData = data;
    console.log("Fetched NYC boundary with", data.boundary.features.length, "features");
    if (nycOutlineVisible) {
      renderNycOutline();
    }
  } catch (error) {
    console.error("Failed to fetch NYC boundary:", error);
  }
}

// Convert quadrant coordinates to pixel position on the grid
function quadrantToPixel(qx, qy) {
  const gridX = config.x;
  const gridY = config.y;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Calculate pixel position relative to the grid
  const col = qx - gridX;
  const row = qy - gridY;

  const px = col * (sizePx + gap);
  const py = row * (sizePx + gap);

  return { x: px, y: py };
}

// Render the NYC outline as SVG paths
function renderNycOutline() {
  const svg = document.getElementById("nycOutlineOverlay");
  if (!svg || !nycBoundaryData) return;

  // Clear existing paths
  svg.innerHTML = "";

  const nx = config.nx;
  const ny = config.ny;
  const sizePx = config.size_px;
  const showLines = document.getElementById("showLines")?.checked || false;
  const gap = showLines ? 2 : 0;

  // Calculate SVG dimensions to match the grid
  const svgWidth = nx * sizePx + (nx - 1) * gap;
  const svgHeight = ny * sizePx + (ny - 1) * gap;

  svg.setAttribute("width", svgWidth);
  svg.setAttribute("height", svgHeight);
  svg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);

  // Borough colors for visual distinction
  const boroughColors = {
    "Manhattan": "#ff6b6b",
    "Brooklyn": "#4ecdc4",
    "Queens": "#45b7d1",
    "Bronx": "#96ceb4",
    "Staten Island": "#ffeaa7"
  };

  // Render each borough - always render all paths, SVG will clip naturally
  nycBoundaryData.boundary.features.forEach((feature) => {
    const name = feature.properties.name;
    const color = boroughColors[name] || "#3b82f6";

    // Process each ring of the polygon
    feature.geometry.coordinates.forEach((ring, ringIndex) => {
      // Build SVG path data
      let pathData = "";

      ring.forEach((coord, i) => {
        const [qx, qy] = coord;
        const pixel = quadrantToPixel(qx, qy);

        const cmd = i === 0 ? "M" : "L";
        pathData += `${cmd}${pixel.x.toFixed(1)},${pixel.y.toFixed(1)}`;
      });

      // Close the path
      pathData += "Z";

      // Always render the path - SVG overflow handles clipping
      if (pathData.length > 2) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathData);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", "2");
        path.setAttribute("stroke-opacity", "0.8");
        path.setAttribute("data-borough", name);
        svg.appendChild(path);
      }
    });
  });

  console.log("Rendered NYC outline");
}

// Clear the NYC outline
function clearNycOutline() {
  const svg = document.getElementById("nycOutlineOverlay");
  if (svg) {
    svg.innerHTML = "";
  }
}

// Re-render NYC outline when grid settings change (lines toggle)
function updateNycOutlineOnSettingsChange() {
  if (nycOutlineVisible && nycBoundaryData) {
    renderNycOutline();
  }
}

// Override toggleLines to also update NYC outline
const originalToggleLines = toggleLines;
toggleLines = function() {
  originalToggleLines();
  updateNycOutlineOnSettingsChange();
};

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Initialize water highlight toggle
  initWaterHighlight();

  // Initialize NYC outline toggle
  initNycOutline();

  // Initialize labels toggle
  initLabels();

  // Initialize saved prompt indicator
  updatePromptButtonIndicator();

  // Initialize saved negative prompt indicator
  updateNegativePromptButtonIndicator();

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
