# Export command button functionality
_Exported on 12/18/2025 at 14:19:35 EST from Cursor (2.2.20)_

---

**User**

I need you to add a new button to @src/isometric_nyc/e2e_generation/app.py / @src/isometric_nyc/e2e_generation/static/viewer.js that's called "export cmd" - if 2 quadrants are selected it's available, and it copies a string to the clipboard of the following shape:

uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl='82,9' --br='83,12'  --overwrite

with the given top left and bottom right given the currently selected quadrants. It should copy this to the clipboard and alert the user with a toast

---

**Cursor**

I'll need to look at the template file to understand how the buttons are structured, then add the new button and its handler.

I can see the structure. I need to add the "Export Cmd" button to the template and add the corresponding JavaScript function. Let me make both changes:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Generated Tiles Viewer</title>
    <link rel="stylesheet" href="/static/viewer.css" />
    <style>
      /* Dynamic grid sizing based on template variables */
      .grid {
        grid-template-columns: repeat({{ nx }}, {{ size_px }}px);
        grid-auto-rows: {{ size_px }}px;
      }
      .tile.placeholder {
        min-width: {{ size_px }}px;
        min-height: {{ size_px }}px;
      }
    </style>
  </head>
  <body>
    <!-- Config data for JS -->
    <script
      id="app-config"
      type="application/json"
      data-config='{"x": {{ x }}, "y": {{ y }}, "nx": {{ nx }}, "ny": {{ ny }}, "size_px": {{ size_px }}, "models": {{ models_config | safe }}, "default_model_id": "{{ default_model_id }}"}'
    ></script>

    <!-- Toast notification container -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- Fixed toolbar -->
    <div class="toolbar">
      <div class="toolbar-row toolbar-row-nav">
        <div class="controls">
          <label>X: <input type="number" id="x" value="{{ x }}" /></label>
          <label>Y: <input type="number" id="y" value="{{ y }}" /></label>
          <label
            >NX: <input type="number" id="nx" value="{{ nx }}" min="1" max="20"
          /></label>
          <label
            >NY: <input type="number" id="ny" value="{{ ny }}" min="1" max="20"
          /></label>
          <label
            >Size:
            <input type="number" id="sizePx" value="{{ size_px }}" step="32"
          /></label>
          <button onclick="goTo()">Go</button>
          <div class="toggle-group">
            <label
              ><input
                type="checkbox"
                id="showLines"
                {%
                if
                show_lines
                %}checked{%
                endif
                %}
                onchange="toggleLines()"
              />
              Lines</label
            >
            <label
              ><input
                type="checkbox"
                id="showCoords"
                {%
                if
                show_coords
                %}checked{%
                endif
                %}
                onchange="toggleCoords()"
              />
              Coords</label
            >
            <label
              ><input
                type="checkbox"
                id="showRender"
                {%
                if
                show_render
                %}checked{%
                endif
                %}
                onchange="toggleRender()"
              />
              Renders</label
            >
          </div>
          <div class="toggle-group model-group">
            <label class="model-label"
              >Model:
              <select id="modelSelect" class="model-select">
                <!-- Populated by JavaScript -->
              </select>
            </label>
          </div>
        </div>
        <div class="toolbar-info">
          <span>({{ x }}, {{ y }}) → ({{ x + nx - 1 }}, {{ y + ny - 1 }})</span>
          <span id="selectedQuadrantsDisplay" class="selected-quadrants"></span>
          <span class="generation-dir-info">{{ generation_dir }}</span>
        </div>
      </div>
      <div class="toolbar-row toolbar-row-tools">
        <div class="tools-group">
          <span class="tools-label">Tools:</span>
          <button
            id="selectTool"
            class="tool-btn"
            onclick="toggleSelectTool()"
            title="Select quadrants (S)"
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
              <path d="M13 13l6 6"></path>
            </svg>
            Select
          </button>
          <button
            id="fixWaterTool"
            class="tool-btn"
            onclick="toggleFixWaterTool()"
            title="Fix water color (W)"
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
            </svg>
            Fix Water
          </button>
          <button
            id="waterFillTool"
            class="tool-btn"
            onclick="toggleWaterFillTool()"
            title="Fill quadrant with water (F)"
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <rect x="3" y="3" width="18" height="18" rx="2"></rect>
              <path d="M12 8v8"></path>
              <path d="M8 12h8"></path>
            </svg>
            Water Fill
          </button>
          <button
            id="waterSelectTool"
            class="tool-btn"
            onclick="toggleWaterSelectTool()"
            title="Mark quadrants as water tiles (T)"
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
              <path d="M12 12v4"></path>
              <circle cx="12" cy="18" r="1"></circle>
            </svg>
            Water Select
          </button>
        </div>
        <div class="toolbar-divider"></div>
        <div class="actions-group">
          <button
            id="deselectAllBtn"
            class="action-btn deselect-btn"
            onclick="deselectAll()"
            disabled
          >
            Deselect
          </button>
          <button
            id="deleteBtn"
            class="action-btn delete-btn"
            onclick="deleteSelected()"
            disabled
            title="Delete data (generation or render based on current view)"
          >
            Delete
          </button>
          <button
            id="flagBtn"
            class="action-btn flag-btn"
            onclick="flagSelected()"
            disabled
            title="Flag selected quadrants for review"
          >
            🚩 Flag
          </button>
          <button
            id="renderBtn"
            class="action-btn render-btn"
            onclick="renderSelected()"
            disabled
          >
            Render
          </button>
          <button
            id="generateBtn"
            class="action-btn generate-btn"
            onclick="generateSelected()"
            disabled
          >
            Generate
          </button>
          <button
            id="generateWithPromptBtn"
            class="action-btn generate-prompt-btn"
            onclick="showPromptDialog()"
            disabled
            title="Generate with additional prompt text"
          >
            + Prompt
          </button>
          <button
            id="generateRectBtn"
            class="action-btn generate-rect-btn"
            onclick="generateRectangle()"
            disabled
            title="Select exactly 2 quadrants to define rectangle corners"
          >
            Gen Rect
          </button>
          <button
            id="exportCmdBtn"
            class="action-btn export-cmd-btn"
            onclick="copyExportCommand()"
            disabled
            title="Copy export command to clipboard (select 2 quadrants)"
          >
            Export Cmd
          </button>
        </div>
        <div class="toolbar-divider"></div>
        <div class="utility-group">
          <button
            id="clearQueueBtn"
            class="utility-btn clear-queue-btn"
            onclick="clearQueue()"
            title="Clear all pending items from the generation queue"
          >
            Clear Queue
          </button>
          <button
            id="hardRefreshBtn"
            class="utility-btn hard-refresh-btn"
            onclick="hardRefresh()"
            title="Clear image cache and reload page"
          >
            ↻ Refresh
          </button>
        </div>
      </div>
      <div class="toolbar-row toolbar-row-status" id="selectionStatus">
        <span id="selectionCount"></span>
      </div>
      <div class="water-fix-status" id="waterFixStatus" style="display: none">
        <span class="water-fix-label">Target Color:</span>
        <span class="color-swatch" id="targetColorSwatch"></span>
        <span class="color-hex" id="targetColorHex"
          >Click a quadrant to pick color</span
        >
        <span class="water-fix-quadrant" id="waterFixQuadrant"></span>
        <button
          id="applyWaterFixBtn"
          class="apply-water-fix-btn"
          onclick="applyWaterFix()"
          disabled
        >
          Apply Fix
        </button>
        <button
          id="cancelWaterFixBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFix()"
        >
          Cancel
        </button>
      </div>
      <div class="water-fill-status" id="waterFillStatus" style="display: none">
        <span class="water-fill-label">Water Fill:</span>
        <span class="color-swatch water-color-preview"></span>
        <span class="water-fill-instruction" id="waterFillInstruction"
          >Click a quadrant to fill with water</span
        >
        <button
          id="cancelWaterFillBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFill()"
        >
          Cancel
        </button>
      </div>
      <div
        class="water-select-status"
        id="waterSelectStatus"
        style="display: none"
      >
        <span class="water-select-label">💧 Water Select:</span>
        <span class="water-select-instruction" id="waterSelectInstruction"
          >Click quadrants to toggle water status</span
        >
        <button
          id="cancelWaterSelectBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterSelect()"
        >
          Cancel
        </button>
      </div>
    </div>

    <div
      class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}"
      id="gridContainer"
    >
      <div class="grid">
        {% for dy in range(ny) %} {% for dx in range(nx) %} {% set qx = x + dx
        %} {% set qy = y + dy %} {% set has_gen = tiles.get((dx, dy), False) %}
        {% set is_flagged = flagged_tiles.get((dx, dy), False) %} {% set
        is_water = water_tiles.get((dx, dy), False) %}
        <div
          class="tile {% if not has_gen %}placeholder{% endif %}{% if is_flagged %} flagged{% endif %}{% if is_water %} water{% endif %}"
          data-coords="{{ qx }},{{ qy }}"
          data-flagged="{{ 'true' if is_flagged else 'false' }}"
          data-water="{{ 'true' if is_water else 'false' }}"
        >
          <span class="coords">({{ qx }}, {{ qy }})</span>
          {% if has_gen %}
          <img
            src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}"
            alt="Tile {{ qx }},{{ qy }}"
          />
          {% endif %} {% if is_water %}
          <span class="water-indicator" title="Water tile">💧</span>
          {% endif %}
        </div>
        {% endfor %} {% endfor %}
      </div>
    </div>

    <!-- Prompt Dialog -->
    <div id="promptDialog" class="dialog-overlay" style="display: none">
      <div class="dialog-content">
        <h3>Generate with Prompt</h3>
        <p class="dialog-description">
          Add additional instructions for the generation:
        </p>
        <textarea
          id="promptInput"
          class="prompt-input"
          placeholder="e.g., 'Add more trees' or 'Make the water darker'"
          rows="3"
        ></textarea>
        <div class="dialog-buttons">
          <button class="dialog-cancel-btn" onclick="hidePromptDialog()">
            Cancel
          </button>
          <button class="dialog-submit-btn" onclick="submitPromptGeneration()">
            Generate
          </button>
        </div>
      </div>
    </div>

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

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

  // Check current water status
  const isCurrentlyWater = tileEl.dataset.water === "true";
  const newWaterStatus = !isCurrentlyWater;

  const instruction = document.getElementById("waterSelectInstruction");
  instruction.textContent = `Updating (${qx}, ${qy})...`;

  try {
    const response = await fetch("/api/water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quadrants: [[qx, qy]],
        is_water: newWaterStatus,
      }),
    });

    const result = await response.json();

    if (result.success) {
      // Update the tile's visual state
      if (newWaterStatus) {
        tileEl.classList.add("water");
        tileEl.dataset.water = "true";
        // Add water indicator if it doesn't exist
        if (!tileEl.querySelector(".water-indicator")) {
          const indicator = document.createElement("span");
          indicator.className = "water-indicator";
          indicator.title = "Water tile";
          indicator.textContent = "💧";
          tileEl.appendChild(indicator);
        }
        showToast("success", "Marked as water", `Quadrant (${qx}, ${qy})`);
      } else {
        tileEl.classList.remove("water");
        tileEl.dataset.water = "false";
        // Remove water indicator
        const indicator = tileEl.querySelector(".water-indicator");
        if (indicator) {
          indicator.remove();
        }
        showToast("info", "Unmarked as water", `Quadrant (${qx}, ${qy})`);
      }

      instruction.textContent = "Click quadrants to toggle water status";
    } else {
      showToast("error", "Failed to update", result.error || "Unknown error");
      instruction.textContent = "Click quadrants to toggle water status";
    }
  } catch (error) {
    console.error("Water select error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click quadrants to toggle water status";
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
  // Generate Rectangle requires exactly 2 selected
  if (generateRectBtn) generateRectBtn.disabled = count !== 2;
  // Export Cmd requires exactly 2 selected
  const exportCmdBtn = document.getElementById("exportCmdBtn");
  if (exportCmdBtn) exportCmdBtn.disabled = count !== 2;
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

  // Check if we're in render view mode
  const isRenderMode = document.getElementById("showRender")?.checked || false;
  const dataType = isRenderMode ? "render" : "generation";
  const apiEndpoint = isRenderMode ? "/api/delete-render" : "/api/delete";

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete ${dataType} data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch(apiEndpoint, {
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

  // Build context info for toast
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";

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

  const showRender = document.getElementById("showRender")?.checked ? "1" : "0";
  const timestamp = Date.now();
  const imgUrl = `/tile/${qx}/${qy}?render=${showRender}&_t=${timestamp}`;

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

  // Check current water status
  const isCurrentlyWater = tileEl.dataset.water === "true";
  const newWaterStatus = !isCurrentlyWater;

  const instruction = document.getElementById("waterSelectInstruction");
  instruction.textContent = `Updating (${qx}, ${qy})...`;

  try {
    const response = await fetch("/api/water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quadrants: [[qx, qy]],
        is_water: newWaterStatus,
      }),
    });

    const result = await response.json();

    if (result.success) {
      // Update the tile's visual state
      if (newWaterStatus) {
        tileEl.classList.add("water");
        tileEl.dataset.water = "true";
        // Add water indicator if it doesn't exist
        if (!tileEl.querySelector(".water-indicator")) {
          const indicator = document.createElement("span");
          indicator.className = "water-indicator";
          indicator.title = "Water tile";
          indicator.textContent = "💧";
          tileEl.appendChild(indicator);
        }
        showToast("success", "Marked as water", `Quadrant (${qx}, ${qy})`);
      } else {
        tileEl.classList.remove("water");
        tileEl.dataset.water = "false";
        // Remove water indicator
        const indicator = tileEl.querySelector(".water-indicator");
        if (indicator) {
          indicator.remove();
        }
        showToast("info", "Unmarked as water", `Quadrant (${qx}, ${qy})`);
      }

      instruction.textContent = "Click quadrants to toggle water status";
    } else {
      showToast("error", "Failed to update", result.error || "Unknown error");
      instruction.textContent = "Click quadrants to toggle water status";
    }
  } catch (error) {
    console.error("Water select error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click quadrants to toggle water status";
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
  // Generate Rectangle requires exactly 2 selected
  if (generateRectBtn) generateRectBtn.disabled = count !== 2;
  // Export Cmd requires exactly 2 selected
  const exportCmdBtn = document.getElementById("exportCmdBtn");
  if (exportCmdBtn) exportCmdBtn.disabled = count !== 2;
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

  // Check if we're in render view mode
  const isRenderMode = document.getElementById("showRender")?.checked || false;
  const dataType = isRenderMode ? "render" : "generation";
  const apiEndpoint = isRenderMode ? "/api/delete-render" : "/api/delete";

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete ${dataType} data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch(apiEndpoint, {
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

  // Build context info for toast
  const contextMsg =
    contextQuadrants.length > 0
      ? ` (using ${contextQuadrants.length} as context)`
      : "";
  const promptMsg = prompt ? " with prompt" : "";

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
  const command = `uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl='${minX},${minY}' --br='${maxX},${maxY}' --overwrite`;

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

  const showRender = document.getElementById("showRender")?.checked ? "1" : "0";
  const timestamp = Date.now();
  const imgUrl = `/tile/${qx}/${qy}?render=${showRender}&_t=${timestamp}`;

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

Let me check for any lint errors:

I've added the "Export Cmd" button. Here's a summary of the changes:

1. **Template (`viewer.html`)**: Added a new button after "Gen Rect" that calls `copyExportCommand()`

2. **JavaScript (`viewer.js`)**: 
   - Added the `copyExportCommand()` function that:
     - Validates exactly 2 quadrants are selected
     - Calculates min/max coordinates to determine top-left and bottom-right
     - Builds the command string with the format you specified
     - Copies to clipboard using `navigator.clipboard.writeText()`
     - Shows a success toast with the coordinates, or an error toast if copy fails
   - Updated `updateSelectionStatus()` to enable/disable the button (requires exactly 2 selected quadrants, same as "Gen Rect")

The button will be disabled until exactly 2 quadrants are selected. When clicked, it copies a command like:

```
uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl='82,9' --br='83,12' --overwrite
```

```plaintext
uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl='82,9' --br='83,12' --overwrite
```

