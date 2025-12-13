# Prompt dialog key handler issue
_Exported on 12/13/2025 at 10:46:48 EST from Cursor (2.1.39)_

---

**User**

Ensure in @src/isometric_nyc/e2e_generation/static/viewer.js that whenever the dialog to add a prompt is open, the key handlers for the keyboard shortcuts don't trigger

---

**Cursor**

```javascript
if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
```

```javascript
// Also ignore if prompt dialog is open
const promptDialog = document.getElementById("promptDialog");
if (promptDialog && promptDialog.style.display !== "none") return;
```

Looking at the code, I can see that the keyboard event handler at line 139-189 checks if the target is an INPUT or SELECT element, but it doesn't account for when the prompt dialog is open. I need to add a check to prevent keyboard shortcuts from triggering when the dialog is visible.

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Add options for each model
  config.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    if (model.model_id === config.default_model_id) {
      option.selected = true;
    }
    select.appendChild(option);
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    if (
      serverStatus.is_generating &&
      serverStatus.quadrants &&
      serverStatus.quadrants.length > 0
    ) {
      const action =
        serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsStr = serverStatus.quadrants
        .map(([x, y]) => `(${x},${y})`)
        .join(" ");
      statusParts.push(`${action} ${coordsStr}`);
    }

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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

    // Track state changes
    const wasGenerating = isGenerating || isRendering;
    isGenerating = status.is_generating && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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

Done! I added a check at line 143-144 that prevents keyboard shortcuts from triggering when the prompt dialog is visible. The handler now returns early if:
1. The focus is on an INPUT or SELECT element (existing check), OR
2. The prompt dialog is open (`display !== "none"`)

This ensures that when users are typing in the prompt dialog, pressing keys like "G", "S", "W", etc. won't accidentally trigger their associated actions (generate, select tool, fix water tool, etc.).

---

**User**

OK - also add functionality so that when I change the model id from the select, it automatically unfocuses the select (so I can hit the arrow keys)

---

**Cursor**

I'll add a change event listener to the model select that automatically blurs it after selection:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Add options for each model
  config.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    if (model.model_id === config.default_model_id) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  // Auto-blur after selection so arrow keys work for navigation
  select.addEventListener("change", () => {
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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

    // Track state changes
    const wasGenerating = isGenerating || isRendering;
    isGenerating = status.is_generating && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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

Done! I added a `change` event listener to the model select that calls `blur()` on it. Now when you select a different model from the dropdown, it will automatically unfocus, allowing you to immediately use the arrow keys for navigation.

---

**User**

OK - finally, ensure that only one of "Select", "Fix Water", and "Water Fill" can be selected at one time

---

**Cursor**

Looking at the code, I can see:
- `toggleWaterFillTool()` already deactivates both other tools ✓
- `toggleSelectTool()` only deactivates `fixWaterTool`, missing `waterFillTool`
- `toggleFixWaterTool()` only deactivates `selectTool`, missing `waterFillTool`

Let me fix both:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Add options for each model
  config.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    if (model.model_id === config.default_model_id) {
      option.selected = true;
    }
    select.appendChild(option);
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Add options for each model
  config.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    if (model.model_id === config.default_model_id) {
      option.selected = true;
    }
    select.appendChild(option);
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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

Done! Now all three tools properly deactivate each other:

- **`toggleSelectTool()`** - now deactivates both Fix Water and Water Fill
- **`toggleFixWaterTool()`** - now deactivates both Select and Water Fill
- **`toggleWaterFillTool()`** - already deactivated both Select and Fix Water ✓

This ensures only one of Select, Fix Water, or Water Fill can be active at any time.

---

**User**

Ok great, ensure that the last selected tool (select, fill water, etc) AND the last selected model id are preserved between page reloads - it should default to no tool if the saved tool doesn't apply, and default to the first select model id if the id isn't present

---

**Cursor**

I'll add localStorage persistence for both the selected tool and model ID. Let me first look at the initialization code to understand where to integrate this:

Now I'll add the localStorage persistence. I'll:
1. Add save/restore functions for tool and model
2. Update `initModelSelector()` to restore and save model ID
3. Update toggle functions to save tool state
4. Restore tool on page load

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";

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
  const savedModelExists = savedModelId && config.models.some(m => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    
    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (!savedModelExists && model.model_id === config.default_model_id) {
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
```

I see the file was modified. Let me apply the changes in parts:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";

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

// Initialize model selector
function initModelSelector() {
  const select = document.getElementById("modelSelect");
  if (!select || !config.models || config.models.length === 0) {
    return;
  }

  // Clear existing options
  select.innerHTML = "";

  // Add options for each model
  config.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    if (model.model_id === config.default_model_id) {
      option.selected = true;
    }
    select.appendChild(option);
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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
  const savedModelExists = savedModelId && config.models.some(m => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    
    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (!savedModelExists && model.model_id === config.default_model_id) {
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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

Now let me update the toggle functions to save tool state. First, let me read the current toggle functions:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";

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
  const savedModelExists = savedModelId && config.models.some(m => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    
    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (!savedModelExists && model.model_id === config.default_model_id) {
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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
  const savedModelExists = savedModelId && config.models.some(m => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    
    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (!savedModelExists && model.model_id === config.default_model_id) {
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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
  const savedModelExists = savedModelId && config.models.some(m => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    
    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (!savedModelExists && model.model_id === config.default_model_id) {
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

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

Now let me add the tool restoration logic to the initialize function:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// LocalStorage keys for persistence
const STORAGE_KEY_MODEL = "viewer_selected_model";
const STORAGE_KEY_TOOL = "viewer_selected_tool";

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
  const savedModelExists = savedModelId && config.models.some(m => m.model_id === savedModelId);

  // Add options for each model
  config.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = model.name;
    
    // Select saved model if it exists, otherwise use default or first
    if (savedModelExists && model.model_id === savedModelId) {
      option.selected = true;
    } else if (!savedModelExists && model.model_id === config.default_model_id) {
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
  const processingQuadrants = status.all_processing_quadrants || status.quadrants || [];
  const isProcessing = status.is_generating || (status.active_model_count > 0);

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
    const processingQuadrants = serverStatus.all_processing_quadrants || serverStatus.quadrants || [];

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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
  const generateWithPromptBtn = document.getElementById("generateWithPromptBtn");
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
      console.log("Generation queued at position:", result.position, "model:", result.model_id);
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
    isGenerating = (status.is_generating || hasActiveModels) && status.status !== "rendering";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

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

Done! I've added localStorage persistence for both the selected tool and model ID. Here's what was implemented:

**Model ID persistence:**
- Saved to localStorage when the select changes
- On page load, restores the saved model if it exists in the available models
- Falls back to the default model or first model if the saved ID isn't found

**Tool persistence:**
- Each toggle function saves the tool name when activated ("select", "fixwater", "waterfill") or clears it when deactivated
- On page load, `restoreSavedTool()` checks if the saved tool's button exists before activating it
- Unknown or invalid saved tools are cleared and default to no tool

The localStorage keys used are:
- `viewer_selected_model` - for the model ID
- `viewer_selected_tool` - for the tool name

---

**User**

Also ensure that the selected quadrants are stored between page reload (arrow navigation) - they should also be reported in the top toolbard

---

**Cursor**

I'll add persistence for selected quadrants and display them in the toolbar. Let me first check the current selection and toolbar code:

Now I understand the structure. I'll add persistence for selected quadrants and update the toolbar to show the coordinates. Let me make the changes:

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

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

Now let me update `updateSelectionStatus` to show the actual coordinates:

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count with coordinates
  if (count > 0) {
    const coordsStr = Array.from(selectedQuadrants)
      .map((key) => {
        const [x, y] = key.split(",");
        return `(${x},${y})`;
      })
      .join(" ");
    statusParts.push(`✓ ${coordsStr}`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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

// Initialize on page load
(async function initialize() {
  // Initialize model selector
  initModelSelector();

  // Restore saved tool
  restoreSavedTool();

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

Now let me add the restore function and call it during initialization:

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count with coordinates
  if (count > 0) {
    const coordsStr = Array.from(selectedQuadrants)
      .map((key) => {
        const [x, y] = key.split(",");
        return `(${x},${y})`;
      })
      .join(" ");
    statusParts.push(`✓ ${coordsStr}`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
      if (!tile.classList.contains("locked") && !tile.classList.contains("queued")) {
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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count with coordinates
  if (count > 0) {
    const coordsStr = Array.from(selectedQuadrants)
      .map((key) => {
        const [x, y] = key.split(",");
        return `(${x},${y})`;
      })
      .join(" ");
    statusParts.push(`✓ ${coordsStr}`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
      if (!tile.classList.contains("locked") && !tile.classList.contains("queued")) {
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

I should also update the places where selections are cleared during generation. Let me check those:

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count with coordinates
  if (count > 0) {
    const coordsStr = Array.from(selectedQuadrants)
      .map((key) => {
        const [x, y] = key.split(",");
        return `(${x},${y})`;
      })
      .join(" ");
    statusParts.push(`✓ ${coordsStr}`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
      if (!tile.classList.contains("locked") && !tile.classList.contains("queued")) {
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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count with coordinates
  if (count > 0) {
    const coordsStr = Array.from(selectedQuadrants)
      .map((key) => {
        const [x, y] = key.split(",");
        return `(${x},${y})`;
      })
      .join(" ");
    statusParts.push(`✓ ${coordsStr}`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
      if (!tile.classList.contains("locked") && !tile.classList.contains("queued")) {
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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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
  }

  // Show selection count with coordinates
  if (count > 0) {
    const coordsStr = Array.from(selectedQuadrants)
      .map((key) => {
        const [x, y] = key.split(",");
        return `(${x},${y})`;
      })
      .join(" ");
    statusParts.push(`✓ ${coordsStr}`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
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
      if (!tile.classList.contains("locked") && !tile.classList.contains("queued")) {
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

Done! I've implemented persistence for selected quadrants and updated the toolbar display. Here's what was added:

**Selection persistence:**
- Selected quadrants are saved to localStorage (`viewer_selected_quadrants`)
- Saved after every selection change, deselection, and when cleared after generation/render
- Restored on page load - only quadrants that exist on the current view are restored
- Locked/queued tiles are skipped during restoration
- Storage is cleaned up if saved quadrants are no longer on the current page

**Toolbar display:**
- The status bar now shows the actual coordinates like `✓ (0,1) (2,3)` instead of just "2 selected"
- This makes it clear which quadrants are selected even when navigating

---

**User**

The selected quadrants are being overwritten by the generation status - can you show that state in the bar above the generation stauts (next to the coordinates + generation dir?)

---

**Cursor**

Let me look at the HTML structure to understand the layout:

I understand - I'll add a dedicated span for selected quadrants in the toolbar-info section (next to coordinates and generation dir) and update the JS to populate it there instead.

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
      <div class="toolbar-row">
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
          <div class="toggle-group tools-group">
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
          </div>
        </div>
        <div class="toolbar-info">
          <span>({{ x }}, {{ y }}) → ({{ x + nx - 1 }}, {{ y + ny - 1 }})</span>
          <span id="selectedQuadrantsDisplay" class="selected-quadrants"></span>
          <span>{{ generation_dir }}</span>
        </div>
      </div>
      <div class="selection-status" id="selectionStatus">
        <span id="selectionCount">0 quadrants selected</span>
        <span class="selection-limit">(max 4)</span>
        <button
          id="deselectAllBtn"
          class="deselect-btn"
          onclick="deselectAll()"
          disabled
        >
          Deselect All
        </button>
        <button
          id="deleteBtn"
          class="delete-btn"
          onclick="deleteSelected()"
          disabled
        >
          Delete
        </button>
        <button
          id="renderBtn"
          class="render-btn"
          onclick="renderSelected()"
          disabled
        >
          Render
        </button>
        <button
          id="generateBtn"
          class="generate-btn"
          onclick="generateSelected()"
          disabled
        >
          Generate
        </button>
        <button
          id="generateWithPromptBtn"
          class="generate-prompt-btn"
          onclick="showPromptDialog()"
          disabled
          title="Generate with additional prompt text"
        >
          + Prompt
        </button>
        <button
          id="generateRectBtn"
          class="generate-rect-btn"
          onclick="generateRectangle()"
          disabled
          title="Select exactly 2 quadrants to define rectangle corners"
        >
          Generate Rectangle
        </button>
        <button
          id="clearQueueBtn"
          class="clear-queue-btn"
          onclick="clearQueue()"
          title="Clear all pending items from the generation queue"
        >
          Clear Queue
        </button>
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
    </div>

    <div
      class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}"
      id="gridContainer"
    >
      <div class="grid">
        {% for dy in range(ny) %} {% for dx in range(nx) %} {% set qx = x + dx
        %} {% set qy = y + dy %} {% set has_gen = tiles.get((dx, dy), False) %}
        <div
          class="tile {% if not has_gen %}placeholder{% endif %}"
          data-coords="{{ qx }},{{ qy }}"
        >
          <span class="coords">({{ qx }}, {{ qy }})</span>
          {% if has_gen %}
          <img
            src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}"
            alt="Tile {{ qx }},{{ qy }}"
          />
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

Now let me update the JS to use this new element and remove the selection from the status bar:

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

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(
            `.tile[data-coords="${qx},${qy}"]`
          );
          if (tile && !tile.classList.contains("locked")) {
            tile.classList.add("queued");
          }
        });
      }
    });
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

Now let me add some CSS styling for this new element:

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

```

Done! I've moved the selected quadrants display to the toolbar-info section:

**Changes made:**

1. **HTML** (`viewer.html`): Added a new span with id `selectedQuadrantsDisplay` in the toolbar-info section, between the coordinate range and generation_dir

2. **JavaScript** (`viewer.js`): Updated `updateSelectionStatus()` to:
   - Populate the new toolbar-info element with selected coordinates (e.g., `✓ (0,1) (2,3)`)
   - Hide the element when nothing is selected
   - The status bar below now just shows the count ("X selected") without coordinates

3. **CSS** (`viewer.css`): Added styling for `.selected-quadrants` to display in cyan (#00d9ff) and bold to make it stand out

Now the selected quadrants will always be visible in the top toolbar row, and won't be overwritten by generation status updates in the bar below.

