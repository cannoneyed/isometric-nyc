// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  } else {
    document.body.classList.remove("generating");
  }

  // Apply locked/queued style to matching tiles, and REMOVE from non-matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    } else {
      // Neither locked nor queued - remove both classes
      tile.classList.remove("locked");
      tile.classList.remove("queued");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
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
  if (e.target.tagName === "INPUT") return;

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
    case "g":
    case "G":
      document.getElementById("showRender").click();
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

  showToast("loading", "Filling with water...", `Processing quadrant (${qx}, ${qy})`);

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

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
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
  if (locked.length > 0 || queued.length > 0) {
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

// Generation/Render state
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

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
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

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

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

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
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
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
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

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      // Actually merge: keep local items that aren't currently being generated
      const localQueued = getQueuedQuadrants();
      const locked = getLockedQuadrants();
      // Keep local items not in server queue and not currently locked (generating)
      const localOnly = localQueued.filter(
        ([lx, ly]) =>
          !serverQueued.some(([sx, sy]) => sx === lx && sy === ly) &&
          !locked.some(([gx, gy]) => gx === lx && gy === ly)
      );
      // Merge: server queue + local-only items (that aren't generating)
      const merged = [...serverQueued, ...localOnly];
      setQueuedQuadrants(merged);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();
