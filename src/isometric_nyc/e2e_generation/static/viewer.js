// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";

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

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  if (locked.length === 0) return;

  // Add generating class to body
  document.body.classList.add("generating");

  // Apply locked style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    if (isLocked) {
      tile.classList.add("locked");
    }
  });
}

function removeLockedStyles() {
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    tile.classList.remove("locked");
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
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      break;
    case "s":
    case "S":
      if (!isGenerating) toggleSelectTool();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
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

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating
  const locked = getLockedQuadrants();
  if (locked.length > 0 && isGenerating) {
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `generating ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isGenerating;
  deleteBtn.disabled = count === 0 || isGenerating;
  generateBtn.disabled = count === 0 || isGenerating;
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

// Generation state
let isGenerating = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;
  if (isGenerating) {
    showToast(
      "info",
      "Generation in progress",
      "Cannot delete while generating."
    );
    return;
  }

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
  if (isGenerating) {
    showToast(
      "info",
      "Generation in progress",
      "Please wait for the current generation to complete."
    );
    return;
  }

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Set loading state and lock the quadrants
  isGenerating = true;
  setLockedQuadrants(coords);
  document.body.classList.add("generating");

  // Mark selected tiles as locked (purple)
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
    tile.classList.add("locked");
  });
  selectedQuadrants.clear();
  updateSelectionStatus();

  const generateBtn = document.getElementById("generateBtn");
  generateBtn.disabled = true;
  generateBtn.classList.add("loading");
  generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

  showToast(
    "loading",
    "Generating tiles...",
    `Processing ${coords.length} quadrant${
      coords.length > 1 ? "s" : ""
    }. This may take a minute.`
  );

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // If request was accepted, start polling for status
    // The server will process in background and we poll for updates
    if (response.status === 429) {
      // Already generating - start polling
      showToast(
        "info",
        "Generation in progress",
        "Reconnected to existing generation."
      );
      startStatusPolling();
      return;
    }

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

      // Clear selection and refresh after a short delay
      deselectAll();
      setTimeout(() => {
        window.location.reload();
      }, 1500);
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
  if (isGenerating) return; // Can't select while generating

  const key = `${qx},${qy}`;
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

    if (status.is_generating) {
      // Update UI to show generation in progress
      setGeneratingUI(status);
    } else {
      // Generation finished
      stopStatusPolling();

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", "Generation complete!", status.message);
        setTimeout(() => window.location.reload(), 1500);
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", "Generation failed", status.error);
        resetGenerateButton();
      }
    }
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setGeneratingUI(status) {
  const generateBtn = document.getElementById("generateBtn");
  if (!generateBtn.classList.contains("loading")) {
    generateBtn.disabled = true;
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;

    // Apply locked styles to tiles
    applyLockedStyles();

    // Update selection status to show generating message
    updateSelectionStatus();

    // Show toast if not already showing
    if (document.querySelectorAll(".toast.loading").length === 0) {
      showToast(
        "loading",
        "Generation in progress...",
        status.message || "Please wait..."
      );
    }
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

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    if (status.is_generating) {
      console.log("Generation in progress, restoring UI state...");
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setGeneratingUI(status);
      startStatusPolling();
    } else {
      // Not generating - clear any stale locked state
      clearLockedQuadrants();
    }
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();
