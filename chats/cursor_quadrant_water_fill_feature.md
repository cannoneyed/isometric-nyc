# Quadrant water fill feature
_Exported on 1/7/2026 at 09:47:13 EST from Cursor (2.2.44)_

---

**User**

Add a new feature to @src/isometric_nyc/e2e_generation/app.py  (and @src/isometric_nyc/e2e_generation/static/viewer.js ) that lets me select the top left and bottom right quadrants, and if two are selected, enables a button that will fill all quadrants defined in that rectangle with the water color (4A6372) - ensure we have similar confirmation dialogs as the delete button

---

**Cursor**

Looking at this request, I need to:
1. Add a new API endpoint in `app.py` to fill a rectangle of quadrants with water color
2. Add a button and handler in `viewer.js` that's enabled when exactly 2 quadrants are selected
3. Also update the HTML template to add the button

Let me first look at the template to understand the button structure:

Now I have a clear picture of the structure. I'll add:
1. A new button "Fill Rect Water" in the template (near the other rectangle operations)
2. A new API endpoint in `app.py`
3. A JavaScript handler in `viewer.js`

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
            <label
              ><input
                type="checkbox"
                id="showWater"
                onchange="toggleWaterHighlight()"
              />
              💧 Water</label
            >
            <label
              ><input
                type="checkbox"
                id="showNycOutline"
                onchange="toggleNycOutline()"
              />
              🗽 NYC</label
            >
            <label
              ><input
                type="checkbox"
                id="showLabels"
                checked
                onchange="toggleLabels()"
              />
              🏷️ Labels</label
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
            id="starBtn"
            class="action-btn star-btn"
            onclick="starSelected()"
            disabled
            title="Star quadrant for dataset curation (select exactly 1)"
          >
            ⭐ Star
          </button>
          <button
            id="prevStarredBtn"
            class="action-btn nav-starred-btn"
            onclick="goToPrevStarred()"
            title="Go to previous starred quadrant ([)"
          >
            ◀
          </button>
          <button
            id="viewStarredBtn"
            class="action-btn view-starred-btn"
            onclick="showStarredDialog()"
            title="View all starred entries"
          >
            📋 Starred
          </button>
          <button
            id="nextStarredBtn"
            class="action-btn nav-starred-btn"
            onclick="goToNextStarred()"
            title="Go to next starred quadrant (])"
          >
            ▶
          </button>
          <button
            id="referenceBtn"
            class="action-btn reference-btn"
            onclick="referenceSelected()"
            disabled
            title="Mark/unmark reference tile (top-left of 2x2) [B]"
          >
            🍌 Reference
          </button>
          <button
            id="clearReferencesBtn"
            class="action-btn clear-references-btn"
            onclick="clearAllReferences()"
            title="Clear all reference tiles"
          >
            🗑️ Clear Refs
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
            id="generateWithNegativePromptBtn"
            class="action-btn generate-prompt-btn"
            onclick="showNegativePromptDialog()"
            disabled
            title="Generate with negative prompt"
          >
            - Neg Prompt
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
            id="fillRectWaterBtn"
            class="action-btn fill-rect-water-btn"
            onclick="fillRectangleWater()"
            disabled
            title="Fill rectangle with water color (select 2 quadrants)"
          >
            💧 Fill Rect
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
          <button
            id="exportBtn"
            class="action-btn export-btn"
            onclick="exportSelected()"
            disabled
            title="Export rectangle as PNG (select 2 quadrants)"
          >
            Export
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
          >Click to cycle: unset → water 💧 → protected 🛡️ → unset</span
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
      <!-- NYC Outline SVG overlay -->
      <svg class="nyc-outline-overlay" id="nycOutlineOverlay"></svg>
      <div class="grid">
        {% for dy in range(ny) %} {% for dx in range(nx) %} {% set qx = x + dx
        %} {% set qy = y + dy %} {% set has_gen = tiles.get((dx, dy), False) %}
        {% set is_flagged = flagged_tiles.get((dx, dy), False) %} {% set
        is_starred = starred_tiles.get((dx, dy), False) %} {% set
        is_water = water_tiles.get((dx, dy), False) %} {% set
        is_explicit_not_water = explicit_not_water_tiles.get((dx, dy), False) %} {% set
        is_reference = reference_tiles.get((dx, dy), False) %}
        <div
          class="tile {% if not has_gen %}placeholder{% endif %}{% if is_flagged %} flagged{% endif %}{% if is_starred %} starred{% endif %}{% if is_water %} water{% endif %}{% if is_explicit_not_water %} explicit-not-water{% endif %}{% if is_reference %} is-reference{% endif %}"
          data-coords="{{ qx }},{{ qy }}"
          data-flagged="{{ 'true' if is_flagged else 'false' }}"
          data-starred="{{ 'true' if is_starred else 'false' }}"
          data-water="{{ 'true' if is_water else 'false' }}"
          data-explicit-not-water="{{ 'true' if is_explicit_not_water else 'false' }}"
          data-is-reference="{{ 'true' if is_reference else 'false' }}"
        >
          <span class="coords">({{ qx }}, {{ qy }})</span>
          {% if has_gen %}
          <img
            src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}"
            alt="Tile {{ qx }},{{ qy }}"
          />
          {% endif %} {% if is_starred %}
          <span class="starred-indicator" title="Starred for dataset">⭐</span>
          {% endif %} {% if is_water %}
          <span class="water-indicator" title="Water tile">💧</span>
          {% endif %} {% if is_explicit_not_water %}
          <span class="explicit-not-water-indicator" title="Explicitly NOT water (protected)">🚫</span>
          {% endif %} {% if is_reference %}
          <span class="reference-indicator" title="Reference tile (top-left of 2x2)">🍌</span>
          {% endif %}
        </div>
        {% endfor %} {% endfor %}
      </div>
    </div>

    <!-- Prompt Dialog -->
    <div id="promptDialog" class="dialog-overlay" style="display: none">
      <div class="dialog-content">
        <h3>Generate with Custom Prompt</h3>
        <p class="dialog-description">
          Provide a custom prompt to completely override the default generation prompt.
          <br><span class="dialog-hint">This prompt will be saved and applied to all future generations.</span>
        </p>
        <div id="savedPromptDisplay" class="saved-prompt-display" style="display: none"></div>
        <textarea
          id="promptInput"
          class="prompt-input"
          placeholder="e.g., 'Add more trees' or 'Make the water darker'"
          rows="3"
        ></textarea>
        <div class="dialog-buttons">
          <button id="clearPromptBtn" class="dialog-clear-btn" onclick="clearSavedPrompt(); hidePromptDialog();" style="display: none">
            Clear Saved
          </button>
          <div class="dialog-buttons-right">
            <button class="dialog-cancel-btn" onclick="hidePromptDialog()">
              Cancel
            </button>
            <button class="dialog-submit-btn" onclick="submitPromptGeneration()">
              Generate
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Negative Prompt Dialog -->
    <div id="negativePromptDialog" class="dialog-overlay" style="display: none">
      <div class="dialog-content">
        <h3>Generate with Negative Prompt</h3>
        <p class="dialog-description">
          Provide a negative prompt to specify what you don't want in the generation.
          <br><span class="dialog-hint">This negative prompt will be saved and applied to all future generations.</span>
        </p>
        <div id="savedNegativePromptDisplay" class="saved-prompt-display" style="display: none"></div>
        <textarea
          id="negativePromptInput"
          class="prompt-input"
          placeholder="e.g., 'no buildings' or 'no water'"
          rows="3"
        ></textarea>
        <div class="dialog-buttons">
          <button id="clearNegativePromptBtn" class="dialog-clear-btn" onclick="clearSavedNegativePrompt(); hideNegativePromptDialog();" style="display: none">
            Clear Saved
          </button>
          <div class="dialog-buttons-right">
            <button class="dialog-cancel-btn" onclick="hideNegativePromptDialog()">
              Cancel
            </button>
            <button class="dialog-submit-btn" onclick="submitNegativePromptGeneration()">
              Generate
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Starred Entries Dialog -->
    <div id="starredDialog" class="dialog-overlay" style="display: none">
      <div class="dialog-content starred-dialog-content">
        <h3>⭐ Starred Entries</h3>
        <p class="dialog-description">
          Click an entry to navigate to that location.
        </p>
        <div id="starredListContainer" class="starred-list-container">
          <div id="starredList" class="starred-list">
            <!-- Populated by JavaScript -->
          </div>
        </div>
        <div id="starredEmptyState" class="starred-empty-state" style="display: none">
          No starred entries yet. Select a quadrant and click ⭐ Star to add one.
        </div>
        <div class="dialog-buttons">
          <div class="starred-count" id="starredCountDisplay">0 starred</div>
          <div class="dialog-buttons-right">
            <button class="dialog-cancel-btn" onclick="hideStarredDialog()">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

```python
"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Command-line flags:
  --no-generate - Disable generation processing (queue items are preserved
                  but not processed until the flag is removed)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
  G          - Generate selected quadrants
  S          - Toggle select tool
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.make_rectangle_plan import (
  Point,
  RectBounds,
  create_rectangle_plan,
  get_plan_summary,
  validate_plan,
)
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cancel_processing_items,
  cancel_queue_item_by_id,
  clear_completed_items,
  clear_pending_queue,
  get_all_processing_items,
  get_next_pending_item_for_available_model,
  get_pending_queue,
  get_queue_position_for_model,
  get_queue_status,
  get_queue_status_by_model,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
  reset_all_processing_items,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
  latlng_to_quadrant_coords,
)
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Global boundary GeoJSON - loaded at startup
BOUNDARY_GEOJSON: dict | None = None

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)


# =============================================================================
# Logging Configuration - Suppress noisy tile request logs
# =============================================================================
class TileRequestFilter(logging.Filter):
  """Filter out noisy tile and static file requests from logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    message = record.getMessage()
    # Filter out tile requests, static files, and api/status polling
    if "/tile/" in message:
      return False
    if "/static/" in message:
      return False
    if "/api/status" in message:
      return False
    return True


# Apply filter to werkzeug logger (Flask's HTTP request logger)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(TileRequestFilter())

# Generation lock - protects generation_state updates
generation_lock = threading.Lock()

# Per-model generation states
# Key is model_id (None for default), value is state dict
model_generation_states: dict[str | None, dict] = {}

# Legacy global generation_state for backwards compatibility with API
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
  "current_item_id": None,
  "model_id": None,
}

# Track which models are currently processing
busy_models: set[str | None] = set()
busy_models_lock = threading.Lock()

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Cancellation flag - set to True to cancel all generations
generation_cancelled = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None
NO_GENERATE_MODE: bool = False


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def ensure_flagged_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the flagged column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "flagged" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'flagged' column to quadrants table")


def ensure_is_water_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


def ensure_starred_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the starred column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "starred" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN starred INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'starred' column to quadrants table")


def ensure_is_reference_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_reference column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_reference" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_reference INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_reference' column to quadrants table")


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


def get_quadrant_info(x: int, y: int, use_render: bool = False) -> dict:
  """
  Get info about a quadrant including whether it has data, is flagged, starred, and water status.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed)
     1: Water tile
  """
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)
    ensure_starred_column_exists(conn)
    ensure_is_reference_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0),
             COALESCE(starred, 0), COALESCE(is_reference, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      water_status = row[2]
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": water_status == 1,  # True if water
        "is_explicit_not_water": water_status == -1,  # True if explicitly not water
        "water_status": water_status,  # Raw value: -1, 0, or 1
        "starred": bool(row[3]),
        "is_reference": bool(row[4]),
      }
    return {
      "has_data": False,
      "flagged": False,
      "is_water": False,
      "is_explicit_not_water": False,
      "water_status": 0,
      "starred": False,
      "is_reference": False,
    }
  finally:
    conn.close()


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data, flagged status, starred status, and water status
  tiles = {}
  flagged_tiles = {}
  starred_tiles = {}
  water_tiles = {}
  explicit_not_water_tiles = {}
  reference_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      starred_tiles[(dx, dy)] = info["starred"]
      water_tiles[(dx, dy)] = info["is_water"]
      explicit_not_water_tiles[(dx, dy)] = info["is_explicit_not_water"]
      reference_tiles[(dx, dy)] = info["is_reference"]

  # Get model configuration for the frontend
  models_config = []
  default_model_id = None
  if APP_CONFIG:
    models_config = [m.to_dict() for m in APP_CONFIG.models]
    default_model_id = APP_CONFIG.default_model_id

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    flagged_tiles=flagged_tiles,
    starred_tiles=starred_tiles,
    water_tiles=water_tiles,
    explicit_not_water_tiles=explicit_not_water_tiles,
    reference_tiles=reference_tiles,
    generation_dir=str(GENERATION_DIR),
    models_config=json.dumps(models_config),
    default_model_id=default_model_id,
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = shared_get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


def run_nano_banana_generation_wrapper(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
  negative_prompt: str | None = None,
  model_config: "ModelConfig | None" = None,  # noqa: F821
) -> dict:
  """Wrapper to call nano banana generation with references from DB."""
  from isometric_nyc.e2e_generation.generate_tile_nano_banana import (
    run_nano_banana_generation,
  )

  # Load all marked references from database
  ensure_is_reference_column_exists(conn)
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE is_reference = 1"
  )
  reference_coords = [(row[0], row[1]) for row in cursor.fetchall()]

  if not reference_coords:
    return {
      "success": False,
      "error": "No reference tiles marked. Please mark at least one 2x2 reference tile.",
    }

  print(f"   🍌 Loaded {len(reference_coords)} reference tile(s)")

  # Call nano banana generation
  try:
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=selected_quadrants,
      reference_coords=reference_coords,
      port=DEFAULT_WEB_PORT,
      prompt=prompt,
      negative_prompt=negative_prompt,
      save=True,
      status_callback=None,
      context_quadrants=context_quadrants,
      generation_dir=GENERATION_DIR,
      model_config=model_config,
    )

    return result
  except Exception as e:
    import traceback

    traceback.print_exc()
    return {"success": False, "error": f"Nano banana generation failed: {str(e)}"}


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
  negative_prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation.
    prompt: Optional additional prompt text for generation
    negative_prompt: Optional negative prompt text for generation

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

  # Route to nano banana if model_type is nano_banana
  if model_config and model_config.model_type == "nano_banana":
    return run_nano_banana_generation_wrapper(
      conn, config, selected_quadrants, context_quadrants, prompt, negative_prompt, model_config
    )

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants,
    prompt=prompt,
    negative_prompt=negative_prompt,
  )


def render_quadrant_with_renderer(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
) -> bytes | None:
  """
  Render a quadrant using the global web renderer.

  Returns the PNG bytes of the rendered quadrant.
  """
  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    save_quadrant_render,
  )
  from isometric_nyc.e2e_generation.web_renderer import get_web_renderer

  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  renderer = get_web_renderer(port=WEB_SERVER_PORT)

  # Render the tile
  quadrant_images = renderer.render_quadrant(
    quadrant_x=x,
    quadrant_y=y,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    camera_azimuth_degrees=config["camera_azimuth_degrees"],
    camera_elevation_degrees=config["camera_elevation_degrees"],
    view_height_meters=config.get("view_height_meters", 200),
  )

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), png_bytes in quadrant_images.items():
    qx, qy = x + dx, y + dy
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


def process_queue_item_from_db(item_id: int) -> dict:
  """Process a single queue item from the database."""
  global generation_state

  conn = get_db_connection()
  try:
    # Get item details
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT item_type, quadrants, model_id, context_quadrants, prompt, negative_prompt
      FROM generation_queue
      WHERE id = ?
      """,
      (item_id,),
    )
    row = cursor.fetchone()
    if not row:
      return {"success": False, "error": "Item not found"}

    item_type = QueueItemType(row[0])
    quadrants = json.loads(row[1])
    model_id = row[2]
    context_quadrants_raw = json.loads(row[3]) if row[3] else None
    prompt = row[4]
    negative_prompt = row[5]

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Calculate context lazily if not explicitly provided
    # This ensures we use the most up-to-date context based on what's
    # actually generated at execution time (not queue time)
    if context_quadrants_raw:
      context_quadrants = [(q[0], q[1]) for q in context_quadrants_raw]
      print(
        f"   📋 Using explicit context from queue: {len(context_quadrants)} quadrant(s)"
      )
    else:
      # Calculate context lazily based on current generation state
      context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
      if context_quadrants:
        print(f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s)")
      else:
        print(
          "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
        )

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = (
      "starting" if item_type == QueueItemType.GENERATE else "rendering"
    )
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(
      f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}"
    )
    if model_id:
      print(f"   Model: {model_id}")
    if context_quadrants:
      print(f"   Context: {context_quadrants}")
    if prompt:
      print(f"   Prompt: {prompt}")
    if negative_prompt:
      print(f"   Negative Prompt: {negative_prompt}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt, negative_prompt
        )

        if result["success"]:
          print(f"✅ Generation complete: {result['message']}")
          generation_state["status"] = "complete"
          generation_state["message"] = result["message"]
          mark_item_complete(conn, item_id, result["message"])
          return result

        # Generation failed
        if gen_attempt < max_generation_retries:
          print(
            f"⚠️  Generation failed (attempt {gen_attempt}/{max_generation_retries}): "
            f"{result['error']}"
          )
          print(f"⏳ Waiting {generation_retry_delay}s before retrying generation...")
          update_generation_state(
            "retrying",
            f"Generation failed, retrying (attempt {gen_attempt + 1}/{max_generation_retries})...",
          )
          time.sleep(generation_retry_delay)
        else:
          # All retries exhausted
          print(
            f"❌ Generation failed after {max_generation_retries} attempts: "
            f"{result['error']}"
          )
          generation_state["status"] = "error"
          generation_state["error"] = result["error"]
          mark_item_error(conn, item_id, result["error"])
          return result

      # Should not reach here, but just in case
      return result

    else:  # render
      update_generation_state("rendering", "Starting render...")

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant_with_renderer(conn, config, qx, qy)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      result_message = f"Rendered {rendered_count} quadrant(s)"
      update_generation_state("complete", result_message)
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")
      mark_item_complete(conn, item_id, result_message)

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    mark_item_error(conn, item_id, str(e))
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def process_model_item(item_id: int, model_id: str | None):
  """Process a single queue item for a specific model in its own thread."""
  global generation_state, generation_cancelled

  try:
    # Check cancellation before starting
    if generation_cancelled:
      print(f"⚠️  Item {item_id} cancelled before processing")
      return

    process_queue_item_from_db(item_id)

  except Exception as e:
    print(f"❌ Model worker error for {model_id}: {e}")
    traceback.print_exc()
  finally:
    # Remove model from busy set
    with busy_models_lock:
      busy_models.discard(model_id)

    # Update global state if this was the active model
    with generation_lock:
      if generation_state.get("model_id") == model_id:
        generation_state["is_generating"] = False
        generation_state["current_item_id"] = None

    # Remove from per-model states
    if model_id in model_generation_states:
      del model_generation_states[model_id]


def queue_worker():
  """Background worker that processes the generation queue from the database.

  This worker supports parallel processing of different models - each model
  can have one active generation at a time, but different models can run
  concurrently.

  If NO_GENERATE_MODE is enabled, the worker will not process any items but
  will keep them preserved in the queue.
  """
  global generation_state, queue_worker_running, generation_cancelled

  if NO_GENERATE_MODE:
    print(
      "🔄 Queue worker started (NO-GENERATE MODE - queue preserved but not processed)"
    )
  else:
    print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
      # If no-generate mode is enabled, just sleep and don't process anything
      if NO_GENERATE_MODE:
        time.sleep(1.0)
        continue

      # Check if we were cancelled
      if generation_cancelled:
        print("⚠️  Generation cancelled, resetting flags...")
        generation_cancelled = False
        with generation_lock:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None
        with busy_models_lock:
          busy_models.clear()
        model_generation_states.clear()
        time.sleep(0.5)
        continue

      conn = get_db_connection()

      # Get current busy models
      with busy_models_lock:
        current_busy = busy_models.copy()

      # Get next pending item for an available model
      item = get_next_pending_item_for_available_model(conn, current_busy)

      if item is None:
        # No items available (either queue empty or all models busy)
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      model_id = item.model_id
      conn.close()
      conn = None

      # Mark this model as busy
      with busy_models_lock:
        if model_id in busy_models:
          # Another thread grabbed this model, skip
          continue
        busy_models.add(model_id)

      # Update global state for display (use most recent)
      with generation_lock:
        generation_state["is_generating"] = True
        generation_state["model_id"] = model_id

      # Initialize per-model state
      model_generation_states[model_id] = {
        "is_generating": True,
        "item_id": item_id,
        "started_at": time.time(),
      }

      # Spawn a thread to process this model's item
      model_name = model_id or "default"
      worker_thread = threading.Thread(
        target=process_model_item,
        args=(item_id, model_id),
        name=f"model-worker-{model_name}",
        daemon=True,
      )
      worker_thread.start()

      print(f"🚀 Started worker for model '{model_name}' (item {item_id})")

      # Small delay before checking for more work
      time.sleep(0.2)

    except Exception as e:
      print(f"❌ Queue worker error: {e}")
      traceback.print_exc()
      time.sleep(1.0)
    finally:
      if conn:
        conn.close()

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue_db(
  quadrants: list[tuple[int, int]],
  item_type: str,
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
  negative_prompt: str | None = None,
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
      context_quadrants,
      prompt,
      negative_prompt,
    )

    # Get model-specific queue position
    model_position = get_queue_position_for_model(conn, queue_item.id, model_id)

    # Get total queue length for backwards compatibility
    pending = get_pending_queue(conn)
    total_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": model_position,  # Position within this model's queue
      "total_position": total_position,  # Overall queue position
      "model_id": model_id,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {model_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)
    model_status = get_queue_status_by_model(conn)

    # Get list of currently busy models
    with busy_models_lock:
      active_models = list(busy_models)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
      # Per-model queue info
      "queue_by_model": model_status["by_model"],
      "processing_models": model_status["processing_models"],
      # All currently processing models (for parallel processing)
      "active_models": active_models,
      "active_model_count": len(active_models),
      # All quadrants being processed across all models
      "all_processing_quadrants": model_status["all_processing_quadrants"],
    }

    # Set is_generating based on whether any models are active
    response["is_generating"] = len(active_models) > 0

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
  """
  API endpoint to clear all items from the generation queue,
  including cancelling any currently running generation.

  Returns:
    {
      "success": true,
      "cleared_count": N,
      "cancelled_count": M,
      "message": "Cleared N pending item(s), cancelled M in-progress item(s)"
    }
  """
  global generation_state, generation_cancelled

  print(f"\n{'=' * 60}")
  print("🗑️  Clear queue request received")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # Clear pending items
    cleared_count = clear_pending_queue(conn)

    # Cancel any in-progress items
    cancelled_count = cancel_processing_items(conn)

    # Set the cancellation flag so the worker knows to stop
    if cancelled_count > 0:
      generation_cancelled = True

    # Clear busy models
    with busy_models_lock:
      busy_models.clear()

    # Clear per-model states
    model_generation_states.clear()

    # Reset the generation state
    generation_state["is_generating"] = False
    generation_state["quadrants"] = []
    generation_state["status"] = "idle"
    generation_state["message"] = "Queue cleared"
    generation_state["error"] = None
    generation_state["current_item_id"] = None

    if cleared_count > 0 or cancelled_count > 0:
      print(
        f"✅ Cleared {cleared_count} pending, cancelled {cancelled_count} in-progress"
      )
    else:
      print("ℹ️  Queue was already empty")

    message_parts = []
    if cleared_count > 0:
      message_parts.append(f"Cleared {cleared_count} pending item(s)")
    if cancelled_count > 0:
      message_parts.append(f"cancelled {cancelled_count} in-progress item(s)")
    message = ", ".join(message_parts) if message_parts else "Queue was already empty"

    return jsonify(
      {
        "success": True,
        "cleared_count": cleared_count,
        "cancelled_count": cancelled_count,
        "message": message,
      }
    )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/queue/cancel/<int:item_id>", methods=["POST"])
def api_cancel_queue_item(item_id: int):
  """
  API endpoint to cancel a specific queue item by its ID.

  Returns:
    {
      "success": true,
      "cancelled": true,
      "item_id": N,
      "message": "Cancelled queue item N"
    }
  """
  global generation_cancelled

  print(f"\n{'=' * 60}")
  print(f"🗑️  Cancel queue item request received: item_id={item_id}")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # First check if this item was processing (not just pending)
    cursor = conn.cursor()
    cursor.execute(
      "SELECT status, model_id FROM generation_queue WHERE id = ?",
      (item_id,),
    )
    row = cursor.fetchone()
    was_processing = row and row[0] == "processing"
    cancelled_model_id = row[1] if row else None

    cancelled = cancel_queue_item_by_id(conn, item_id)

    if cancelled:
      print(f"✅ Cancelled queue item {item_id}")

      # Only set the global cancellation flag if this was a PROCESSING item
      # Pending items just get marked as cancelled in the database
      if was_processing:
        generation_cancelled = True
        print("   ⚠️  Item was processing, signaling cancellation")

        # Also remove this model from busy set so it can pick up new work
        if cancelled_model_id:
          with busy_models_lock:
            busy_models.discard(cancelled_model_id)

      return jsonify(
        {
          "success": True,
          "cancelled": True,
          "item_id": item_id,
          "message": f"Cancelled queue item {item_id}",
        }
      )
    else:
      print(f"ℹ️  Queue item {item_id} not found or already completed")
      return jsonify(
        {
          "success": True,
          "cancelled": False,
          "item_id": item_id,
          "message": f"Queue item {item_id} not found or already completed",
        }
      )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify(
    {
      "models": [m.to_dict() for m in APP_CONFIG.models],
      "default_model_id": APP_CONFIG.default_model_id,
    }
  )


@app.route("/api/nyc-boundary")
def api_nyc_boundary():
  """
  API endpoint to get the NYC boundary GeoJSON with coordinate transformation info.

  Returns the NYC borough boundaries along with the generation config needed
  to transform lat/lng coordinates to quadrant (x, y) coordinates.
  """
  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    # Pre-compute boundary points in quadrant coordinates for the frontend
    # This avoids complex math in JavaScript
    boundary_in_quadrants = {"type": "FeatureCollection", "features": []}

    for feature in BOUNDARY_GEOJSON["features"]:
      new_feature = {
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": {"type": feature["geometry"]["type"], "coordinates": []},
      }

      # Process each ring of the polygon
      for ring in feature["geometry"]["coordinates"]:
        new_ring = []
        for coord in ring:
          lng, lat = coord[0], coord[1]
          qx, qy = latlng_to_quadrant_coords(config, lat, lng)
          new_ring.append([qx, qy])
        new_feature["geometry"]["coordinates"].append(new_ring)

      boundary_in_quadrants["features"].append(new_feature)

    return jsonify(
      {
        "boundary": boundary_in_quadrants,
        "seed": config["seed"],
      }
    )
  finally:
    conn.close()


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) and also clear flagged status
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL, flagged = 0
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/delete-render", methods=["POST"])
def api_delete_render():
  """API endpoint to delete render data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the render column (set to NULL) and also clear flagged status
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL, flagged = 0
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted render data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/flag", methods=["POST"])
def api_flag():
  """API endpoint to flag/unflag selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get flag value (default to True/1 for flagging, False/0 for unflagging)
  flag_value = 1 if data.get("flag", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the flagged column exists
    ensure_flagged_column_exists(conn)

    flagged_count = 0
    for qx, qy in quadrants:
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET flagged = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (flag_value, qx, qy),
      )
      if cursor.rowcount > 0:
        flagged_count += 1

    conn.commit()

    action = "Flagged" if flag_value else "Unflagged"
    return jsonify(
      {
        "success": True,
        "message": f"{action} {flagged_count} quadrant{'s' if flagged_count != 1 else ''}",
        "count": flagged_count,
        "flagged": bool(flag_value),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/star", methods=["POST"])
def api_star():
  """
  API endpoint to star/unstar a single quadrant.

  Note: Only one quadrant can be starred at a time.
  """
  data = request.get_json()
  if not data or "quadrant" not in data:
    return jsonify({"success": False, "error": "No quadrant specified"})

  quadrant = data["quadrant"]
  if not isinstance(quadrant, list) or len(quadrant) != 2:
    return jsonify({"success": False, "error": "Quadrant must be [x, y]"})

  qx, qy = int(quadrant[0]), int(quadrant[1])
  star_value = 1 if data.get("star", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the starred column exists
    ensure_starred_column_exists(conn)

    cursor = conn.execute(
      """
      UPDATE quadrants
      SET starred = ?
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (star_value, qx, qy),
    )

    if cursor.rowcount > 0:
      conn.commit()
      action = "Starred" if star_value else "Unstarred"
      return jsonify(
        {
          "success": True,
          "message": f"{action} quadrant ({qx}, {qy})",
          "starred": bool(star_value),
          "quadrant": [qx, qy],
        }
      )
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Quadrant ({qx}, {qy}) not found in database",
        }
      )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/starred")
def api_starred():
  """
  API endpoint to get all starred quadrants.

  Returns a list of starred quadrant coordinates with their info.
  """
  conn = get_db_connection()

  try:
    # Ensure the starred column exists
    ensure_starred_column_exists(conn)

    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT quadrant_x, quadrant_y, generation IS NOT NULL, render IS NOT NULL
      FROM quadrants
      WHERE starred = 1
      ORDER BY quadrant_y, quadrant_x
      """
    )

    starred = []
    for row in cursor.fetchall():
      starred.append(
        {
          "x": row[0],
          "y": row[1],
          "has_generation": bool(row[2]),
          "has_render": bool(row[3]),
        }
      )

    return jsonify(
      {
        "success": True,
        "starred": starred,
        "count": len(starred),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/reference", methods=["POST"])
def api_reference():
  """
  API endpoint to mark/unmark a single quadrant as reference (top-left of 2x2 tile).

  Reference tiles are used for nano banana generation to provide style context.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  data = request.get_json()
  if not data or "quadrant" not in data:
    return jsonify({"success": False, "error": "No quadrant specified"})

  quadrant = data["quadrant"]
  if not isinstance(quadrant, list) or len(quadrant) != 2:
    return jsonify({"success": False, "error": "Quadrant must be [x, y]"})

  qx, qy = int(quadrant[0]), int(quadrant[1])
  reference_value = 1 if data.get("reference", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the is_reference column exists
    ensure_is_reference_column_exists(conn)

    # Validate that this is a valid 2x2 tile (all 4 quadrants have generations)
    if reference_value == 1:
      # Check all 4 quadrants exist with generations
      for dx in [0, 1]:
        for dy in [0, 1]:
          gen = shared_get_quadrant_generation(conn, qx + dx, qy + dy)
          if gen is None:
            return jsonify(
              {
                "success": False,
                "error": f"Incomplete 2x2 tile at ({qx + dx}, {qy + dy}). All 4 quadrants must have generations.",
              }
            )

    cursor = conn.execute(
      """
      UPDATE quadrants
      SET is_reference = ?
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (reference_value, qx, qy),
    )

    if cursor.rowcount > 0:
      conn.commit()
      action = "Marked as reference" if reference_value else "Unmarked as reference"
      return jsonify(
        {
          "success": True,
          "message": f"{action}: ({qx}, {qy})",
          "is_reference": bool(reference_value),
          "quadrant": [qx, qy],
        }
      )
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Quadrant ({qx}, {qy}) not found in database",
        }
      )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/references")
def api_references():
  """
  API endpoint to get all reference quadrants.

  Returns a list of reference quadrant coordinates (top-left of 2x2 tiles).
  """
  conn = get_db_connection()

  try:
    # Ensure the is_reference column exists
    ensure_is_reference_column_exists(conn)

    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT quadrant_x, quadrant_y
      FROM quadrants
      WHERE is_reference = 1
      ORDER BY quadrant_y, quadrant_x
      """
    )

    references = [[row[0], row[1]] for row in cursor.fetchall()]

    return jsonify(
      {
        "success": True,
        "references": references,
        "count": len(references),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/references/clear", methods=["POST"])
def api_clear_references():
  """
  API endpoint to clear all reference quadrants.

  Removes reference status from all quadrants.
  """
  conn = get_db_connection()

  try:
    # Ensure the is_reference column exists
    ensure_is_reference_column_exists(conn)

    cursor = conn.execute("UPDATE quadrants SET is_reference = 0 WHERE is_reference = 1")

    cleared_count = cursor.rowcount
    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Cleared {cleared_count} reference(s)",
        "count": cleared_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/water", methods=["POST"])
def api_water():
  """
  API endpoint to mark/unmark selected quadrants as water tiles.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed by script)
     1: Water tile

  Request body:
    {
      "quadrants": [[x, y], ...],
      "is_water": true/false,  // true=water(1), false=not water(0)
      "explicit_not_water": true  // Optional: if true, sets to -1 (protected)
    }
  """
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Determine water value:
  # - explicit_not_water=true → -1 (protected from auto-detection)
  # - is_water=true → 1 (water)
  # - is_water=false → 0 (not water, can be auto-changed)
  if data.get("explicit_not_water", False):
    water_value = -1
    action = "Marked as explicitly NOT water (protected)"
  elif data.get("is_water", True):
    water_value = 1
    action = "Marked as water"
  else:
    water_value = 0
    action = "Unmarked as water"

  conn = get_db_connection()

  try:
    # Ensure the is_water column exists
    ensure_is_water_column_exists(conn)

    water_count = 0
    for qx, qy in quadrants:
      # First ensure the quadrant exists in the database
      cursor = conn.execute(
        "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (qx, qy),
      )
      if cursor.fetchone() is None:
        # Quadrant doesn't exist, skip it
        continue

      cursor = conn.execute(
        """
        UPDATE quadrants
        SET is_water = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (water_value, qx, qy),
      )
      if cursor.rowcount > 0:
        water_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "water_status": water_value,
        "is_water": water_value == 1,
        "is_explicit_not_water": water_value == -1,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill-rectangle", methods=["POST"])
def api_water_fill_rectangle():
  """
  API endpoint to fill a rectangular region of quadrants with the water color.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y}   // Bottom-right corner
    }

  Returns:
    {
      "success": true,
      "message": "Filled N quadrants with water color",
      "filled_count": N,
      "color": "#4A6372"
    }
  """
  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    image_to_png_bytes,
    save_quadrant_generation,
  )

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl_x, tl_y = int(tl_raw[0]), int(tl_raw[1])
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl_x, tl_y = int(tl_raw["x"]), int(tl_raw["y"])
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br_x, br_y = int(br_raw[0]), int(br_raw[1])
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br_x, br_y = int(br_raw["x"]), int(br_raw["y"])
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl_x > br_x or tl_y > br_y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1
  total_quadrants = width_count * height_count

  print(f"\n{'=' * 60}")
  print(
    f"💧 Water fill rectangle request: ({tl_x},{tl_y}) to ({br_x},{br_y}) "
    f"({width_count}x{height_count} = {total_quadrants} quadrants)"
  )
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Determine quadrant size from config (quadrant is half the tile size)
    width = config.get("width_px", 512) // 2
    height = config.get("height_px", 512) // 2

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
    fill_color = (*water_rgb, 255)  # Add full alpha for RGBA

    # Create the solid water color image once (reuse for all quadrants)
    water_img = Image.new("RGBA", (width, height), fill_color)
    png_bytes = image_to_png_bytes(water_img)

    filled_count = 0
    for dy in range(height_count):
      for dx in range(width_count):
        qx, qy = tl_x + dx, tl_y + dy
        save_quadrant_generation(conn, config, qx, qy, png_bytes)
        filled_count += 1
        print(f"   ✓ Filled quadrant ({qx}, {qy})")

    print(f"✅ Water fill rectangle complete: {filled_count} quadrants")
    return jsonify(
      {
        "success": True,
        "message": f"Filled {filled_count} quadrant(s) with water color",
        "filled_count": filled_count,
        "color": WATER_REPLACEMENT_COLOR,
        "bounds": {
          "tl": [tl_x, tl_y],
          "br": [br_x, br_y],
          "width": width_count,
          "height": height_count,
        },
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  print(f"\n{'=' * 60}")
  print(f"🎨 Render request: {selected_quadrants}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "render")
  return jsonify(result), 202  # 202 Accepted


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """
  API endpoint to generate tiles for selected quadrants.

  Request body:
    {
      "quadrants": [[x, y], ...] or [{"x": x, "y": y}, ...],
      "model_id": "optional-model-id",
      "context": [[x, y], ...] or [{"x": x, "y": y}, ...]  // Optional context quadrants
    }

  The context quadrants are used to provide surrounding pixel art context for
  the generation. If a context quadrant has an existing generation, that will
  be used; otherwise the render content will be used.
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Get optional model_id from request
  model_id = data.get("model_id")

  # Convert quadrants to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Parse optional context quadrants
  context_quadrants = None
  context_raw = data.get("context")
  if context_raw:
    if not isinstance(context_raw, list):
      return jsonify(
        {
          "success": False,
          "error": "context must be a list of quadrant coordinates",
        }
      ), 400

    context_quadrants = []
    for q in context_raw:
      if isinstance(q, list) and len(q) == 2:
        context_quadrants.append((int(q[0]), int(q[1])))
      elif isinstance(q, dict) and "x" in q and "y" in q:
        context_quadrants.append((int(q["x"]), int(q["y"])))
      else:
        return jsonify(
          {
            "success": False,
            "error": f"Invalid context quadrant format: {q}",
          }
        ), 400

  # Parse optional prompt
  prompt = data.get("prompt")
  if prompt and not isinstance(prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "prompt must be a string",
      }
    ), 400

  # Clean up prompt (strip whitespace, None if empty)
  if prompt:
    prompt = prompt.strip()
    if not prompt:
      prompt = None

  # Parse optional negative_prompt
  negative_prompt = data.get("negative_prompt")
  if negative_prompt and not isinstance(negative_prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "negative_prompt must be a string",
      }
    ), 400

  # Clean up negative_prompt (strip whitespace, None if empty)
  if negative_prompt:
    negative_prompt = negative_prompt.strip()
    if not negative_prompt:
      negative_prompt = None

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  if negative_prompt:
    print(f"   Negative Prompt: {negative_prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt, negative_prompt
  )
  return jsonify(result), 202  # 202 Accepted


# =============================================================================
# Rectangle Generation API
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def load_queued_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """
  Load all quadrants from pending and processing queue items.

  These quadrants are scheduled for generation and should be considered
  when planning new rectangles to avoid seam issues.
  """
  queued: set[Point] = set()

  # Get pending items
  pending_items = get_pending_queue(conn)
  for item in pending_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  # Get processing items
  processing_items = get_all_processing_items(conn)
  for item in processing_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  return queued


@app.route("/api/export", methods=["POST"])
def api_export():
  """
  API endpoint to export a rectangular region of quadrants as a single PNG image.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "use_render": false                 // Optional: export render instead of generation
    }

  Returns:
    PNG image as attachment download
  """
  import io

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    png_bytes_to_image,
  )

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl_x, tl_y = int(tl_raw[0]), int(tl_raw[1])
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl_x, tl_y = int(tl_raw["x"]), int(tl_raw["y"])
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br_x, br_y = int(br_raw[0]), int(br_raw[1])
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br_x, br_y = int(br_raw["x"]), int(br_raw["y"])
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl_x > br_x or tl_y > br_y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  use_render = data.get("use_render", False)
  data_type = "render" if use_render else "generation"

  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1

  print(f"\n{'=' * 60}")
  print(
    f"📤 Export request: ({tl_x},{tl_y}) to ({br_x},{br_y}) "
    f"({width_count}x{height_count} quadrants, {data_type})"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    quadrant_images: dict[tuple[int, int], Image.Image] = {}
    missing_quadrants = []

    for dy in range(height_count):
      for dx in range(width_count):
        qx, qy = tl_x + dx, tl_y + dy

        # Get the appropriate data (render or generation)
        if use_render:
          img_bytes = get_quadrant_render(qx, qy)
        else:
          img_bytes = get_quadrant_generation(conn, qx, qy)

        if img_bytes is None:
          missing_quadrants.append((qx, qy))
        else:
          quadrant_images[(dx, dy)] = png_bytes_to_image(img_bytes)
          print(f"   ✓ Quadrant ({qx}, {qy})")

    if missing_quadrants:
      print(
        f"❌ Export failed: Missing {data_type} for {len(missing_quadrants)} quadrant(s)"
      )
      return jsonify(
        {
          "success": False,
          "error": f"Missing {data_type} for quadrants: {missing_quadrants}",
        }
      ), 400

    # Stitch quadrants into a single image
    sample_quad = next(iter(quadrant_images.values()))
    quad_w, quad_h = sample_quad.size

    tile_image = Image.new("RGBA", (quad_w * width_count, quad_h * height_count))
    for (dx, dy), quad_img in quadrant_images.items():
      pos = (dx * quad_w, dy * quad_h)
      tile_image.paste(quad_img, pos)

    # Convert to PNG bytes
    buffer = io.BytesIO()
    tile_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Generate filename
    filename = f"export_tl_{tl_x}_{tl_y}_br_{br_x}_{br_y}.png"

    print(f"✅ Export complete: {tile_image.size[0]}x{tile_image.size[1]} pixels")

    return Response(
      buffer.getvalue(),
      mimetype="image/png",
      headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/generate-rectangle", methods=["POST"])
def api_generate_rectangle():
  """
  API endpoint to generate all quadrants within a rectangle.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "model_id": "optional-model-id"    // Optional model ID
    }

  Returns:
    {
      "success": true,
      "plan_summary": {...},
      "queued_count": N,
      "message": "Queued N generation steps"
    }
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl = Point(int(tl_raw[0]), int(tl_raw[1]))
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl = Point(int(tl_raw["x"]), int(tl_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br = Point(int(br_raw[0]), int(br_raw[1]))
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br = Point(int(br_raw["x"]), int(br_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  # Get optional model_id
  model_id = data.get("model_id")

  print(f"\n{'=' * 60}")
  print(f"📐 Rectangle generation request: ({tl.x},{tl.y}) to ({br.x},{br.y})")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Load existing generated quadrants and pending/processing quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)
    queued = load_queued_quadrants(conn)

    if queued:
      print(
        f"   Considering {len(queued)} queued/processing quadrant(s) for seam avoidance"
      )

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated, queued)

    # Validate the plan
    is_valid, errors = validate_plan(plan)
    if not is_valid:
      print(f"❌ Invalid plan generated: {errors}")
      return jsonify(
        {
          "success": False,
          "error": f"Internal error: invalid plan generated - {errors}",
        }
      ), 500

    # Get plan summary for response
    summary = get_plan_summary(plan)

    if len(plan.steps) == 0:
      print("ℹ️  No quadrants to generate (all already generated)")
      return jsonify(
        {
          "success": True,
          "plan_summary": summary,
          "queued_count": 0,
          "message": "No quadrants to generate - all already generated",
        }
      )

    # Queue all generation steps
    queued_count = 0
    for step in plan.steps:
      quadrants = [(q.x, q.y) for q in step.quadrants]
      add_to_queue(conn, QueueItemType.GENERATE, quadrants, model_id)
      queued_count += 1

    # Ensure queue worker is running
    start_queue_worker()

    print(f"✅ Queued {queued_count} generation step(s)")
    print(f"   Steps by type: {summary['steps_by_type']}")

    return jsonify(
      {
        "success": True,
        "plan_summary": summary,
        "queued_count": queued_count,
        "message": f"Queued {queued_count} generation step(s) for {summary['total_quadrants']} quadrant(s)",
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG, BOUNDARY_GEOJSON, NO_GENERATE_MODE

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to app_config.json (default: looks in the e2e_generation directory)",
  )
  parser.add_argument(
    "--bounds",
    type=Path,
    default=None,
    help="Path to custom bounds GeoJSON file (default: NYC boundary)",
  )
  parser.add_argument(
    "--no-generate",
    action="store_true",
    default=False,
    help="Disable generation processing (queue items are preserved but not processed)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port
  NO_GENERATE_MODE = args.no_generate

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Load app configuration
  APP_CONFIG = load_app_config(args.config)
  print(f"📦 Loaded {len(APP_CONFIG.models)} model(s) from configuration")
  for model in APP_CONFIG.models:
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Load boundary GeoJSON
  bounds_path = args.bounds.resolve() if args.bounds else None
  BOUNDARY_GEOJSON = load_bounds(bounds_path)
  bounds_name = bounds_path.name if bounds_path else "NYC (default)"
  print(f"📍 Boundary: {bounds_name}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Reset any items that were mid-processing when server shut down
    # These will be retried automatically
    reset_count = reset_all_processing_items(conn)
    if reset_count > 0:
      print(f"🔄 Reset {reset_count} interrupted generation(s) - will be retried")
    # Clean up old completed items
    deleted_count = clear_completed_items(conn)
    if deleted_count > 0:
      print(f"🧹 Cleaned up {deleted_count} old completed queue item(s)")
  finally:
    conn.close()

  # Start the queue worker
  start_queue_worker()

  # Start the global web renderer
  try:
    start_global_renderer(port=WEB_SERVER_PORT)
  except Exception as e:
    print(f"⚠️  Failed to start web renderer: {e}")
    print("   Rendering will start on demand")

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  if NO_GENERATE_MODE:
    print("   ⚠️  NO-GENERATE MODE: Queue items preserved but not processed")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web renderer
    print("🛑 Stopping web renderer...")
    stop_global_renderer()

  return 0


if __name__ == "__main__":
  exit(main())

```

```javascript
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

  // Check if we're in render view mode
  const isRenderMode = document.getElementById("showRender")?.checked || false;
  const dataType = isRenderMode ? "render" : "generation";
  const apiEndpoint = isRenderMode ? "/api/delete-render" : "/api/delete";

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
  const useRender = document.getElementById("showRender")?.checked || false;
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

```

Now I need to update the `updateSelectionStatus()` function to enable/disable the new button when exactly 2 quadrants are selected:

```javascript
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

  // Check if we're in render view mode
  const isRenderMode = document.getElementById("showRender")?.checked || false;
  const dataType = isRenderMode ? "render" : "generation";
  const apiEndpoint = isRenderMode ? "/api/delete-render" : "/api/delete";

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
  const useRender = document.getElementById("showRender")?.checked || false;
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

```

I've implemented the new "Fill Rectangle Water" feature. Here's a summary of the changes:

## Changes Made

### 1. `templates/viewer.html` - Added Button
Added a new button `💧 Fill Rect` in the toolbar, placed after the "Gen Rect" button. It's enabled only when exactly 2 quadrants are selected.

### 2. `app.py` - Added API Endpoint
Added `/api/water-fill-rectangle` endpoint that:
- Accepts `tl` (top-left) and `br` (bottom-right) coordinates
- Validates the bounds
- Fills all quadrants in the rectangle with the water color `#4A6372`
- Returns the count of filled quadrants

### 3. `viewer.js` - Added JavaScript Handler
Added `fillRectangleWater()` function that:
- Requires exactly 2 quadrants selected (shows error otherwise)
- Calculates the rectangle bounds from the two selected corners
- Shows a confirmation dialog similar to delete, warning that it will **OVERWRITE** existing generations:
  ```
  Fill rectangle from (X1, Y1) to (X2, Y2) with water color?
  
  Size: W × H = N quadrant(s)
  
  ⚠️ This will OVERWRITE any existing generations in this area with solid water color (#4A6372).
  ```
- Shows loading state on the button while processing
- Refreshes all tile images in the rectangle after completion
- Also updated `updateSelectionStatus()` to enable/disable the button when exactly 2 quadrants are selected

## Usage
1. Enable the Select tool (S key or click "Select" button)
2. Select exactly 2 quadrants (the top-left and bottom-right corners of your desired rectangle)
3. Click the "💧 Fill Rect" button
4. Confirm the action in the dialog
5. All quadrants in the rectangle will be filled with the water color

```plaintext
  Fill rectangle from (X1, Y1) to (X2, Y2) with water color?
  
  Size: W × H = N quadrant(s)
  
  ⚠️ This will OVERWRITE any existing generations in this area with solid water color (#4A6372).
```

---

**User**

What's the compression image quality of the pmtiles by default?

---

**Cursor**

Based on the code in `export_pmtiles.py`, here are the default compression/quality settings:

## Default Image Format: **PNG**
- Uses lossless compression with `optimize=True`
- Larger file sizes but no quality loss

## Optional WebP Format
- Enabled with `--webp` flag
- **Default quality: 85** (defined as `DEFAULT_WEBP_QUALITY = 85`)
- Configurable via `--webp-quality` argument (0-100)
- Typically 25-35% smaller files than PNG

## PMTiles Container Compression
- **Tile compression: NONE** (`Compression.NONE`)
- The image format itself handles compression, so no additional container-level compression is applied

### Relevant code:

```80:80:src/isometric_nyc/e2e_generation/export_pmtiles.py
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
```

```572:576:src/isometric_nyc/e2e_generation/export_pmtiles.py
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
```

### Usage examples:
```bash
# Default PNG (lossless)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

# WebP with default quality (85)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

# WebP with custom quality
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp --webp-quality 90
```

```python
DEFAULT_WEBP_QUALITY = 85  # Good balance of quality and size
```

```python
  if format == FORMAT_WEBP:
    # WebP with lossy compression - much smaller than PNG
    img.save(buffer, format="WEBP", quality=webp_quality, method=4)
  else:
    img.save(buffer, format="PNG", optimize=True)
```

```shellscript
# Default PNG (lossless)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

# WebP with default quality (85)
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp

# WebP with custom quality
uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --webp --webp-quality 90
```

