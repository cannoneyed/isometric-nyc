# Generation app improvements plan
_Exported on 12/12/2025 at 14:12:05 EST from Cursor (2.1.39)_

---

**User**

Implement the plan in @tasks/023_generation_app_improvements.md 

---

**Cursor**

I'll implement the plan from the task file. Let me start by exploring the current codebase to understand the existing structure.

Now I have a comprehensive understanding of the codebase. Let me create a task list and implement these improvements.

Let me start implementing. First, I'll create a new module for the persistent queue and web renderer.

```python
"""
Database-backed generation queue.

Provides a robust, persistent queue for generation and render requests
that survives server restarts and doesn't rely on client-side state.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class QueueItemType(str, Enum):
  GENERATE = "generate"
  RENDER = "render"


class QueueItemStatus(str, Enum):
  PENDING = "pending"
  PROCESSING = "processing"
  COMPLETE = "complete"
  ERROR = "error"


@dataclass
class QueueItem:
  """Represents a single item in the generation queue."""

  id: int
  item_type: QueueItemType
  quadrants: list[tuple[int, int]]
  model_id: str | None
  status: QueueItemStatus
  created_at: float
  started_at: float | None
  completed_at: float | None
  error_message: str | None
  result_message: str | None

  @classmethod
  def from_row(cls, row: tuple) -> "QueueItem":
    """Create a QueueItem from a database row."""
    return cls(
      id=row[0],
      item_type=QueueItemType(row[1]),
      quadrants=json.loads(row[2]),
      model_id=row[3],
      status=QueueItemStatus(row[4]),
      created_at=row[5],
      started_at=row[6],
      completed_at=row[7],
      error_message=row[8],
      result_message=row[9],
    )

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization."""
    return {
      "id": self.id,
      "type": self.item_type.value,
      "quadrants": self.quadrants,
      "model_id": self.model_id,
      "status": self.status.value,
      "created_at": self.created_at,
      "started_at": self.started_at,
      "completed_at": self.completed_at,
      "error_message": self.error_message,
      "result_message": self.result_message,
    }


def init_queue_table(conn: sqlite3.Connection) -> None:
  """Initialize the generation_queue table if it doesn't exist."""
  cursor = conn.cursor()
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS generation_queue (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      item_type TEXT NOT NULL,
      quadrants TEXT NOT NULL,
      model_id TEXT,
      status TEXT NOT NULL DEFAULT 'pending',
      created_at REAL NOT NULL,
      started_at REAL,
      completed_at REAL,
      error_message TEXT,
      result_message TEXT
    )
  """)
  # Create index on status for efficient queue queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_queue_status ON generation_queue(status)
  """)
  conn.commit()


def add_to_queue(
  conn: sqlite3.Connection,
  item_type: QueueItemType,
  quadrants: list[tuple[int, int]],
  model_id: str | None = None,
) -> QueueItem:
  """
  Add a new item to the generation queue.

  Args:
    conn: Database connection
    item_type: Type of operation (generate or render)
    quadrants: List of (x, y) quadrant coordinates
    model_id: Optional model ID for generation

  Returns:
    The created QueueItem
  """
  cursor = conn.cursor()
  created_at = time.time()

  cursor.execute(
    """
    INSERT INTO generation_queue (item_type, quadrants, model_id, status, created_at)
    VALUES (?, ?, ?, ?, ?)
    """,
    (
      item_type.value,
      json.dumps(quadrants),
      model_id,
      QueueItemStatus.PENDING.value,
      created_at,
    ),
  )
  conn.commit()

  item_id = cursor.lastrowid
  return QueueItem(
    id=item_id,
    item_type=item_type,
    quadrants=quadrants,
    model_id=model_id,
    status=QueueItemStatus.PENDING,
    created_at=created_at,
    started_at=None,
    completed_at=None,
    error_message=None,
    result_message=None,
  )


def get_next_pending_item(conn: sqlite3.Connection) -> QueueItem | None:
  """
  Get the next pending item from the queue.

  Returns the oldest pending item, or None if queue is empty.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message
    FROM generation_queue
    WHERE status = ?
    ORDER BY created_at ASC
    LIMIT 1
    """,
    (QueueItemStatus.PENDING.value,),
  )
  row = cursor.fetchone()
  return QueueItem.from_row(row) if row else None


def get_processing_item(conn: sqlite3.Connection) -> QueueItem | None:
  """
  Get the currently processing item, if any.

  Returns the item currently being processed, or None.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message
    FROM generation_queue
    WHERE status = ?
    ORDER BY started_at DESC
    LIMIT 1
    """,
    (QueueItemStatus.PROCESSING.value,),
  )
  row = cursor.fetchone()
  return QueueItem.from_row(row) if row else None


def mark_item_processing(conn: sqlite3.Connection, item_id: int) -> None:
  """Mark a queue item as processing."""
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, started_at = ?
    WHERE id = ?
    """,
    (QueueItemStatus.PROCESSING.value, time.time(), item_id),
  )
  conn.commit()


def mark_item_complete(
  conn: sqlite3.Connection, item_id: int, result_message: str | None = None
) -> None:
  """Mark a queue item as complete."""
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, completed_at = ?, result_message = ?
    WHERE id = ?
    """,
    (QueueItemStatus.COMPLETE.value, time.time(), result_message, item_id),
  )
  conn.commit()


def mark_item_error(
  conn: sqlite3.Connection, item_id: int, error_message: str
) -> None:
  """Mark a queue item as errored."""
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, completed_at = ?, error_message = ?
    WHERE id = ?
    """,
    (QueueItemStatus.ERROR.value, time.time(), error_message, item_id),
  )
  conn.commit()


def get_pending_queue(conn: sqlite3.Connection) -> list[QueueItem]:
  """Get all pending items in the queue, ordered by creation time."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, item_type, quadrants, model_id, status,
           created_at, started_at, completed_at, error_message, result_message
    FROM generation_queue
    WHERE status = ?
    ORDER BY created_at ASC
    """,
    (QueueItemStatus.PENDING.value,),
  )
  return [QueueItem.from_row(row) for row in cursor.fetchall()]


def get_queue_position(conn: sqlite3.Connection, item_id: int) -> int:
  """
  Get the position of an item in the queue.

  Returns 0 if the item is currently processing,
  1 if it's first in the pending queue, etc.
  Returns -1 if the item is not found or already complete.
  """
  cursor = conn.cursor()

  # Check if it's processing
  cursor.execute(
    "SELECT 1 FROM generation_queue WHERE id = ? AND status = ?",
    (item_id, QueueItemStatus.PROCESSING.value),
  )
  if cursor.fetchone():
    return 0

  # Check if it's pending and get position
  cursor.execute(
    """
    SELECT COUNT(*) + 1
    FROM generation_queue
    WHERE status = ? AND created_at < (
      SELECT created_at FROM generation_queue WHERE id = ?
    )
    """,
    (QueueItemStatus.PENDING.value, item_id),
  )
  row = cursor.fetchone()
  if row and row[0] > 0:
    return row[0]

  return -1


def get_queue_status(conn: sqlite3.Connection) -> dict[str, Any]:
  """
  Get a summary of the current queue status.

  Returns a dictionary with:
    - is_processing: bool
    - current_item: dict | None
    - pending_count: int
    - pending_items: list of dicts
  """
  processing = get_processing_item(conn)
  pending = get_pending_queue(conn)

  return {
    "is_processing": processing is not None,
    "current_item": processing.to_dict() if processing else None,
    "pending_count": len(pending),
    "pending_items": [item.to_dict() for item in pending],
  }


def cleanup_stale_processing(
  conn: sqlite3.Connection, max_age_seconds: float = 3600.0
) -> int:
  """
  Clean up items stuck in 'processing' state for too long.

  This handles cases where the server crashed during processing.
  Items older than max_age_seconds are reset to 'pending'.

  Returns the number of items reset.
  """
  cursor = conn.cursor()
  cutoff_time = time.time() - max_age_seconds

  cursor.execute(
    """
    UPDATE generation_queue
    SET status = ?, started_at = NULL
    WHERE status = ? AND started_at < ?
    """,
    (QueueItemStatus.PENDING.value, QueueItemStatus.PROCESSING.value, cutoff_time),
  )
  conn.commit()
  return cursor.rowcount


def clear_completed_items(
  conn: sqlite3.Connection, max_age_seconds: float = 86400.0
) -> int:
  """
  Delete completed/errored items older than max_age_seconds.

  Returns the number of items deleted.
  """
  cursor = conn.cursor()
  cutoff_time = time.time() - max_age_seconds

  cursor.execute(
    """
    DELETE FROM generation_queue
    WHERE status IN (?, ?) AND completed_at < ?
    """,
    (QueueItemStatus.COMPLETE.value, QueueItemStatus.ERROR.value, cutoff_time),
  )
  conn.commit()
  return cursor.rowcount

```

```python
"""
Model configuration for the generation app.

Loads model configurations from app_config.json and provides
utilities for selecting and using different AI models.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
  """Configuration for a single AI model."""

  name: str
  model_id: str
  api_key_env: str  # Environment variable name for the API key
  endpoint: str = "https://hub.oxen.ai/api/images/edit"
  num_inference_steps: int = 28

  @property
  def api_key(self) -> str | None:
    """Get the API key from environment variables."""
    return os.getenv(self.api_key_env)

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization (without API key)."""
    return {
      "name": self.name,
      "model_id": self.model_id,
      "endpoint": self.endpoint,
      "num_inference_steps": self.num_inference_steps,
    }


@dataclass
class AppConfig:
  """Full application configuration."""

  models: list[ModelConfig]
  default_model_id: str | None = None

  def get_model(self, model_id: str) -> ModelConfig | None:
    """Get a model configuration by its ID."""
    for model in self.models:
      if model.model_id == model_id:
        return model
    return None

  def get_default_model(self) -> ModelConfig | None:
    """Get the default model, or the first model if no default is set."""
    if self.default_model_id:
      return self.get_model(self.default_model_id)
    return self.models[0] if self.models else None

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization."""
    return {
      "models": [m.to_dict() for m in self.models],
      "default_model_id": self.default_model_id,
    }


def load_app_config(config_path: Path | None = None) -> AppConfig:
  """
  Load the application configuration from app_config.json.

  Args:
    config_path: Path to the config file. If None, looks in the generation dir.

  Returns:
    AppConfig object with model configurations

  If the config file doesn't exist, returns a default configuration
  with the legacy Oxen models.
  """
  if config_path is None:
    config_path = Path(__file__).parent / "app_config.json"

  if not config_path.exists():
    # Return default configuration with legacy models
    return get_default_config()

  with open(config_path) as f:
    data = json.load(f)

  models = []
  for model_data in data.get("models", []):
    models.append(
      ModelConfig(
        name=model_data["name"],
        model_id=model_data["model_id"],
        api_key_env=model_data["api_key_env"],
        endpoint=model_data.get("endpoint", "https://hub.oxen.ai/api/images/edit"),
        num_inference_steps=model_data.get("num_inference_steps", 28),
      )
    )

  return AppConfig(
    models=models,
    default_model_id=data.get("default_model_id"),
  )


def get_default_config() -> AppConfig:
  """
  Get the default configuration with legacy Oxen models.

  This is used when no app_config.json exists.
  """
  return AppConfig(
    models=[
      ModelConfig(
        name="Omni Water v1",
        model_id="cannoneyed-quiet-green-lamprey",
        api_key_env="OXEN_OMNI_v04_WATER_API_KEY",
      ),
      ModelConfig(
        name="Omni Water v2",
        model_id="cannoneyed-rural-rose-dingo",
        api_key_env="OXEN_OMNI_WATER_V2_API_KEY",
      ),
      ModelConfig(
        name="Omni (Original)",
        model_id="cannoneyed-gentle-gold-antlion",
        api_key_env="OXEN_OMNI_API_KEY",
      ),
    ],
    default_model_id="cannoneyed-quiet-green-lamprey",
  )


def save_app_config(config: AppConfig, config_path: Path | None = None) -> None:
  """
  Save the application configuration to app_config.json.

  Note: This does NOT save API keys - those should remain in environment variables.
  """
  if config_path is None:
    config_path = Path(__file__).parent / "app_config.json"

  data = {
    "models": [],
    "default_model_id": config.default_model_id,
  }

  for model in config.models:
    data["models"].append(
      {
        "name": model.name,
        "model_id": model.model_id,
        "api_key_env": model.api_key_env,
        "endpoint": model.endpoint,
        "num_inference_steps": model.num_inference_steps,
      }
    )

  with open(config_path, "w") as f:
    json.dump(data, f, indent=2)

```

Now let me create the persisted web renderer:

```python
"""
Persisted web renderer for tile generation.

This module manages a single web server process that stays running
for the lifetime of the application, with an internal queue for
handling render requests.
"""

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  image_to_png_bytes,
  split_tile_into_quadrants,
  wait_for_server,
)


@dataclass
class RenderRequest:
  """A request to render a quadrant."""

  quadrant_x: int
  quadrant_y: int
  lat: float
  lng: float
  width_px: int
  height_px: int
  camera_azimuth_degrees: float
  camera_elevation_degrees: float
  view_height_meters: float
  callback: Callable[[dict[tuple[int, int], bytes] | None, str | None], None]


class WebRenderer:
  """
  Manages a persistent web server and render queue.

  This class:
  - Starts a web server (bun/vite) when initialized
  - Maintains an internal queue of render requests
  - Processes renders one at a time using Playwright
  - Provides thread-safe access to queue operations
  """

  def __init__(self, web_dir: Path = WEB_DIR, port: int = DEFAULT_WEB_PORT):
    self.web_dir = web_dir
    self.port = port
    self.web_process: subprocess.Popen | None = None
    self.render_queue: queue.Queue[RenderRequest | None] = queue.Queue()
    self.worker_thread: threading.Thread | None = None
    self.running = False
    self._lock = threading.Lock()

  def start(self) -> None:
    """Start the web server and render worker thread."""
    with self._lock:
      if self.running:
        return

      print(f"🌐 Starting web renderer on port {self.port}...")

      # Start the web server
      self._start_web_server()

      # Start the worker thread
      self.running = True
      self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
      self.worker_thread.start()

      print(f"✅ Web renderer ready on http://localhost:{self.port}")

  def stop(self) -> None:
    """Stop the web server and worker thread."""
    with self._lock:
      if not self.running:
        return

      print("🛑 Stopping web renderer...")
      self.running = False

      # Signal worker to stop
      self.render_queue.put(None)

      # Wait for worker to finish
      if self.worker_thread and self.worker_thread.is_alive():
        self.worker_thread.join(timeout=5.0)

      # Stop the web server
      if self.web_process:
        self.web_process.terminate()
        try:
          self.web_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
          self.web_process.kill()
        self.web_process = None

      print("✅ Web renderer stopped")

  def _start_web_server(self) -> None:
    """Start the web server subprocess."""
    if not self.web_dir.exists():
      raise RuntimeError(f"Web directory not found: {self.web_dir}")

    self.web_process = subprocess.Popen(
      ["bun", "run", "dev", "--port", str(self.port)],
      cwd=self.web_dir,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )

    # Give the process a moment to start
    time.sleep(2)

    # Check if process died immediately
    if self.web_process.poll() is not None:
      stdout = self.web_process.stdout.read().decode() if self.web_process.stdout else ""
      stderr = self.web_process.stderr.read().decode() if self.web_process.stderr else ""
      raise RuntimeError(
        f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
      )

    # Wait for server to be ready
    print("   ⏳ Waiting for web server to be ready...")
    if wait_for_server(self.port, timeout=30.0):
      print(f"   ✅ Web server ready on http://localhost:{self.port}")
    else:
      if self.web_process.poll() is not None:
        stdout = self.web_process.stdout.read().decode() if self.web_process.stdout else ""
        stderr = self.web_process.stderr.read().decode() if self.web_process.stderr else ""
        raise RuntimeError(
          f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
        )
      print("   ⚠️  Web server may not be fully ready, continuing anyway...")

  def _ensure_web_server_running(self) -> bool:
    """Ensure the web server is still running, restart if needed."""
    if self.web_process is None or self.web_process.poll() is not None:
      print("   ⚠️  Web server died, restarting...")
      try:
        self._start_web_server()
        return True
      except Exception as e:
        print(f"   ❌ Failed to restart web server: {e}")
        return False
    return True

  def _worker_loop(self) -> None:
    """Main worker loop that processes render requests."""
    print("🔄 Render worker started")

    while self.running:
      try:
        # Wait for a request with timeout to allow checking running flag
        try:
          request = self.render_queue.get(timeout=1.0)
        except queue.Empty:
          continue

        # Check for stop signal
        if request is None:
          break

        # Ensure web server is running
        if not self._ensure_web_server_running():
          request.callback(None, "Web server not available")
          continue

        # Process the render request
        try:
          result = self._render_tile(request)
          request.callback(result, None)
        except Exception as e:
          print(f"   ❌ Render error: {e}")
          request.callback(None, str(e))

      except Exception as e:
        print(f"   ❌ Worker loop error: {e}")

    print("🛑 Render worker stopped")

  def _render_tile(
    self, request: RenderRequest
  ) -> dict[tuple[int, int], bytes]:
    """
    Render a tile and return all 4 quadrants as PNG bytes.

    Returns a dict mapping (dx, dy) offset to PNG bytes.
    """
    # Build URL for rendering
    params = {
      "export": "true",
      "lat": request.lat,
      "lon": request.lng,
      "width": request.width_px,
      "height": request.height_px,
      "azimuth": request.camera_azimuth_degrees,
      "elevation": request.camera_elevation_degrees,
      "view_height": request.view_height_meters,
    }
    query_string = urlencode(params)
    url = f"http://localhost:{self.port}/?{query_string}"

    print(f"   🎨 Rendering tile at ({request.lat:.6f}, {request.lng:.6f})...")

    # Render using Playwright
    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": request.width_px, "height": request.height_px},
        device_scale_factor=1,
      )
      page = context.new_page()

      page.goto(url, wait_until="networkidle")

      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception:
        print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

      # Get screenshot as bytes
      screenshot_bytes = page.screenshot(type="png")

      page.close()
      context.close()
      browser.close()

    # Open as PIL image and split into quadrants
    full_tile = Image.open(BytesIO(screenshot_bytes))
    quadrant_images = split_tile_into_quadrants(full_tile)

    # Convert to bytes
    result = {}
    for offset, quad_img in quadrant_images.items():
      result[offset] = image_to_png_bytes(quad_img)

    return result

  def render_quadrant(
    self,
    quadrant_x: int,
    quadrant_y: int,
    lat: float,
    lng: float,
    width_px: int,
    height_px: int,
    camera_azimuth_degrees: float,
    camera_elevation_degrees: float,
    view_height_meters: float,
    timeout: float = 120.0,
  ) -> dict[tuple[int, int], bytes]:
    """
    Synchronously render a quadrant and wait for the result.

    This is a blocking call that waits for the render to complete.

    Args:
      quadrant_x, quadrant_y: Quadrant coordinates
      lat, lng: Geographic coordinates for the render
      width_px, height_px: Tile dimensions
      camera_azimuth_degrees, camera_elevation_degrees: Camera angles
      view_height_meters: View height
      timeout: Maximum time to wait for render

    Returns:
      Dict mapping (dx, dy) offset to PNG bytes for all 4 quadrants

    Raises:
      RuntimeError: If render fails or times out
    """
    result_event = threading.Event()
    result_data: dict[str, Any] = {"result": None, "error": None}

    def callback(result: dict[tuple[int, int], bytes] | None, error: str | None) -> None:
      result_data["result"] = result
      result_data["error"] = error
      result_event.set()

    request = RenderRequest(
      quadrant_x=quadrant_x,
      quadrant_y=quadrant_y,
      lat=lat,
      lng=lng,
      width_px=width_px,
      height_px=height_px,
      camera_azimuth_degrees=camera_azimuth_degrees,
      camera_elevation_degrees=camera_elevation_degrees,
      view_height_meters=view_height_meters,
      callback=callback,
    )

    self.render_queue.put(request)

    if not result_event.wait(timeout=timeout):
      raise RuntimeError(f"Render timed out after {timeout}s")

    if result_data["error"]:
      raise RuntimeError(result_data["error"])

    return result_data["result"]

  @property
  def queue_size(self) -> int:
    """Get the current size of the render queue."""
    return self.render_queue.qsize()

  @property
  def is_running(self) -> bool:
    """Check if the renderer is running."""
    return self.running


# Global singleton instance
_renderer: WebRenderer | None = None
_renderer_lock = threading.Lock()


def get_web_renderer(
  web_dir: Path = WEB_DIR, port: int = DEFAULT_WEB_PORT
) -> WebRenderer:
  """
  Get the global web renderer instance, creating it if necessary.

  This provides a singleton instance of the WebRenderer that can be
  shared across the application.
  """
  global _renderer

  with _renderer_lock:
    if _renderer is None:
      _renderer = WebRenderer(web_dir, port)
    return _renderer


def start_global_renderer(
  web_dir: Path = WEB_DIR, port: int = DEFAULT_WEB_PORT
) -> WebRenderer:
  """Start the global web renderer."""
  renderer = get_web_renderer(web_dir, port)
  renderer.start()
  return renderer


def stop_global_renderer() -> None:
  """Stop the global web renderer."""
  global _renderer

  with _renderer_lock:
    if _renderer is not None:
      _renderer.stop()
      _renderer = None

```

```json
{
  "models": [
    {
      "name": "Omni Water v1",
      "model_id": "cannoneyed-quiet-green-lamprey",
      "api_key_env": "OXEN_OMNI_v04_WATER_API_KEY",
      "endpoint": "https://hub.oxen.ai/api/images/edit",
      "num_inference_steps": 28
    },
    {
      "name": "Omni Water v2",
      "model_id": "cannoneyed-rural-rose-dingo",
      "api_key_env": "OXEN_OMNI_WATER_V2_API_KEY",
      "endpoint": "https://hub.oxen.ai/api/images/edit",
      "num_inference_steps": 28
    },
    {
      "name": "Omni (Original)",
      "model_id": "cannoneyed-gentle-gold-antlion",
      "api_key_env": "OXEN_OMNI_API_KEY",
      "endpoint": "https://hub.oxen.ai/api/images/edit",
      "num_inference_steps": 28
    }
  ],
  "default_model_id": "cannoneyed-quiet-green-lamprey"
}

```

Now let me update the main `app.py` to integrate all these new components:

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
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemStatus,
  QueueItemType,
  add_to_queue,
  cleanup_stale_processing,
  clear_completed_items,
  get_next_pending_item,
  get_pending_queue,
  get_processing_item,
  get_queue_status,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
)
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
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

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


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

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

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


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

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
      SELECT item_type, quadrants, model_id
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

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting" if item_type == QueueItemType.GENERATE else "rendering"
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}")
    if model_id:
      print(f"   Model: {model_id}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      result = run_generation(conn, config, selected_quadrants, model_id)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        mark_item_complete(conn, item_id, result["message"])
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        mark_item_error(conn, item_id, result["error"])
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


def queue_worker():
  """Background worker that processes the generation queue from the database."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    conn = None
    try:
      conn = get_db_connection()

      # Get next pending item
      item = get_next_pending_item(conn)

      if item is None:
        # No items in queue, wait a bit and check again
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      conn.close()
      conn = None

      # Acquire the generation lock and process the item
      with generation_lock:
        try:
          process_queue_item_from_db(item_id)
        finally:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None

      # Small delay between items
      time.sleep(0.5)

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
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
    )

    # Get queue position
    pending = get_pending_queue(conn)
    queue_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": queue_position,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {queue_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
    }

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify({
    "models": [m.to_dict() for m in APP_CONFIG.models],
    "default_model_id": APP_CONFIG.default_model_id,
  })


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
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
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
  """API endpoint to generate tiles for selected quadrants."""
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
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "generate", model_id)
  return jsonify(result), 202  # 202 Accepted


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG

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

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

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
    default_marker = " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Clean up any stale processing items from previous runs
    stale_count = cleanup_stale_processing(conn)
    if stale_count > 0:
      print(f"⚠️  Reset {stale_count} stale processing item(s) from previous run")
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

Now let me update `generate_omni.py` to accept the model configuration:

```python
"""
Reusable library for generating pixel art using the Oxen.ai model.

This module provides the core generation logic that can be used by:
- view_generations.py (Flask web server)
- generate_tiles_omni.py (command-line script)
- automatic_generation.py (automated generation)

The main entry point is `run_generation_for_quadrants()` which handles:
1. Validating the quadrant selection
2. Rendering any missing quadrants
3. Building the template image
4. Uploading to GCS and calling the Oxen API
5. Saving the generated quadrants to the database
"""

import os
import re
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  ensure_quadrant_exists,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  upload_to_gcs,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation as shared_get_quadrant_generation,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_render as shared_get_quadrant_render,
)

# Load environment variables
load_dotenv()

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
OMNI_WATER_MODEL_ID = "cannoneyed-quiet-green-lamprey"
OMNI_WATER_V2_MODEL_ID = "cannoneyed-rural-rose-dingo"

GCS_BUCKET_NAME = "isometric-nyc-infills"


# =============================================================================
# Quadrant Parsing Utilities
# =============================================================================


def parse_quadrant_tuple(s: str) -> tuple[int, int]:
  """
  Parse a quadrant tuple string like "(0,1)" or "0,1" into a tuple.

  Args:
      s: String in format "(x,y)" or "x,y"

  Returns:
      Tuple of (x, y) coordinates

  Raises:
      ValueError: If the format is invalid
  """
  s = s.strip()
  # Remove optional parentheses
  if s.startswith("(") and s.endswith(")"):
    s = s[1:-1]
  parts = s.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid quadrant tuple format: {s}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def parse_quadrant_list(s: str) -> list[tuple[int, int]]:
  """
  Parse a comma-separated list of quadrant tuples.

  Args:
      s: String like "(0,1),(0,2)" or "(0,1), (0,2)"

  Returns:
      List of (x, y) coordinate tuples

  Raises:
      ValueError: If the format is invalid
  """
  # Use regex to find all (x,y) patterns
  pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
  matches = re.findall(pattern, s)
  if not matches:
    raise ValueError(f"No valid quadrant tuples found in: {s}")
  return [(int(x), int(y)) for x, y in matches]


# =============================================================================
# Oxen API Functions
# =============================================================================


def call_oxen_api(
  image_url: str,
  model_config: "ModelConfig | None" = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image
      model_config: Optional model configuration. If not provided, uses defaults.

  Returns:
      URL of the generated image

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  from isometric_nyc.e2e_generation.model_config import ModelConfig

  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    model_id = model_config.model_id
    api_key = model_config.api_key
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "https://hub.oxen.ai/api/images/edit"
    model_id = OMNI_WATER_MODEL_ID
    api_key = os.getenv("OXEN_OMNI_v04_WATER_API_KEY")
    num_inference_steps = 28

  if not api_key:
    raise ValueError(f"API key not found for model {model_id}")

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  prompt = (
    "Fill in the outlined section with the missing pixels corresponding to "
    "the <isometric nyc pixel art> style, removing the border and exactly "
    "following the shape/style/structure of the surrounding image (if present)."
  )

  payload = {
    "model": model_id,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": num_inference_steps,
  }

  print(f"   🤖 Calling Oxen API with model {model_id}...")
  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()

  if "images" in result and len(result["images"]) > 0:
    return result["images"][0]["url"]
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  else:
    raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> Image.Image:
  """
  Download an image from a URL and return as PIL Image.

  Args:
      url: URL of the image to download

  Returns:
      PIL Image object
  """
  response = requests.get(url, timeout=120)
  response.raise_for_status()
  return Image.open(BytesIO(response.content))


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
      conn: Database connection
      config: Generation config dict
      x: Quadrant x coordinate
      y: Quadrant y coordinate
      port: Web server port for rendering

  Returns:
      PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  # Build URL for rendering
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  # Render using Playwright
  with sync_playwright() as p:
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

    context = browser.new_context(
      viewport={"width": config["width_px"], "height": config["height_px"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    page.goto(url, wait_until="networkidle")

    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
    except Exception:
      print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_generation_for_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = GCS_BUCKET_NAME,
  status_callback: Callable[[str, str], None] | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Uploads to GCS and calls the Oxen API
  5. Saves the generated quadrants to the database

  Args:
      conn: Database connection
      config: Generation config dict
      selected_quadrants: List of (x, y) quadrant coordinates to generate
      port: Web server port for rendering (default: 5173)
      bucket_name: GCS bucket name for uploads
      status_callback: Optional callback(status, message) for progress updates

  Returns:
      Dict with:
          - success: bool
          - message: str (on success)
          - error: str (on failure)
          - quadrants: list of generated quadrant coords (on success)
  """

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = shared_get_quadrant_generation(conn, qx, qy)
    return gen is not None

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)
    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result

  # Save template to temp file and upload to GCS
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = Path(tmp.name)
    template_image.save(template_path)

  try:
    update_status("uploading", "Uploading template to cloud...")
    print("📤 Uploading template to GCS...")
    image_url = upload_to_gcs(template_path, bucket_name)

    update_status("generating", "Calling AI model (this may take a minute)...")
    print("🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url)

    update_status("saving", "Downloading and saving results...")
    print("📥 Downloading generated image...")
    generated_image = download_image_to_pil(generated_url)

    # Extract quadrants from generated image and save to database
    print("💾 Saving generated quadrants to database...")

    # Figure out what quadrants are in the infill region
    all_infill_quadrants = (
      placement.all_infill_quadrants
      if placement.all_infill_quadrants
      else region.overlapping_quadrants()
    )

    # For each infill quadrant, extract pixels from the generated image
    saved_count = 0
    for qx, qy in all_infill_quadrants:
      # Calculate position in the generated image
      quad_world_x = qx * QUADRANT_SIZE
      quad_world_y = qy * QUADRANT_SIZE

      template_x = quad_world_x - placement.world_offset_x
      template_y = quad_world_y - placement.world_offset_y

      # Crop this quadrant from the generated image
      crop_box = (
        template_x,
        template_y,
        template_x + QUADRANT_SIZE,
        template_y + QUADRANT_SIZE,
      )
      quad_img = generated_image.crop(crop_box)
      png_bytes = image_to_png_bytes(quad_img)

      # Only save primary quadrants (not padding)
      if (qx, qy) in primary_quadrants or (qx, qy) in [
        (q[0], q[1]) for q in primary_quadrants
      ]:
        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for ({qx}, {qy})")
          saved_count += 1
        else:
          print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
      else:
        print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

    update_status("complete", f"Generated {saved_count} quadrant(s)")
    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": list(primary_quadrants),
    }

  finally:
    # Clean up temp file
    template_path.unlink(missing_ok=True)

```

```python
"""
Reusable library for generating pixel art using the Oxen.ai model.

This module provides the core generation logic that can be used by:
- view_generations.py (Flask web server)
- generate_tiles_omni.py (command-line script)
- automatic_generation.py (automated generation)

The main entry point is `run_generation_for_quadrants()` which handles:
1. Validating the quadrant selection
2. Rendering any missing quadrants
3. Building the template image
4. Uploading to GCS and calling the Oxen API
5. Saving the generated quadrants to the database
"""

import os
import re
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  ensure_quadrant_exists,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  upload_to_gcs,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation as shared_get_quadrant_generation,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_render as shared_get_quadrant_render,
)

# Load environment variables
load_dotenv()

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
OMNI_WATER_MODEL_ID = "cannoneyed-quiet-green-lamprey"
OMNI_WATER_V2_MODEL_ID = "cannoneyed-rural-rose-dingo"

GCS_BUCKET_NAME = "isometric-nyc-infills"


# =============================================================================
# Quadrant Parsing Utilities
# =============================================================================


def parse_quadrant_tuple(s: str) -> tuple[int, int]:
  """
  Parse a quadrant tuple string like "(0,1)" or "0,1" into a tuple.

  Args:
      s: String in format "(x,y)" or "x,y"

  Returns:
      Tuple of (x, y) coordinates

  Raises:
      ValueError: If the format is invalid
  """
  s = s.strip()
  # Remove optional parentheses
  if s.startswith("(") and s.endswith(")"):
    s = s[1:-1]
  parts = s.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid quadrant tuple format: {s}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def parse_quadrant_list(s: str) -> list[tuple[int, int]]:
  """
  Parse a comma-separated list of quadrant tuples.

  Args:
      s: String like "(0,1),(0,2)" or "(0,1), (0,2)"

  Returns:
      List of (x, y) coordinate tuples

  Raises:
      ValueError: If the format is invalid
  """
  # Use regex to find all (x,y) patterns
  pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
  matches = re.findall(pattern, s)
  if not matches:
    raise ValueError(f"No valid quadrant tuples found in: {s}")
  return [(int(x), int(y)) for x, y in matches]


# =============================================================================
# Oxen API Functions
# =============================================================================


def call_oxen_api(
  image_url: str,
  model_config: "ModelConfig | None" = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image
      model_config: Optional model configuration. If not provided, uses defaults.

  Returns:
      URL of the generated image

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  from isometric_nyc.e2e_generation.model_config import ModelConfig

  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    model_id = model_config.model_id
    api_key = model_config.api_key
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "https://hub.oxen.ai/api/images/edit"
    model_id = OMNI_WATER_MODEL_ID
    api_key = os.getenv("OXEN_OMNI_v04_WATER_API_KEY")
    num_inference_steps = 28

  if not api_key:
    raise ValueError(f"API key not found for model {model_id}")

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  prompt = (
    "Fill in the outlined section with the missing pixels corresponding to "
    "the <isometric nyc pixel art> style, removing the border and exactly "
    "following the shape/style/structure of the surrounding image (if present)."
  )

  payload = {
    "model": model_id,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": num_inference_steps,
  }

  print(f"   🤖 Calling Oxen API with model {model_id}...")
  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()

  if "images" in result and len(result["images"]) > 0:
    return result["images"][0]["url"]
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  else:
    raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> Image.Image:
  """
  Download an image from a URL and return as PIL Image.

  Args:
      url: URL of the image to download

  Returns:
      PIL Image object
  """
  response = requests.get(url, timeout=120)
  response.raise_for_status()
  return Image.open(BytesIO(response.content))


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
      conn: Database connection
      config: Generation config dict
      x: Quadrant x coordinate
      y: Quadrant y coordinate
      port: Web server port for rendering

  Returns:
      PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  # Build URL for rendering
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  # Render using Playwright
  with sync_playwright() as p:
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

    context = browser.new_context(
      viewport={"width": config["width_px"], "height": config["height_px"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    page.goto(url, wait_until="networkidle")

    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
    except Exception:
      print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_generation_for_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = GCS_BUCKET_NAME,
  status_callback: Callable[[str, str], None] | None = None,
  model_config: "ModelConfig | None" = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Uploads to GCS and calls the Oxen API
  5. Saves the generated quadrants to the database

  Args:
      conn: Database connection
      config: Generation config dict
      selected_quadrants: List of (x, y) quadrant coordinates to generate
      port: Web server port for rendering (default: 5173)
      bucket_name: GCS bucket name for uploads
      status_callback: Optional callback(status, message) for progress updates
      model_config: Optional model configuration for the Oxen API

  Returns:
      Dict with:
          - success: bool
          - message: str (on success)
          - error: str (on failure)
          - quadrants: list of generated quadrant coords (on success)
  """
  from isometric_nyc.e2e_generation.model_config import ModelConfig

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = shared_get_quadrant_generation(conn, qx, qy)
    return gen is not None

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)
    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result

  # Save template to temp file and upload to GCS
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = Path(tmp.name)
    template_image.save(template_path)

  try:
    update_status("uploading", "Uploading template to cloud...")
    print("📤 Uploading template to GCS...")
    image_url = upload_to_gcs(template_path, bucket_name)

    update_status("generating", "Calling AI model (this may take a minute)...")
    print("🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url)

    update_status("saving", "Downloading and saving results...")
    print("📥 Downloading generated image...")
    generated_image = download_image_to_pil(generated_url)

    # Extract quadrants from generated image and save to database
    print("💾 Saving generated quadrants to database...")

    # Figure out what quadrants are in the infill region
    all_infill_quadrants = (
      placement.all_infill_quadrants
      if placement.all_infill_quadrants
      else region.overlapping_quadrants()
    )

    # For each infill quadrant, extract pixels from the generated image
    saved_count = 0
    for qx, qy in all_infill_quadrants:
      # Calculate position in the generated image
      quad_world_x = qx * QUADRANT_SIZE
      quad_world_y = qy * QUADRANT_SIZE

      template_x = quad_world_x - placement.world_offset_x
      template_y = quad_world_y - placement.world_offset_y

      # Crop this quadrant from the generated image
      crop_box = (
        template_x,
        template_y,
        template_x + QUADRANT_SIZE,
        template_y + QUADRANT_SIZE,
      )
      quad_img = generated_image.crop(crop_box)
      png_bytes = image_to_png_bytes(quad_img)

      # Only save primary quadrants (not padding)
      if (qx, qy) in primary_quadrants or (qx, qy) in [
        (q[0], q[1]) for q in primary_quadrants
      ]:
        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for ({qx}, {qy})")
          saved_count += 1
        else:
          print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
      else:
        print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

    update_status("complete", f"Generated {saved_count} quadrant(s)")
    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": list(primary_quadrants),
    }

  finally:
    # Clean up temp file
    template_path.unlink(missing_ok=True)

```

```python
"""
Reusable library for generating pixel art using the Oxen.ai model.

This module provides the core generation logic that can be used by:
- view_generations.py (Flask web server)
- generate_tiles_omni.py (command-line script)
- automatic_generation.py (automated generation)

The main entry point is `run_generation_for_quadrants()` which handles:
1. Validating the quadrant selection
2. Rendering any missing quadrants
3. Building the template image
4. Uploading to GCS and calling the Oxen API
5. Saving the generated quadrants to the database
"""

import os
import re
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  ensure_quadrant_exists,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  upload_to_gcs,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation as shared_get_quadrant_generation,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_render as shared_get_quadrant_render,
)

# Load environment variables
load_dotenv()

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
OMNI_WATER_MODEL_ID = "cannoneyed-quiet-green-lamprey"
OMNI_WATER_V2_MODEL_ID = "cannoneyed-rural-rose-dingo"

GCS_BUCKET_NAME = "isometric-nyc-infills"


# =============================================================================
# Quadrant Parsing Utilities
# =============================================================================


def parse_quadrant_tuple(s: str) -> tuple[int, int]:
  """
  Parse a quadrant tuple string like "(0,1)" or "0,1" into a tuple.

  Args:
      s: String in format "(x,y)" or "x,y"

  Returns:
      Tuple of (x, y) coordinates

  Raises:
      ValueError: If the format is invalid
  """
  s = s.strip()
  # Remove optional parentheses
  if s.startswith("(") and s.endswith(")"):
    s = s[1:-1]
  parts = s.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid quadrant tuple format: {s}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def parse_quadrant_list(s: str) -> list[tuple[int, int]]:
  """
  Parse a comma-separated list of quadrant tuples.

  Args:
      s: String like "(0,1),(0,2)" or "(0,1), (0,2)"

  Returns:
      List of (x, y) coordinate tuples

  Raises:
      ValueError: If the format is invalid
  """
  # Use regex to find all (x,y) patterns
  pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
  matches = re.findall(pattern, s)
  if not matches:
    raise ValueError(f"No valid quadrant tuples found in: {s}")
  return [(int(x), int(y)) for x, y in matches]


# =============================================================================
# Oxen API Functions
# =============================================================================


def call_oxen_api(
  image_url: str,
  model_config: "ModelConfig | None" = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image
      model_config: Optional model configuration. If not provided, uses defaults.

  Returns:
      URL of the generated image

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  from isometric_nyc.e2e_generation.model_config import ModelConfig

  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    model_id = model_config.model_id
    api_key = model_config.api_key
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "https://hub.oxen.ai/api/images/edit"
    model_id = OMNI_WATER_MODEL_ID
    api_key = os.getenv("OXEN_OMNI_v04_WATER_API_KEY")
    num_inference_steps = 28

  if not api_key:
    raise ValueError(f"API key not found for model {model_id}")

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  prompt = (
    "Fill in the outlined section with the missing pixels corresponding to "
    "the <isometric nyc pixel art> style, removing the border and exactly "
    "following the shape/style/structure of the surrounding image (if present)."
  )

  payload = {
    "model": model_id,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": num_inference_steps,
  }

  print(f"   🤖 Calling Oxen API with model {model_id}...")
  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()

  if "images" in result and len(result["images"]) > 0:
    return result["images"][0]["url"]
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  else:
    raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> Image.Image:
  """
  Download an image from a URL and return as PIL Image.

  Args:
      url: URL of the image to download

  Returns:
      PIL Image object
  """
  response = requests.get(url, timeout=120)
  response.raise_for_status()
  return Image.open(BytesIO(response.content))


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
      conn: Database connection
      config: Generation config dict
      x: Quadrant x coordinate
      y: Quadrant y coordinate
      port: Web server port for rendering

  Returns:
      PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  # Build URL for rendering
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  # Render using Playwright
  with sync_playwright() as p:
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

    context = browser.new_context(
      viewport={"width": config["width_px"], "height": config["height_px"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    page.goto(url, wait_until="networkidle")

    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
    except Exception:
      print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_generation_for_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = GCS_BUCKET_NAME,
  status_callback: Callable[[str, str], None] | None = None,
  model_config: "ModelConfig | None" = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Uploads to GCS and calls the Oxen API
  5. Saves the generated quadrants to the database

  Args:
      conn: Database connection
      config: Generation config dict
      selected_quadrants: List of (x, y) quadrant coordinates to generate
      port: Web server port for rendering (default: 5173)
      bucket_name: GCS bucket name for uploads
      status_callback: Optional callback(status, message) for progress updates
      model_config: Optional model configuration for the Oxen API

  Returns:
      Dict with:
          - success: bool
          - message: str (on success)
          - error: str (on failure)
          - quadrants: list of generated quadrant coords (on success)
  """
  from isometric_nyc.e2e_generation.model_config import ModelConfig

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = shared_get_quadrant_generation(conn, qx, qy)
    return gen is not None

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)
    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result

  # Save template to temp file and upload to GCS
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = Path(tmp.name)
    template_image.save(template_path)

  try:
    update_status("uploading", "Uploading template to cloud...")
    print("📤 Uploading template to GCS...")
    image_url = upload_to_gcs(template_path, bucket_name)

    update_status("generating", "Calling AI model (this may take a minute)...")
    print("🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url, model_config)

    update_status("saving", "Downloading and saving results...")
    print("📥 Downloading generated image...")
    generated_image = download_image_to_pil(generated_url)

    # Extract quadrants from generated image and save to database
    print("💾 Saving generated quadrants to database...")

    # Figure out what quadrants are in the infill region
    all_infill_quadrants = (
      placement.all_infill_quadrants
      if placement.all_infill_quadrants
      else region.overlapping_quadrants()
    )

    # For each infill quadrant, extract pixels from the generated image
    saved_count = 0
    for qx, qy in all_infill_quadrants:
      # Calculate position in the generated image
      quad_world_x = qx * QUADRANT_SIZE
      quad_world_y = qy * QUADRANT_SIZE

      template_x = quad_world_x - placement.world_offset_x
      template_y = quad_world_y - placement.world_offset_y

      # Crop this quadrant from the generated image
      crop_box = (
        template_x,
        template_y,
        template_x + QUADRANT_SIZE,
        template_y + QUADRANT_SIZE,
      )
      quad_img = generated_image.crop(crop_box)
      png_bytes = image_to_png_bytes(quad_img)

      # Only save primary quadrants (not padding)
      if (qx, qy) in primary_quadrants or (qx, qy) in [
        (q[0], q[1]) for q in primary_quadrants
      ]:
        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for ({qx}, {qy})")
          saved_count += 1
        else:
          print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
      else:
        print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

    update_status("complete", f"Generated {saved_count} quadrant(s)")
    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": list(primary_quadrants),
    }

  finally:
    # Clean up temp file
    template_path.unlink(missing_ok=True)

```

Now let me update the frontend. First, let me update the HTML template:

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

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

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
            <label class="model-label">Model:
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

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

Now let me update the JavaScript to remove client-side queue state and add model selector:

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

// Apply locked/queued styles based on server status
function applyStatusStyles(status) {
  // Clear all existing locked/queued styles first
  document.querySelectorAll(".tile.locked, .tile.queued").forEach((tile) => {
    tile.classList.remove("locked", "queued");
  });

  // Apply locked style to currently processing quadrants
  if (status.is_generating && status.quadrants && status.quadrants.length > 0) {
    document.body.classList.add("generating");
    status.quadrants.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });
  } else {
    document.body.classList.remove("generating");
  }

  // Apply queued style to pending queue items
  if (status.queue && status.queue.length > 0) {
    status.queue.forEach((item) => {
      if (item.quadrants) {
        item.quadrants.forEach(([qx, qy]) => {
          const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
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

  let statusParts = [];

  // Show current processing status from server
  if (serverStatus) {
    if (serverStatus.is_generating && serverStatus.quadrants && serverStatus.quadrants.length > 0) {
      const action = serverStatus.status === "rendering" ? "Rendering" : "Generating";
      const coordsStr = serverStatus.quadrants.map(([x, y]) => `(${x},${y})`).join(" ");
      statusParts.push(`${action} ${coordsStr}`);
    }

    // Show queue count
    if (serverStatus.queue_length > 0) {
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
  const isProcessing = serverStatus && (serverStatus.is_generating || serverStatus.queue_length > 0);
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

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  const modelId = getSelectedModelId();

  console.log("Generate requested for:", coords, "with model:", modelId);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Show immediate feedback
  showToast(
    "info",
    "Adding to queue...",
    `Queueing ${coords.length} quadrant(s) for generation`
  );

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        quadrants: coords,
        model_id: modelId,
      }),
    });

    const result = await response.json();

    if (result.queued) {
      console.log("Generation queued at position:", result.position);
      showToast(
        "success",
        "Added to queue",
        `Generation queued at position ${result.position}.`
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
  if (tileEl.classList.contains("locked") || tileEl.classList.contains("queued")) {
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

    // Update buttons based on state
    const generateBtn = document.getElementById("generateBtn");
    const renderBtn = document.getElementById("renderBtn");

    if (nowProcessing) {
      // Show loading state on appropriate button
      if (isRendering) {
        renderBtn.classList.add("loading");
        renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
      } else {
        generateBtn.classList.add("loading");
        generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
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
      const loadingToast = document.querySelector(".toast.loading .toast-message");
      if (loadingToast && status.message) {
        loadingToast.textContent = status.message;
      }
    } else {
      // Reset buttons
      generateBtn.classList.remove("loading");
      generateBtn.innerHTML = "Generate";
      renderBtn.classList.remove("loading");
      renderBtn.innerHTML = "Render";
    }

    // Handle status transitions
    if (status.status === "complete" && wasGenerating && !nowProcessing) {
      clearLoadingToasts();
      showToast("success", "Complete!", status.message);

      // Check if there are more items in queue
      if (status.queue_length > 0) {
        showToast(
          "info",
          "Processing queue",
          `${status.queue_length} more item(s) in queue...`
        );
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
    } else if (status.status === "idle" && status.queue_length === 0 && !nowProcessing) {
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
      console.log("Processing in progress or queue non-empty, starting polling...");
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

Now let me add CSS styles for the model selector:

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

```

Now let me check for any linting errors in the Python files:

Let me run the linter to make sure everything is formatted correctly:

Let me fix these linting errors:

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
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cleanup_stale_processing,
  clear_completed_items,
  get_next_pending_item,
  get_pending_queue,
  get_queue_status,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
)
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
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

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


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

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

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


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

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
      SELECT item_type, quadrants, model_id
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

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting" if item_type == QueueItemType.GENERATE else "rendering"
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}")
    if model_id:
      print(f"   Model: {model_id}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      result = run_generation(conn, config, selected_quadrants, model_id)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        mark_item_complete(conn, item_id, result["message"])
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        mark_item_error(conn, item_id, result["error"])
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


def queue_worker():
  """Background worker that processes the generation queue from the database."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    conn = None
    try:
      conn = get_db_connection()

      # Get next pending item
      item = get_next_pending_item(conn)

      if item is None:
        # No items in queue, wait a bit and check again
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      conn.close()
      conn = None

      # Acquire the generation lock and process the item
      with generation_lock:
        try:
          process_queue_item_from_db(item_id)
        finally:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None

      # Small delay between items
      time.sleep(0.5)

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
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
    )

    # Get queue position
    pending = get_pending_queue(conn)
    queue_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": queue_position,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {queue_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
    }

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify({
    "models": [m.to_dict() for m in APP_CONFIG.models],
    "default_model_id": APP_CONFIG.default_model_id,
  })


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
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
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
  """API endpoint to generate tiles for selected quadrants."""
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
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "generate", model_id)
  return jsonify(result), 202  # 202 Accepted


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG

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

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

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
    default_marker = " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Clean up any stale processing items from previous runs
    stale_count = cleanup_stale_processing(conn)
    if stale_count > 0:
      print(f"⚠️  Reset {stale_count} stale processing item(s) from previous run")
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

```python
"""
Reusable library for generating pixel art using the Oxen.ai model.

This module provides the core generation logic that can be used by:
- view_generations.py (Flask web server)
- generate_tiles_omni.py (command-line script)
- automatic_generation.py (automated generation)

The main entry point is `run_generation_for_quadrants()` which handles:
1. Validating the quadrant selection
2. Rendering any missing quadrants
3. Building the template image
4. Uploading to GCS and calling the Oxen API
5. Saving the generated quadrants to the database
"""

import os
import re
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  ensure_quadrant_exists,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  upload_to_gcs,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation as shared_get_quadrant_generation,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_render as shared_get_quadrant_render,
)

# Load environment variables
load_dotenv()

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
OMNI_WATER_MODEL_ID = "cannoneyed-quiet-green-lamprey"
OMNI_WATER_V2_MODEL_ID = "cannoneyed-rural-rose-dingo"

GCS_BUCKET_NAME = "isometric-nyc-infills"


# =============================================================================
# Quadrant Parsing Utilities
# =============================================================================


def parse_quadrant_tuple(s: str) -> tuple[int, int]:
  """
  Parse a quadrant tuple string like "(0,1)" or "0,1" into a tuple.

  Args:
      s: String in format "(x,y)" or "x,y"

  Returns:
      Tuple of (x, y) coordinates

  Raises:
      ValueError: If the format is invalid
  """
  s = s.strip()
  # Remove optional parentheses
  if s.startswith("(") and s.endswith(")"):
    s = s[1:-1]
  parts = s.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid quadrant tuple format: {s}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def parse_quadrant_list(s: str) -> list[tuple[int, int]]:
  """
  Parse a comma-separated list of quadrant tuples.

  Args:
      s: String like "(0,1),(0,2)" or "(0,1), (0,2)"

  Returns:
      List of (x, y) coordinate tuples

  Raises:
      ValueError: If the format is invalid
  """
  # Use regex to find all (x,y) patterns
  pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
  matches = re.findall(pattern, s)
  if not matches:
    raise ValueError(f"No valid quadrant tuples found in: {s}")
  return [(int(x), int(y)) for x, y in matches]


# =============================================================================
# Oxen API Functions
# =============================================================================


def call_oxen_api(
  image_url: str,
  model_config: "ModelConfig | None" = None,  # noqa: F821
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image
      model_config: Optional model configuration (ModelConfig from model_config.py).
        If not provided, uses defaults.

  Returns:
      URL of the generated image

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    model_id = model_config.model_id
    api_key = model_config.api_key
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "https://hub.oxen.ai/api/images/edit"
    model_id = OMNI_WATER_MODEL_ID
    api_key = os.getenv("OXEN_OMNI_v04_WATER_API_KEY")
    num_inference_steps = 28

  if not api_key:
    raise ValueError(f"API key not found for model {model_id}")

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  prompt = (
    "Fill in the outlined section with the missing pixels corresponding to "
    "the <isometric nyc pixel art> style, removing the border and exactly "
    "following the shape/style/structure of the surrounding image (if present)."
  )

  payload = {
    "model": model_id,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": num_inference_steps,
  }

  print(f"   🤖 Calling Oxen API with model {model_id}...")
  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()

  if "images" in result and len(result["images"]) > 0:
    return result["images"][0]["url"]
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  else:
    raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> Image.Image:
  """
  Download an image from a URL and return as PIL Image.

  Args:
      url: URL of the image to download

  Returns:
      PIL Image object
  """
  response = requests.get(url, timeout=120)
  response.raise_for_status()
  return Image.open(BytesIO(response.content))


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
      conn: Database connection
      config: Generation config dict
      x: Quadrant x coordinate
      y: Quadrant y coordinate
      port: Web server port for rendering

  Returns:
      PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  # Build URL for rendering
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  # Render using Playwright
  with sync_playwright() as p:
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

    context = browser.new_context(
      viewport={"width": config["width_px"], "height": config["height_px"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    page.goto(url, wait_until="networkidle")

    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
    except Exception:
      print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_generation_for_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = GCS_BUCKET_NAME,
  status_callback: Callable[[str, str], None] | None = None,
  model_config: "ModelConfig | None" = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Uploads to GCS and calls the Oxen API
  5. Saves the generated quadrants to the database

  Args:
      conn: Database connection
      config: Generation config dict
      selected_quadrants: List of (x, y) quadrant coordinates to generate
      port: Web server port for rendering (default: 5173)
      bucket_name: GCS bucket name for uploads
      status_callback: Optional callback(status, message) for progress updates
      model_config: Optional model configuration for the Oxen API

  Returns:
      Dict with:
          - success: bool
          - message: str (on success)
          - error: str (on failure)
          - quadrants: list of generated quadrant coords (on success)
  """
  from isometric_nyc.e2e_generation.model_config import ModelConfig

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = shared_get_quadrant_generation(conn, qx, qy)
    return gen is not None

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)
    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result

  # Save template to temp file and upload to GCS
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = Path(tmp.name)
    template_image.save(template_path)

  try:
    update_status("uploading", "Uploading template to cloud...")
    print("📤 Uploading template to GCS...")
    image_url = upload_to_gcs(template_path, bucket_name)

    update_status("generating", "Calling AI model (this may take a minute)...")
    print("🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url, model_config)

    update_status("saving", "Downloading and saving results...")
    print("📥 Downloading generated image...")
    generated_image = download_image_to_pil(generated_url)

    # Extract quadrants from generated image and save to database
    print("💾 Saving generated quadrants to database...")

    # Figure out what quadrants are in the infill region
    all_infill_quadrants = (
      placement.all_infill_quadrants
      if placement.all_infill_quadrants
      else region.overlapping_quadrants()
    )

    # For each infill quadrant, extract pixels from the generated image
    saved_count = 0
    for qx, qy in all_infill_quadrants:
      # Calculate position in the generated image
      quad_world_x = qx * QUADRANT_SIZE
      quad_world_y = qy * QUADRANT_SIZE

      template_x = quad_world_x - placement.world_offset_x
      template_y = quad_world_y - placement.world_offset_y

      # Crop this quadrant from the generated image
      crop_box = (
        template_x,
        template_y,
        template_x + QUADRANT_SIZE,
        template_y + QUADRANT_SIZE,
      )
      quad_img = generated_image.crop(crop_box)
      png_bytes = image_to_png_bytes(quad_img)

      # Only save primary quadrants (not padding)
      if (qx, qy) in primary_quadrants or (qx, qy) in [
        (q[0], q[1]) for q in primary_quadrants
      ]:
        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for ({qx}, {qy})")
          saved_count += 1
        else:
          print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
      else:
        print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

    update_status("complete", f"Generated {saved_count} quadrant(s)")
    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": list(primary_quadrants),
    }

  finally:
    # Clean up temp file
    template_path.unlink(missing_ok=True)

```

```python
"""
Reusable library for generating pixel art using the Oxen.ai model.

This module provides the core generation logic that can be used by:
- view_generations.py (Flask web server)
- generate_tiles_omni.py (command-line script)
- automatic_generation.py (automated generation)

The main entry point is `run_generation_for_quadrants()` which handles:
1. Validating the quadrant selection
2. Rendering any missing quadrants
3. Building the template image
4. Uploading to GCS and calling the Oxen API
5. Saving the generated quadrants to the database
"""

import os
import re
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  ensure_quadrant_exists,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  upload_to_gcs,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation as shared_get_quadrant_generation,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_render as shared_get_quadrant_render,
)

# Load environment variables
load_dotenv()

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
OMNI_WATER_MODEL_ID = "cannoneyed-quiet-green-lamprey"
OMNI_WATER_V2_MODEL_ID = "cannoneyed-rural-rose-dingo"

GCS_BUCKET_NAME = "isometric-nyc-infills"


# =============================================================================
# Quadrant Parsing Utilities
# =============================================================================


def parse_quadrant_tuple(s: str) -> tuple[int, int]:
  """
  Parse a quadrant tuple string like "(0,1)" or "0,1" into a tuple.

  Args:
      s: String in format "(x,y)" or "x,y"

  Returns:
      Tuple of (x, y) coordinates

  Raises:
      ValueError: If the format is invalid
  """
  s = s.strip()
  # Remove optional parentheses
  if s.startswith("(") and s.endswith(")"):
    s = s[1:-1]
  parts = s.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid quadrant tuple format: {s}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def parse_quadrant_list(s: str) -> list[tuple[int, int]]:
  """
  Parse a comma-separated list of quadrant tuples.

  Args:
      s: String like "(0,1),(0,2)" or "(0,1), (0,2)"

  Returns:
      List of (x, y) coordinate tuples

  Raises:
      ValueError: If the format is invalid
  """
  # Use regex to find all (x,y) patterns
  pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
  matches = re.findall(pattern, s)
  if not matches:
    raise ValueError(f"No valid quadrant tuples found in: {s}")
  return [(int(x), int(y)) for x, y in matches]


# =============================================================================
# Oxen API Functions
# =============================================================================


def call_oxen_api(
  image_url: str,
  model_config: "ModelConfig | None" = None,  # noqa: F821
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image
      model_config: Optional model configuration (ModelConfig from model_config.py).
        If not provided, uses defaults.

  Returns:
      URL of the generated image

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    model_id = model_config.model_id
    api_key = model_config.api_key
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "https://hub.oxen.ai/api/images/edit"
    model_id = OMNI_WATER_MODEL_ID
    api_key = os.getenv("OXEN_OMNI_v04_WATER_API_KEY")
    num_inference_steps = 28

  if not api_key:
    raise ValueError(f"API key not found for model {model_id}")

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  prompt = (
    "Fill in the outlined section with the missing pixels corresponding to "
    "the <isometric nyc pixel art> style, removing the border and exactly "
    "following the shape/style/structure of the surrounding image (if present)."
  )

  payload = {
    "model": model_id,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": num_inference_steps,
  }

  print(f"   🤖 Calling Oxen API with model {model_id}...")
  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()

  if "images" in result and len(result["images"]) > 0:
    return result["images"][0]["url"]
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  else:
    raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> Image.Image:
  """
  Download an image from a URL and return as PIL Image.

  Args:
      url: URL of the image to download

  Returns:
      PIL Image object
  """
  response = requests.get(url, timeout=120)
  response.raise_for_status()
  return Image.open(BytesIO(response.content))


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
      conn: Database connection
      config: Generation config dict
      x: Quadrant x coordinate
      y: Quadrant y coordinate
      port: Web server port for rendering

  Returns:
      PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  # Build URL for rendering
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  # Render using Playwright
  with sync_playwright() as p:
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

    context = browser.new_context(
      viewport={"width": config["width_px"], "height": config["height_px"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    page.goto(url, wait_until="networkidle")

    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
    except Exception:
      print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_generation_for_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = GCS_BUCKET_NAME,
  status_callback: Callable[[str, str], None] | None = None,
  model_config: "ModelConfig | None" = None,  # noqa: F821
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Uploads to GCS and calls the Oxen API
  5. Saves the generated quadrants to the database

  Args:
      conn: Database connection
      config: Generation config dict
      selected_quadrants: List of (x, y) quadrant coordinates to generate
      port: Web server port for rendering (default: 5173)
      bucket_name: GCS bucket name for uploads
      status_callback: Optional callback(status, message) for progress updates
      model_config: Optional model configuration for the Oxen API (ModelConfig from model_config.py)

  Returns:
      Dict with:
          - success: bool
          - message: str (on success)
          - error: str (on failure)
          - quadrants: list of generated quadrant coords (on success)
  """

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = shared_get_quadrant_generation(conn, qx, qy)
    return gen is not None

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)
    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result

  # Save template to temp file and upload to GCS
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = Path(tmp.name)
    template_image.save(template_path)

  try:
    update_status("uploading", "Uploading template to cloud...")
    print("📤 Uploading template to GCS...")
    image_url = upload_to_gcs(template_path, bucket_name)

    update_status("generating", "Calling AI model (this may take a minute)...")
    print("🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url, model_config)

    update_status("saving", "Downloading and saving results...")
    print("📥 Downloading generated image...")
    generated_image = download_image_to_pil(generated_url)

    # Extract quadrants from generated image and save to database
    print("💾 Saving generated quadrants to database...")

    # Figure out what quadrants are in the infill region
    all_infill_quadrants = (
      placement.all_infill_quadrants
      if placement.all_infill_quadrants
      else region.overlapping_quadrants()
    )

    # For each infill quadrant, extract pixels from the generated image
    saved_count = 0
    for qx, qy in all_infill_quadrants:
      # Calculate position in the generated image
      quad_world_x = qx * QUADRANT_SIZE
      quad_world_y = qy * QUADRANT_SIZE

      template_x = quad_world_x - placement.world_offset_x
      template_y = quad_world_y - placement.world_offset_y

      # Crop this quadrant from the generated image
      crop_box = (
        template_x,
        template_y,
        template_x + QUADRANT_SIZE,
        template_y + QUADRANT_SIZE,
      )
      quad_img = generated_image.crop(crop_box)
      png_bytes = image_to_png_bytes(quad_img)

      # Only save primary quadrants (not padding)
      if (qx, qy) in primary_quadrants or (qx, qy) in [
        (q[0], q[1]) for q in primary_quadrants
      ]:
        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for ({qx}, {qy})")
          saved_count += 1
        else:
          print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
      else:
        print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

    update_status("complete", f"Generated {saved_count} quadrant(s)")
    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": list(primary_quadrants),
    }

  finally:
    # Clean up temp file
    template_path.unlink(missing_ok=True)

```

Now let me run the tests to make sure nothing is broken:

All tests pass. Let me update the task file to mark it as completed and provide a summary:

```markdown
# Generation App Improvements

**Status: ✅ COMPLETED**

## 1. Ensure queue robustness ✅

We want to make sure that the current queue system for generations is robust.
Rather than keeping the queue completely in memory, let's create a generations
db table that keeps track of an ordered queue of the generations to be
processed.

Let's also make sure that the generation queue is maintained completely server
side, that is - there's no local storage or client side memory of checking the
queue.

We can also add generation plans to the queue from other sources/scripts.

**Implementation:**
- Created `queue_db.py` with a SQLite-backed queue (`generation_queue` table)
- Queue supports pending, processing, complete, and error states
- Includes automatic cleanup of stale processing items on restart
- Frontend now polls server for queue status instead of using localStorage

## 2. Multiple models ✅

When we boot the app up, we shoul read from an `app_config.json` what model or
models to use. Each model will have a `name`, `model_id` and an `api_key` param,
and that model will be used for generation - all queue entries should also
contain a `model_id` so that we can use that model for generation.

The frontend needs to have a dropdown select added to the controls that lets the
user pick which model is to be used for generation. The model `name` \ `ids` in
this list should be bootstrapped into the page when served, and the selected
model id should be sent to the generate request to be added to the generation
queue.

**Implementation:**
- Created `model_config.py` with `ModelConfig` and `AppConfig` dataclasses
- Created `app_config.json` with 3 pre-configured models (Omni Water v1/v2, Omni Original)
- Model selector dropdown added to the toolbar
- Queue entries now include `model_id` field
- `generate_omni.py` updated to accept model configuration

## 3. Persisted web renderer ✅

Since every single generation request needs to create a new node/web server to
render the web view of tile, we should have one single web renderer process
that's started when the app server loads - the app server should make requests
to this web_renderer subprocess through a simple interface, and this interface
must account for the fact that multiple requests may come in at onece, so it
needs a queue manager for handling those requests to render specific tiles in
the order they're received.

**Implementation:**
- Created `web_renderer.py` with `WebRenderer` class
- Singleton pattern for global renderer instance
- Internal queue for render requests with callback-based results
- Auto-restarts web server if it crashes
- Started automatically when app server boots

```

I've completed implementing all three improvements from the task. Here's a summary of what was created and modified:

## Summary of Changes

### 1. Database-Backed Queue (`queue_db.py`)
- Created a new `generation_queue` table in the SQLite database
- Supports pending, processing, complete, and error states
- Includes automatic cleanup of stale items on server restart
- Queue entries now include `model_id` for multi-model support

### 2. Multi-Model Support
**New files:**
- `model_config.py` - Defines `ModelConfig` and `AppConfig` classes
- `app_config.json` - Configuration file with 3 pre-configured models:
  - Omni Water v1 (default)
  - Omni Water v2
  - Omni (Original)

**Changes to `generate_omni.py`:**
- Updated `call_oxen_api()` to accept a `model_config` parameter
- Updated `run_generation_for_quadrants()` to pass model config to the API

### 3. Persisted Web Renderer (`web_renderer.py`)
- Created `WebRenderer` class with internal render queue
- Singleton pattern via `get_web_renderer()` function
- Auto-restarts web server if it crashes
- Started automatically when the app server boots

### 4. Frontend Updates
**Template (`viewer.html`):**
- Added model selector dropdown in the toolbar
- Models are bootstrapped into the page config on load

**JavaScript (`viewer.js`):**
- Removed all localStorage-based queue tracking
- Now relies entirely on server-side queue status
- Added model selector initialization and selection

**CSS (`viewer.css`):**
- Added styles for the model selector dropdown

### 5. App Server Updates (`app.py`)
- Integrated the database-backed queue system
- Starts the global web renderer on boot
- All API endpoints now use the database queue
- Loads and exposes model configuration to the frontend

