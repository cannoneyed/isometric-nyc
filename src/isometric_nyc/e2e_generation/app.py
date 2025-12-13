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
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

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
from isometric_nyc.e2e_generation.shared import DEFAULT_WEB_PORT, get_generation_config
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
  """Get info about a quadrant including whether it has data and is flagged."""
  conn = get_db_connection()
  try:
    # Ensure flagged column exists
    ensure_flagged_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      return {"has_data": bool(row[0]), "flagged": bool(row[1])}
    return {"has_data": False, "flagged": False}
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

  # Check which tiles have data and flagged status
  tiles = {}
  flagged_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]

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


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
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
    context_quadrants=context_quadrants,
    prompt=prompt,
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
      SELECT item_type, quadrants, model_id, context_quadrants, prompt
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
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      result = run_generation(
        conn, config, selected_quadrants, model_id, context_quadrants, prompt
      )
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
  """
  global generation_state, queue_worker_running, generation_cancelled

  print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
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

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt
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

  # Load existing generated quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated)

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
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

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
