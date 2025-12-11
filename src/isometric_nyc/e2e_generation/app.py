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
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import hashlib
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
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
}

# Generation queue for multiple requests
generation_queue: list[
  dict
] = []  # List of {"quadrants": [...], "type": "generate"|"render"}
queue_lock = threading.Lock()
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None


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
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Ensure web server is running before generation
  ensure_web_server_running()

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
  )


def process_queue_item(item: dict) -> dict:
  """Process a single queue item (generate or render)."""
  global generation_state

  quadrants = item["quadrants"]
  item_type = item["type"]

  # Convert to list of tuples
  selected_quadrants = [(q[0], q[1]) for q in quadrants]

  # Initialize generation state
  generation_state["is_generating"] = True
  generation_state["quadrants"] = selected_quadrants
  generation_state["status"] = "starting" if item_type == "generate" else "rendering"
  generation_state["message"] = f"Starting {item_type}..."
  generation_state["error"] = None
  generation_state["started_at"] = time.time()

  print(f"\n{'=' * 60}")
  print(
    f"{'🎯' if item_type == 'generate' else '🎨'} {item_type.title()} request from queue: {selected_quadrants}"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    if item_type == "generate":
      result = run_generation(conn, config, selected_quadrants)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
      return result
    else:  # render
      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def queue_worker():
  """Background worker that processes the generation queue."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    item = None

    # Get next item from queue
    with queue_lock:
      if generation_queue:
        item = generation_queue.pop(0)

    if item is None:
      # No items in queue, wait a bit and check again
      time.sleep(0.5)
      continue

    # Acquire the generation lock and process the item
    with generation_lock:
      try:
        process_queue_item(item)
      finally:
        generation_state["is_generating"] = False

    # Small delay between items
    time.sleep(0.5)

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


def add_to_queue(quadrants: list[tuple[int, int]], item_type: str) -> dict:
  """Add a generation/render request to the queue."""
  with queue_lock:
    generation_queue.append(
      {
        "quadrants": quadrants,
        "type": item_type,
      }
    )
    queue_position = len(generation_queue)

  # Ensure the queue worker is running
  start_queue_worker()

  return {
    "success": True,
    "queued": True,
    "position": queue_position,
    "message": f"Added to queue at position {queue_position}",
  }


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  with queue_lock:
    queue_info = [
      {"quadrants": item["quadrants"], "type": item["type"]}
      for item in generation_queue
    ]
  return jsonify(
    {
      **generation_state,
      "queue": queue_info,
      "queue_length": len(queue_info),
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

  # Check if already generating/rendering - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "render")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state (reuse for rendering)
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "rendering"
    generation_state["message"] = "Starting render..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎨 Render request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)

      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return jsonify(
        {
          "success": True,
          "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
          "quadrants": selected_quadrants,
        }
      ), 200

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


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

  # Check if already generating - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "generate")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting"
    generation_state["message"] = "Starting generation..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎯 Generation request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)
      result = run_generation(conn, config, selected_quadrants)

      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        return jsonify(result), 200
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        return jsonify(result), 400

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


def ensure_web_server_running() -> None:
  """Ensure the web server for rendering is running."""
  global WEB_SERVER_PROCESS

  if WEB_SERVER_PROCESS is not None:
    # Check if still running
    if WEB_SERVER_PROCESS.poll() is None:
      return  # Still running

  # Start the web server
  print(f"🌐 Starting web server for rendering on port {WEB_SERVER_PORT}...")
  WEB_SERVER_PROCESS = start_web_server(WEB_DIR, WEB_SERVER_PORT)


def main():
  global GENERATION_DIR, WEB_SERVER_PORT

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

    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())
