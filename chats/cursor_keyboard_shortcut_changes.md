# Keyboard shortcut changes
_Exported on 12/12/2025 at 07:11:50 EST from Cursor (2.1.39)_

---

**User**

Can you change the keyboard shortcut in @src/isometric_nyc/e2e_generation/app.py @src/isometric_nyc/e2e_generation/static/viewer.js  "G" to trigger the "generate" button functionality, and switch the render/generated view toggle key command to "R"?

---

**Cursor**

```javascript
case "g":
case "G":
  document.getElementById("showRender").click();
  break;
```

```plaintext
Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
  S          - Toggle select tool
```

I'll update both files to swap the keyboard shortcuts - "G" will trigger generate and "R" will toggle the render/generation view.

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

```

```javascript
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
    case "r":
    case "R":
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

```

Done! I've updated both files:

**`app.py`** - Updated the docstring to document the new shortcuts:
- `R` - Toggle render/generation mode  
- `G` - Generate selected quadrants

**`viewer.js`** - Swapped the keyboard handlers:
- `R` now clicks the "showRender" checkbox (toggle render/generation view)
- `G` now calls `generateSelected()` to trigger generation for selected quadrants

---

**User**

Can you also edit the top status bar to not show all the enqueued generations, just the one being generated and the count of how many are enqueued?

---

**Cursor**

```javascript
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

I see the problem is that the current implementation is displaying detailed coordinate information for queued items, which clutters the status bar. To simplify the view, I'll modify the queue display to only show the number of queued items without listing their specific coordinates.

By changing the code to just push the queue length, the status bar will become more concise and readable. This approach focuses on providing a quick overview of the queue state without overwhelming the user with granular details.
```

I'll update the `updateSelectionStatus` function in `viewer.js` to show just the count of queued items instead of listing all their coordinates.

```javascript
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
    case "r":
    case "R":
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
    statusParts.push(`📋 Queued: ${queued.length}`);
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

```

Done! The status bar will now show a cleaner format like:

**Before:** `Generating (0,0) • 📋 Queue: 3 - (1,0) (1,1) (2,0)`

**After:** `Generating (0,0) • 📋 Queued: 3`

The currently generating quadrant coordinates are still shown, but the queued items now only show the count.

