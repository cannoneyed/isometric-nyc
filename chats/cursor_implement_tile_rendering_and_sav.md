# Implement tile rendering and saving
_Exported on 12/9/2025 at 08:46:29 EST from Cursor (2.1.39)_

---

**User**

OK - let's add a new button to the @src/isometric_nyc/e2e_generation/view_generations.py web app (html/js in @static ) that's called "render" - any tile that's selected should be be web rendered and then saved to the db, with the same toast notifications and locking the generate/render when in progress etc

---

**Cursor**

I'll add a "Render" button to the web app. Let me first read the existing static files and template to understand the current implementation.

Now I understand the structure. I'll add a "Render" button that renders selected tiles and saves them to the database. Let me implement this:

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
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import os
import sqlite3
import tempfile
import threading
import time
import traceback
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
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
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation as shared_get_quadrant_generation,
)
from isometric_nyc.e2e_generation.shared import (
  get_quadrant_render as shared_get_quadrant_render,
)
from isometric_nyc.e2e_generation.shared import (
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  upload_to_gcs,
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

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
GCS_BUCKET_NAME = "isometric-nyc-infills"


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

  return Response(data, mimetype="image/png")


# =============================================================================
# Generation API
# =============================================================================


def call_oxen_api(image_url: str, api_key: str) -> str:
  """Call the Oxen API to generate pixel art."""
  endpoint = "https://hub.oxen.ai/api/images/edit"

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
    "model": OMNI_MODEL_ID,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": 28,
  }

  print(f"   🤖 Calling Oxen API with model {OMNI_MODEL_ID}...")
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
  """Download an image from a URL and return as PIL Image."""
  response = requests.get(url, timeout=120)
  response.raise_for_status()
  return Image.open(BytesIO(response.content))


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  Returns the PNG bytes of the rendered quadrant.
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

  Returns dict with success status and message/error.
  """
  global generation_state

  update_generation_state("validating", "Checking API key...")

  # Check for API key
  api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
  if not api_key:
    update_generation_state("error", error="OXEN_OMNI_v04_API_KEY not set")
    return {
      "success": False,
      "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
    }

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = shared_get_quadrant_generation(conn, qx, qy)
    return gen is not None

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render - make sure web server is running
    update_generation_state("rendering", f"Rendering quadrant ({qx}, {qy})...")
    ensure_web_server_running()
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)
    return None

  update_generation_state("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_generation_state("error", error=msg)
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
  update_generation_state("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_generation_state("error", error=error_msg)
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
    update_generation_state("uploading", "Uploading template to cloud...")
    print("📤 Uploading template to GCS...")
    image_url = upload_to_gcs(template_path, GCS_BUCKET_NAME)

    update_generation_state(
      "generating", "Calling AI model (this may take a minute)..."
    )
    print("🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url, api_key)

    update_generation_state("saving", "Downloading and saving results...")
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
      # The quadrant's world position is (qx * QUADRANT_SIZE, qy * QUADRANT_SIZE)
      # The template's world offset is (placement.world_offset_x, placement.world_offset_y)
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

    update_generation_state("complete", f"Generated {saved_count} quadrant(s)")
    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": primary_quadrants,
    }

  finally:
    # Clean up temp file
    template_path.unlink(missing_ok=True)


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status."""
  return jsonify(generation_state)


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  # Check if already generating
  if generation_state.get("is_generating"):
    return jsonify(
      {
        "success": False,
        "error": "Cannot delete while generation is in progress.",
      }
    )

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


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Check if already generating/rendering
  if not generation_lock.acquire(blocking=False):
    return jsonify(
      {
        "success": False,
        "error": "Operation already in progress. Please wait.",
        "status": generation_state,
      }
    ), 429

  try:
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

  # Check if already generating
  if not generation_lock.acquire(blocking=False):
    return jsonify(
      {
        "success": False,
        "error": "Generation already in progress. Please wait.",
        "status": generation_state,
      }
    ), 429

  try:
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
    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("\n🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())

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
      data-config='{"x": {{ x }}, "y": {{ y }}, "nx": {{ nx }}, "ny": {{ ny }}, "size_px": {{ size_px }}}'
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

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
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

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
    );
    return;
  }

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Set loading state and lock the quadrants
  isRendering = true;
  setLockedQuadrants(coords);
  document.body.classList.add("generating");

  // Mark selected tiles as locked (purple)
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
    tile.classList.add("locked");
  });
  selectedQuadrants.clear();
  updateSelectionStatus();

  const renderBtn = document.getElementById("renderBtn");
  renderBtn.disabled = true;
  renderBtn.classList.add("loading");
  renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

  showToast(
    "loading",
    "Rendering tiles...",
    `Processing ${coords.length} quadrant${
      coords.length > 1 ? "s" : ""
    }. This may take a moment.`
  );

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // If request was accepted, start polling for status
    if (response.status === 429) {
      // Already rendering - start polling
      showToast(
        "info",
        "Render in progress",
        "Reconnected to existing render operation."
      );
      startStatusPolling();
      return;
    }

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

      // Clear selection and refresh after a short delay
      deselectAll();
      setTimeout(() => {
        window.location.reload();
      }, 1500);
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
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

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
    );
    return;
  }

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Set loading state and lock the quadrants
  isRendering = true;
  setLockedQuadrants(coords);
  document.body.classList.add("generating");

  // Mark selected tiles as locked (purple)
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
    tile.classList.add("locked");
  });
  selectedQuadrants.clear();
  updateSelectionStatus();

  const renderBtn = document.getElementById("renderBtn");
  renderBtn.disabled = true;
  renderBtn.classList.add("loading");
  renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

  showToast(
    "loading",
    "Rendering tiles...",
    `Processing ${coords.length} quadrant${
      coords.length > 1 ? "s" : ""
    }. This may take a moment.`
  );

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // If request was accepted, start polling for status
    if (response.status === 429) {
      // Already rendering - start polling
      showToast(
        "info",
        "Render in progress",
        "Reconnected to existing render operation."
      );
      startStatusPolling();
      return;
    }

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

      // Clear selection and refresh after a short delay
      deselectAll();
      setTimeout(() => {
        window.location.reload();
      }, 1500);
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
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

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
    );
    return;
  }

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Set loading state and lock the quadrants
  isRendering = true;
  setLockedQuadrants(coords);
  document.body.classList.add("generating");

  // Mark selected tiles as locked (purple)
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
    tile.classList.add("locked");
  });
  selectedQuadrants.clear();
  updateSelectionStatus();

  const renderBtn = document.getElementById("renderBtn");
  renderBtn.disabled = true;
  renderBtn.classList.add("loading");
  renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

  showToast(
    "loading",
    "Rendering tiles...",
    `Processing ${coords.length} quadrant${
      coords.length > 1 ? "s" : ""
    }. This may take a moment.`
  );

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // If request was accepted, start polling for status
    if (response.status === 429) {
      // Already rendering - start polling
      showToast(
        "info",
        "Render in progress",
        "Reconnected to existing render operation."
      );
      startStatusPolling();
      return;
    }

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

      // Clear selection and refresh after a short delay
      deselectAll();
      setTimeout(() => {
        window.location.reload();
      }, 1500);
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
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Operation finished
      stopStatusPolling();

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);
        setTimeout(() => window.location.reload(), 1500);
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);
        if (isRendering) {
          resetRenderButton();
        } else {
          resetGenerateButton();
        }
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
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

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
    );
    return;
  }

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Set loading state and lock the quadrants
  isRendering = true;
  setLockedQuadrants(coords);
  document.body.classList.add("generating");

  // Mark selected tiles as locked (purple)
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
    tile.classList.add("locked");
  });
  selectedQuadrants.clear();
  updateSelectionStatus();

  const renderBtn = document.getElementById("renderBtn");
  renderBtn.disabled = true;
  renderBtn.classList.add("loading");
  renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

  showToast(
    "loading",
    "Rendering tiles...",
    `Processing ${coords.length} quadrant${
      coords.length > 1 ? "s" : ""
    }. This may take a moment.`
  );

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // If request was accepted, start polling for status
    if (response.status === 429) {
      // Already rendering - start polling
      showToast(
        "info",
        "Render in progress",
        "Reconnected to existing render operation."
      );
      startStatusPolling();
      return;
    }

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

      // Clear selection and refresh after a short delay
      deselectAll();
      setTimeout(() => {
        window.location.reload();
      }, 1500);
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
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Operation finished
      stopStatusPolling();

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);
        setTimeout(() => window.location.reload(), 1500);
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);
        if (isRendering) {
          resetRenderButton();
        } else {
          resetGenerateButton();
        }
      }
    }
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  if (isRenderOp) {
    if (!renderBtn.classList.contains("loading")) {
      renderBtn.disabled = true;
      renderBtn.classList.add("loading");
      renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      isRendering = true;

      // Disable generate button too
      generateBtn.disabled = true;

      // Apply locked styles to tiles
      applyLockedStyles();

      // Update selection status to show rendering message
      updateSelectionStatus();

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        showToast(
          "loading",
          "Render in progress...",
          status.message || "Please wait..."
        );
      }
    }
  } else {
    if (!generateBtn.classList.contains("loading")) {
      generateBtn.disabled = true;
      generateBtn.classList.add("loading");
      generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
      isGenerating = true;

      // Disable render button too
      renderBtn.disabled = true;

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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
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

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
    );
    return;
  }

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Set loading state and lock the quadrants
  isRendering = true;
  setLockedQuadrants(coords);
  document.body.classList.add("generating");

  // Mark selected tiles as locked (purple)
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
    tile.classList.add("locked");
  });
  selectedQuadrants.clear();
  updateSelectionStatus();

  const renderBtn = document.getElementById("renderBtn");
  renderBtn.disabled = true;
  renderBtn.classList.add("loading");
  renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

  showToast(
    "loading",
    "Rendering tiles...",
    `Processing ${coords.length} quadrant${
      coords.length > 1 ? "s" : ""
    }. This may take a moment.`
  );

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // If request was accepted, start polling for status
    if (response.status === 429) {
      // Already rendering - start polling
      showToast(
        "info",
        "Render in progress",
        "Reconnected to existing render operation."
      );
      startStatusPolling();
      return;
    }

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

      // Clear selection and refresh after a short delay
      deselectAll();
      setTimeout(() => {
        window.location.reload();
      }, 1500);
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
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Operation finished
      stopStatusPolling();

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);
        setTimeout(() => window.location.reload(), 1500);
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);
        if (isRendering) {
          resetRenderButton();
        } else {
          resetGenerateButton();
        }
      }
    }
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  if (isRenderOp) {
    if (!renderBtn.classList.contains("loading")) {
      renderBtn.disabled = true;
      renderBtn.classList.add("loading");
      renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      isRendering = true;

      // Disable generate button too
      generateBtn.disabled = true;

      // Apply locked styles to tiles
      applyLockedStyles();

      // Update selection status to show rendering message
      updateSelectionStatus();

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        showToast(
          "loading",
          "Render in progress...",
          status.message || "Please wait..."
        );
      }
    }
  } else {
    if (!generateBtn.classList.contains("loading")) {
      generateBtn.disabled = true;
      generateBtn.classList.add("loading");
      generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
      isGenerating = true;

      // Disable render button too
      renderBtn.disabled = true;

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
      startStatusPolling();
    } else {
      // Not generating/rendering - clear any stale locked state
      clearLockedQuadrants();
    }
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

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

/* Disable selection while generating */
body.generating .tile.selectable {
  cursor: not-allowed;
}

body.generating .tool-btn {
  opacity: 0.5;
  pointer-events: none;
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
  top: 100px;
  right: 15px;
  z-index: 1000;
  display: flex;
  flex-direction: column;
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

```

```javascript
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
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const isProcessing = isGenerating || isRendering;
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "rendering" : "generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    countEl.textContent = `${action} ${coordsStr}`;
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    countEl.textContent = `${count} quadrant${count !== 1 ? "s" : ""} selected`;
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  deselectBtn.disabled = count === 0 || isProcessing;
  deleteBtn.disabled = count === 0 || isProcessing;
  renderBtn.disabled = count === 0 || isProcessing;
  generateBtn.disabled = count === 0 || isProcessing;
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Cannot delete while generating or rendering."
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
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
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

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;
  if (isGenerating || isRendering) {
    showToast(
      "info",
      "Operation in progress",
      "Please wait for the current operation to complete."
    );
    return;
  }

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Set loading state and lock the quadrants
  isRendering = true;
  setLockedQuadrants(coords);
  document.body.classList.add("generating");

  // Mark selected tiles as locked (purple)
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
    tile.classList.add("locked");
  });
  selectedQuadrants.clear();
  updateSelectionStatus();

  const renderBtn = document.getElementById("renderBtn");
  renderBtn.disabled = true;
  renderBtn.classList.add("loading");
  renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

  showToast(
    "loading",
    "Rendering tiles...",
    `Processing ${coords.length} quadrant${
      coords.length > 1 ? "s" : ""
    }. This may take a moment.`
  );

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // If request was accepted, start polling for status
    if (response.status === 429) {
      // Already rendering - start polling
      showToast(
        "info",
        "Render in progress",
        "Reconnected to existing render operation."
      );
      startStatusPolling();
      return;
    }

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

      // Clear selection and refresh after a short delay
      deselectAll();
      setTimeout(() => {
        window.location.reload();
      }, 1500);
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
  if (isGenerating || isRendering) return; // Can't select while generating/rendering

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
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Operation finished
      stopStatusPolling();

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);
        setTimeout(() => window.location.reload(), 1500);
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);
        if (isRendering) {
          resetRenderButton();
        } else {
          resetGenerateButton();
        }
      }
    }
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  if (isRenderOp) {
    if (!renderBtn.classList.contains("loading")) {
      renderBtn.disabled = true;
      renderBtn.classList.add("loading");
      renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
      isRendering = true;

      // Disable generate button too
      generateBtn.disabled = true;

      // Apply locked styles to tiles
      applyLockedStyles();

      // Update selection status to show rendering message
      updateSelectionStatus();

      // Show toast if not already showing
      if (document.querySelectorAll(".toast.loading").length === 0) {
        showToast(
          "loading",
          "Render in progress...",
          status.message || "Please wait..."
        );
      }
    }
  } else {
    if (!generateBtn.classList.contains("loading")) {
      generateBtn.disabled = true;
      generateBtn.classList.add("loading");
      generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
      isGenerating = true;

      // Disable render button too
      renderBtn.disabled = true;

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
      startStatusPolling();
    } else {
      // Not generating/rendering - clear any stale locked state
      clearLockedQuadrants();
    }
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

I've added the "Render" button feature. Here's a summary of the changes:

### Python Backend (`view_generations.py`)
- Added `/api/render` endpoint that:
  - Accepts a list of quadrants to render
  - Uses the same locking mechanism as generate (only one operation at a time)
  - Shows progress through the status API (`rendering` status)
  - Ensures the web server is running, then renders each selected quadrant
  - Saves renders to the database

### HTML Template (`viewer.html`)
- Added a "Render" button next to the "Generate" button in the selection status bar

### JavaScript (`viewer.js`)
- Added `isRendering` state variable alongside `isGenerating`
- Added `renderSelected()` function that mirrors `generateSelected()` behavior
- Added `resetRenderButton()` function
- Updated `updateSelectionStatus()` to disable all buttons during either operation
- Renamed `setGeneratingUI()` to `setProcessingUI()` to handle both operations
- Updated `checkGenerationStatus()` to distinguish between render and generate results
- Updated all checks to use `isGenerating || isRendering` for proper mutual exclusion

### CSS (`viewer.css`)
- Added `.render-btn` styles with an amber/orange color (`#f59e0b`) to distinguish it from the cyan generate button
- Added loading state and spinner styles for the render button

The render button:
- Is disabled when no tiles are selected or during any operation
- Shows "Rendering..." with a spinner when active
- Displays toast notifications for progress and completion/errors
- Locks both Generate and Render buttons while either is in progress
- Refreshes the page on completion to show the new renders

