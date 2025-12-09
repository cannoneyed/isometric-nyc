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
