# Implement generation playbook plan
_Exported on 12/9/2025 at 08:46:50 EST from Cursor (2.1.39)_

---

**User**

Implement the plan in @tasks/016_generation_playbook.md 

---

**Cursor**

I'll implement the plan from the task file. Let me start by exploring the existing codebase to understand the current implementation.

Let me search for the web server that handles view generation and understand the existing architecture:

```json
[{
  "quadrants": "(x,y),(x,y)",
  "status": "pending"|"done"|"error"
}, ...]
```

Now I have a clear understanding of the codebase. Let me create a task list and implement the plan.

Let me create the `generate_omni.py` library by extracting the core generation logic:

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
    upload_to_gcs,
)

# Load environment variables
load_dotenv()

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
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


def call_oxen_api(image_url: str, api_key: str) -> str:
    """
    Call the Oxen API to generate pixel art.

    Args:
        image_url: Public URL of the input template image
        api_key: Oxen API key

    Returns:
        URL of the generated image

    Raises:
        requests.HTTPError: If the API call fails
        ValueError: If the response format is unexpected
    """
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

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        update_status("error", "OXEN_OMNI_v04_API_KEY not set")
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
        generated_url = call_oxen_api(image_url, api_key)

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

Now let me update `view_generations.py` to use the new library:

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
import sqlite3
import threading
import time
import traceback
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  image_to_png_bytes,
  save_quadrant_render,
  split_tile_into_quadrants,
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

Now let me remove the duplicated Oxen API and generation functions from `view_generations.py`, and update the generate API to use the library:

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
import sqlite3
import threading
import time
import traceback
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  image_to_png_bytes,
  save_quadrant_render,
  split_tile_into_quadrants,
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
import sqlite3
import threading
import time
import traceback
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  image_to_png_bytes,
  save_quadrant_render,
  split_tile_into_quadrants,
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

  return Response(data, mimetype="image/png")


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

```json
[{
  "quadrants": "(x,y),(x,y)",
  "status": "pending"|"done"|"error"
}, ...]
```

Now let me create the `generate_tiles_omni.py` command-line script:

```python
"""
Command-line script to generate pixel art for quadrants using the Oxen.ai model.

This script generates pixel art for specified quadrants, supporting both
direct quadrant specification and batch processing via JSON files.

Usage:
  # Generate specific quadrants:
  uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \\
    <generation_dir> \\
    --quadrants "(0,1),(0,2)"

  # Process a batch JSON file:
  uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \\
    <generation_dir> \\
    --quadrants-json path/to/quadrants.json

JSON File Format:
  [
    {"quadrants": "(0,0),(0,1)", "status": "pending"},
    {"quadrants": "(1,0),(1,1)", "status": "pending"},
    ...
  ]

The script will:
- Process entries with status "pending"
- Update status to "done" on success or "error" on failure
- Save progress after each entry so it can resume if interrupted
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.generate_omni import (
    parse_quadrant_list,
    run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.shared import (
    DEFAULT_WEB_PORT,
    WEB_DIR,
    get_generation_config,
    start_web_server,
)


def load_quadrants_json(json_path: Path) -> list[dict]:
    """
    Load quadrant entries from a JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        List of quadrant entry dicts with 'quadrants' and 'status' keys
    """
    with open(json_path) as f:
        return json.load(f)


def save_quadrants_json(json_path: Path, entries: list[dict]) -> None:
    """
    Save quadrant entries to a JSON file.

    Args:
        json_path: Path to the JSON file
        entries: List of quadrant entry dicts
    """
    with open(json_path, "w") as f:
        json.dump(entries, f, indent=2)


def generate_quadrants(
    generation_dir: Path,
    quadrant_tuples: list[tuple[int, int]],
    port: int,
    no_start_server: bool,
) -> dict:
    """
    Generate pixel art for the specified quadrants.

    Args:
        generation_dir: Path to the generation directory
        quadrant_tuples: List of (x, y) quadrant coordinates
        port: Web server port
        no_start_server: If True, don't start the web server

    Returns:
        Dict with success status and message/error
    """
    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        return {"success": False, "error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    web_server = None

    try:
        config = get_generation_config(conn)

        if not no_start_server:
            web_server = start_web_server(WEB_DIR, port)

        result = run_generation_for_quadrants(
            conn=conn,
            config=config,
            selected_quadrants=quadrant_tuples,
            port=port,
        )
        return result

    finally:
        conn.close()
        if web_server:
            print("🛑 Stopping web server...")
            web_server.terminate()
            web_server.wait()


def process_quadrants_list(
    generation_dir: Path,
    quadrants_str: str,
    port: int,
    no_start_server: bool,
) -> int:
    """
    Process a comma-separated list of quadrant tuples.

    Args:
        generation_dir: Path to the generation directory
        quadrants_str: String like "(0,1),(0,2)"
        port: Web server port
        no_start_server: If True, don't start the web server

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        quadrant_tuples = parse_quadrant_list(quadrants_str)
    except ValueError as e:
        print(f"❌ Error parsing quadrants: {e}")
        return 1

    print(f"🎯 Generating quadrants: {quadrant_tuples}")
    result = generate_quadrants(generation_dir, quadrant_tuples, port, no_start_server)

    if result.get("success"):
        print(f"✅ {result.get('message')}")
        return 0
    else:
        print(f"❌ Error: {result.get('error')}")
        return 1


def process_quadrants_json(
    generation_dir: Path,
    json_path: Path,
    port: int,
    no_start_server: bool,
) -> int:
    """
    Process a JSON file of quadrant entries.

    The script will:
    - Process entries with status "pending"
    - Update status to "done" on success or "error" on failure
    - Save progress after each entry

    Args:
        generation_dir: Path to the generation directory
        json_path: Path to the JSON file
        port: Web server port
        no_start_server: If True, don't start the web server

    Returns:
        Exit code (0 for all success, 1 if any error)
    """
    # Load entries
    try:
        entries = load_quadrants_json(json_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading JSON file: {e}")
        return 1

    # Count pending entries
    pending_entries = [e for e in entries if e.get("status") == "pending"]
    if not pending_entries:
        print("✅ No pending entries to process")
        return 0

    print(f"📋 Found {len(pending_entries)} pending entries out of {len(entries)} total")

    # Connect to database
    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    web_server = None
    had_error = False

    try:
        config = get_generation_config(conn)

        if not no_start_server:
            web_server = start_web_server(WEB_DIR, port)

        # Process each pending entry
        for i, entry in enumerate(entries):
            if entry.get("status") != "pending":
                continue

            quadrants_str = entry.get("quadrants", "")
            entry_idx = entries.index(entry)

            print(f"\n{'=' * 60}")
            print(f"📦 Processing entry {i + 1}/{len(pending_entries)}: {quadrants_str}")
            print(f"{'=' * 60}")

            try:
                quadrant_tuples = parse_quadrant_list(quadrants_str)
            except ValueError as e:
                print(f"❌ Error parsing quadrants: {e}")
                entries[entry_idx]["status"] = "error"
                entries[entry_idx]["error"] = str(e)
                save_quadrants_json(json_path, entries)
                had_error = True
                break

            try:
                result = run_generation_for_quadrants(
                    conn=conn,
                    config=config,
                    selected_quadrants=quadrant_tuples,
                    port=port,
                )

                if result.get("success"):
                    print(f"✅ {result.get('message')}")
                    entries[entry_idx]["status"] = "done"
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"❌ Error: {error_msg}")
                    entries[entry_idx]["status"] = "error"
                    entries[entry_idx]["error"] = error_msg
                    had_error = True

            except Exception as e:
                print(f"❌ Exception: {e}")
                entries[entry_idx]["status"] = "error"
                entries[entry_idx]["error"] = str(e)
                had_error = True

            # Save progress after each entry
            save_quadrants_json(json_path, entries)

            # Stop on error
            if had_error:
                print("\n⛔ Stopping due to error. Progress has been saved.")
                break

    finally:
        conn.close()
        if web_server:
            print("\n🛑 Stopping web server...")
            web_server.terminate()
            web_server.wait()

    # Print summary
    done_count = sum(1 for e in entries if e.get("status") == "done")
    error_count = sum(1 for e in entries if e.get("status") == "error")
    pending_count = sum(1 for e in entries if e.get("status") == "pending")

    print(f"\n{'=' * 60}")
    print("📊 Summary:")
    print(f"   Done: {done_count}")
    print(f"   Error: {error_count}")
    print(f"   Pending: {pending_count}")
    print(f"{'=' * 60}")

    return 1 if had_error else 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate pixel art for quadrants using the Oxen.ai model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )

    # Quadrant specification (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--quadrants",
        type=str,
        help='Comma-separated quadrant tuples to generate, e.g. "(0,1),(0,2)"',
    )
    input_group.add_argument(
        "--quadrants-json",
        type=Path,
        help="Path to JSON file with quadrant entries to process",
    )

    # Server options
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help=f"Web server port for rendering (default: {DEFAULT_WEB_PORT})",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )

    args = parser.parse_args()

    generation_dir = args.generation_dir.resolve()

    if not generation_dir.exists():
        print(f"❌ Error: Directory not found: {generation_dir}")
        return 1

    if not generation_dir.is_dir():
        print(f"❌ Error: Not a directory: {generation_dir}")
        return 1

    print(f"📂 Generation directory: {generation_dir}")

    try:
        if args.quadrants:
            return process_quadrants_list(
                generation_dir,
                args.quadrants,
                args.port,
                args.no_start_server,
            )
        else:
            json_path = args.quadrants_json.resolve()
            if not json_path.exists():
                print(f"❌ Error: JSON file not found: {json_path}")
                return 1
            print(f"📄 JSON file: {json_path}")
            return process_quadrants_json(
                generation_dir,
                json_path,
                args.port,
                args.no_start_server,
            )

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    exit(main())

```

Let me check for any linting errors in all the modified files:

Let me run the formatter and linter to ensure code quality:

Let me fix the linting issues by running the formatter with auto-fix:

Let me verify the files are correct after the auto-fix by reading them:

Let me update the task file to indicate completion:

```markdown
# Generation playbook

> **Status: COMPLETED** ✅

## Task Description

Next, I want to create a script that works like the logic in
`src/isometric_nyc/e2e_generation/view_generations.py`, except it generates
quadrants programatically - please extract the logic out of that web server into
a "generate_omni.py" library, and use that logic for both the `view_generations`
script and a new `generate_tiles_omni` script.

The script should take a few parameters:

- generation_dir - the directory containing the generation config

- quadrants - a list of comma separated quadrant tuples to generate (e.g.
  "(0,1),(0,2)")
- quadrants_json - a json file containing a list of quadrant generations to
  generate in order

If provided a quadrants_json parameter, the JSON schema will look like this:

```
[{
  "quadrants": "(x,y),(x,y)",
  "status": "pending"|"done"|"error"
}, ...]
```

If given a quadrants json, the script should generate the quadrants in order
until done or an error is encountered, and update the json file accordingly
if/when an entry is finished. It should also be able to pick up where it left
off.

## Implementation

### Files Created/Modified

1. **`src/isometric_nyc/e2e_generation/generate_omni.py`** (NEW)
   - Reusable library with core generation logic
   - `parse_quadrant_tuple()` / `parse_quadrant_list()` - parsing quadrant strings
   - `call_oxen_api()` - calling the Oxen API
   - `download_image_to_pil()` - downloading images
   - `render_quadrant()` - rendering a quadrant via Playwright
   - `run_generation_for_quadrants()` - main generation pipeline

2. **`src/isometric_nyc/e2e_generation/generate_tiles_omni.py`** (NEW)
   - Command-line script for batch generation
   - Supports `--quadrants "(0,1),(0,2)"` for direct specification
   - Supports `--quadrants-json path/to/file.json` for batch processing
   - Updates JSON file after each entry for resume capability

3. **`src/isometric_nyc/e2e_generation/view_generations.py`** (MODIFIED)
   - Now imports from `generate_omni.py`
   - Uses shared `run_generation_for_quadrants()` function
   - Removed duplicated code

### Usage

```bash
# Generate specific quadrants:
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  <generation_dir> \
  --quadrants "(0,1),(0,2)"

# Process a batch JSON file:
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  <generation_dir> \
  --quadrants-json path/to/quadrants.json
```

```

I've successfully implemented the plan from the task file. Here's a summary of what was created:

## Summary

### 1. `generate_omni.py` - Reusable Library
A new library containing the core generation logic extracted from `view_generations.py`:

- **`parse_quadrant_tuple()`** / **`parse_quadrant_list()`** - Parse quadrant strings like `"(0,1)"` or `"(0,1),(0,2)"`
- **`call_oxen_api()`** - Call the Oxen API to generate pixel art
- **`download_image_to_pil()`** - Download images from URLs
- **`render_quadrant()`** - Render a quadrant using Playwright
- **`run_generation_for_quadrants()`** - Main entry point that handles the full generation pipeline (validation, rendering, template building, API calling, saving)

### 2. `generate_tiles_omni.py` - Command-Line Script
A new script that supports:

- **Direct quadrant specification**: `--quadrants "(0,1),(0,2)"`
- **Batch JSON processing**: `--quadrants-json path/to/file.json`

The JSON format is as specified:
```json
[
  {"quadrants": "(0,0),(0,1)", "status": "pending"},
  {"quadrants": "(1,0),(1,1)", "status": "pending"}
]
```

Features:
- Processes entries with status `"pending"`
- Updates status to `"done"` on success or `"error"` on failure
- Saves progress after each entry (resumable)
- Stops on first error to prevent cascading failures

### 3. Updated `view_generations.py`
- Now imports from `generate_omni.py`
- Uses the shared `run_generation_for_quadrants()` function
- Removed duplicated code (~150 lines)

### Usage Examples

```bash
# Generate specific quadrants:
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  generations/v01 \
  --quadrants "(0,1),(0,2)"

# Process a batch JSON file:
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  generations/v01 \
  --quadrants-json my_quadrants.json

# With custom port (if server already running):
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  generations/v01 \
  --quadrants "(0,0),(0,1)" \
  --no-start-server \
  --port 5173
```

```json
[
  {"quadrants": "(0,0),(0,1)", "status": "pending"},
  {"quadrants": "(1,0),(1,1)", "status": "pending"}
]
```

```shellscript
# Generate specific quadrants:
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  generations/v01 \
  --quadrants "(0,1),(0,2)"

# Process a batch JSON file:
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  generations/v01 \
  --quadrants-json my_quadrants.json

# With custom port (if server already running):
uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \
  generations/v01 \
  --quadrants "(0,0),(0,1)" \
  --no-start-server \
  --port 5173
```

---

**User**

can you create a new script in @e2e_generation called `get_tile_data` that takes a given x,y coordinate and a generation_dir and outputs a `render_x_y.png` and a `generation_x_y.png` image for the 2x2 tile defined by the top left x y quadrant?

---

**Cursor**

I'll create a new script `get_tile_data.py` that extracts and saves the render and generation images for a 2x2 tile.

```python
"""
Extract tile data (render and generation) for a 2x2 tile.

This script retrieves the render and generation images for a 2x2 tile
defined by its top-left quadrant coordinate and saves them as PNG files.

Usage:
  uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \\
    <generation_dir> <x> <y>

  # Specify output directory:
  uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \\
    <generation_dir> <x> <y> --output-dir ./exports

Output:
  - render_<x>_<y>.png: The stitched 2x2 render image
  - generation_<x>_<y>.png: The stitched 2x2 generation image
"""

import argparse
import sqlite3
from pathlib import Path

from PIL import Image

from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation,
  get_quadrant_render,
  png_bytes_to_image,
  stitch_quadrants_to_tile,
)


def get_tile_images(
  conn: sqlite3.Connection,
  x: int,
  y: int,
) -> tuple[Image.Image | None, Image.Image | None]:
  """
  Get the stitched render and generation images for a 2x2 tile.

  Args:
    conn: Database connection
    x: Top-left quadrant x coordinate
    y: Top-left quadrant y coordinate

  Returns:
    Tuple of (render_image, generation_image). Either may be None if
    any quadrant is missing data.
  """
  # The 4 quadrants of a 2x2 tile
  quadrant_offsets = [
    (0, 0),  # top-left
    (1, 0),  # top-right
    (0, 1),  # bottom-left
    (1, 1),  # bottom-right
  ]

  # Collect render quadrants
  render_quadrants: dict[tuple[int, int], Image.Image] = {}
  render_missing = []

  for dx, dy in quadrant_offsets:
    qx, qy = x + dx, y + dy
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      render_quadrants[(dx, dy)] = png_bytes_to_image(render_bytes)
    else:
      render_missing.append((qx, qy))

  # Collect generation quadrants
  generation_quadrants: dict[tuple[int, int], Image.Image] = {}
  generation_missing = []

  for dx, dy in quadrant_offsets:
    qx, qy = x + dx, y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      generation_quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)
    else:
      generation_missing.append((qx, qy))

  # Stitch render if all quadrants present
  render_image = None
  if len(render_quadrants) == 4:
    render_image = stitch_quadrants_to_tile(render_quadrants)
  elif render_missing:
    print(f"⚠️  Missing render data for quadrants: {render_missing}")

  # Stitch generation if all quadrants present
  generation_image = None
  if len(generation_quadrants) == 4:
    generation_image = stitch_quadrants_to_tile(generation_quadrants)
  elif generation_missing:
    print(f"⚠️  Missing generation data for quadrants: {generation_missing}")

  return render_image, generation_image


def main():
  parser = argparse.ArgumentParser(
    description="Extract tile data (render and generation) for a 2x2 tile.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Top-left quadrant x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Top-left quadrant y coordinate",
  )
  parser.add_argument(
    "--output-dir",
    "-o",
    type=Path,
    default=None,
    help="Output directory for images (default: generation_dir)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()
  output_dir = args.output_dir.resolve() if args.output_dir else generation_dir

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Ensure output directory exists
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"📂 Generation directory: {generation_dir}")
  print(f"📂 Output directory: {output_dir}")
  print(f"🎯 Tile at ({args.x}, {args.y})")
  print(f"   Quadrants: ({args.x},{args.y}), ({args.x + 1},{args.y}), "
        f"({args.x},{args.y + 1}), ({args.x + 1},{args.y + 1})")

  conn = sqlite3.connect(db_path)

  try:
    render_image, generation_image = get_tile_images(conn, args.x, args.y)

    saved_any = False

    if render_image:
      render_path = output_dir / f"render_{args.x}_{args.y}.png"
      render_image.save(render_path)
      print(f"✅ Saved render to: {render_path}")
      saved_any = True
    else:
      print("❌ Could not create render image (missing quadrant data)")

    if generation_image:
      generation_path = output_dir / f"generation_{args.x}_{args.y}.png"
      generation_image.save(generation_path)
      print(f"✅ Saved generation to: {generation_path}")
      saved_any = True
    else:
      print("❌ Could not create generation image (missing quadrant data)")

    return 0 if saved_any else 1

  finally:
    conn.close()


if __name__ == "__main__":
  exit(main())

```

The script is ready. Here's a summary:

## `get_tile_data.py`

A new script that extracts render and generation images for a 2x2 tile.

### Usage

```bash
# Basic usage - outputs to generation_dir
uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \
  generations/v01 0 0

# Specify output directory
uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \
  generations/v01 0 0 --output-dir ./exports
```

### Output Files
- `render_<x>_<y>.png` - The stitched 2x2 render image (4 quadrants combined)
- `generation_<x>_<y>.png` - The stitched 2x2 generation image (4 quadrants combined)

### Features
- Takes a top-left quadrant coordinate (x, y) and retrieves all 4 quadrants:
  - (x, y), (x+1, y), (x, y+1), (x+1, y+1)
- Reports which quadrants are missing if any data is unavailable
- Creates output directory if it doesn't exist
- Uses the existing `stitch_quadrants_to_tile()` utility from `shared.py`

```shellscript
# Basic usage - outputs to generation_dir
uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \
  generations/v01 0 0

# Specify output directory
uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \
  generations/v01 0 0 --output-dir ./exports
```

---

**User**

saveto exports subdir in the generation dir please

---

**Cursor**

```python
"""
Extract tile data (render and generation) for a 2x2 tile.

This script retrieves the render and generation images for a 2x2 tile
defined by its top-left quadrant coordinate and saves them as PNG files.

Usage:
  uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \\
    <generation_dir> <x> <y>

  # Specify output directory:
  uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \\
    <generation_dir> <x> <y> --output-dir ./exports

Output:
  - render_<x>_<y>.png: The stitched 2x2 render image
  - generation_<x>_<y>.png: The stitched 2x2 generation image
"""

import argparse
import sqlite3
from pathlib import Path

from PIL import Image

from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation,
  get_quadrant_render,
  png_bytes_to_image,
  stitch_quadrants_to_tile,
)


def get_tile_images(
  conn: sqlite3.Connection,
  x: int,
  y: int,
) -> tuple[Image.Image | None, Image.Image | None]:
  """
  Get the stitched render and generation images for a 2x2 tile.

  Args:
    conn: Database connection
    x: Top-left quadrant x coordinate
    y: Top-left quadrant y coordinate

  Returns:
    Tuple of (render_image, generation_image). Either may be None if
    any quadrant is missing data.
  """
  # The 4 quadrants of a 2x2 tile
  quadrant_offsets = [
    (0, 0),  # top-left
    (1, 0),  # top-right
    (0, 1),  # bottom-left
    (1, 1),  # bottom-right
  ]

  # Collect render quadrants
  render_quadrants: dict[tuple[int, int], Image.Image] = {}
  render_missing = []

  for dx, dy in quadrant_offsets:
    qx, qy = x + dx, y + dy
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      render_quadrants[(dx, dy)] = png_bytes_to_image(render_bytes)
    else:
      render_missing.append((qx, qy))

  # Collect generation quadrants
  generation_quadrants: dict[tuple[int, int], Image.Image] = {}
  generation_missing = []

  for dx, dy in quadrant_offsets:
    qx, qy = x + dx, y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      generation_quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)
    else:
      generation_missing.append((qx, qy))

  # Stitch render if all quadrants present
  render_image = None
  if len(render_quadrants) == 4:
    render_image = stitch_quadrants_to_tile(render_quadrants)
  elif render_missing:
    print(f"⚠️  Missing render data for quadrants: {render_missing}")

  # Stitch generation if all quadrants present
  generation_image = None
  if len(generation_quadrants) == 4:
    generation_image = stitch_quadrants_to_tile(generation_quadrants)
  elif generation_missing:
    print(f"⚠️  Missing generation data for quadrants: {generation_missing}")

  return render_image, generation_image


def main():
  parser = argparse.ArgumentParser(
    description="Extract tile data (render and generation) for a 2x2 tile.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Top-left quadrant x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Top-left quadrant y coordinate",
  )
  parser.add_argument(
    "--output-dir",
    "-o",
    type=Path,
    default=None,
    help="Output directory for images (default: generation_dir/exports)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()
  output_dir = args.output_dir.resolve() if args.output_dir else generation_dir / "exports"

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Ensure output directory exists
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"📂 Generation directory: {generation_dir}")
  print(f"📂 Output directory: {output_dir}")
  print(f"🎯 Tile at ({args.x}, {args.y})")
  print(
    f"   Quadrants: ({args.x},{args.y}), ({args.x + 1},{args.y}), "
    f"({args.x},{args.y + 1}), ({args.x + 1},{args.y + 1})"
  )

  conn = sqlite3.connect(db_path)

  try:
    render_image, generation_image = get_tile_images(conn, args.x, args.y)

    saved_any = False

    if render_image:
      render_path = output_dir / f"render_{args.x}_{args.y}.png"
      render_image.save(render_path)
      print(f"✅ Saved render to: {render_path}")
      saved_any = True
    else:
      print("❌ Could not create render image (missing quadrant data)")

    if generation_image:
      generation_path = output_dir / f"generation_{args.x}_{args.y}.png"
      generation_image.save(generation_path)
      print(f"✅ Saved generation to: {generation_path}")
      saved_any = True
    else:
      print("❌ Could not create generation image (missing quadrant data)")

    return 0 if saved_any else 1

  finally:
    conn.close()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Extract tile data (render and generation) for a 2x2 tile.

This script retrieves the render and generation images for a 2x2 tile
defined by its top-left quadrant coordinate and saves them as PNG files.

Usage:
  uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \\
    <generation_dir> <x> <y>

  # Specify custom output directory:
  uv run python src/isometric_nyc/e2e_generation/get_tile_data.py \\
    <generation_dir> <x> <y> --output-dir ./my_exports

Output (saved to <generation_dir>/exports/ by default):
  - render_<x>_<y>.png: The stitched 2x2 render image
  - generation_<x>_<y>.png: The stitched 2x2 generation image
"""

import argparse
import sqlite3
from pathlib import Path

from PIL import Image

from isometric_nyc.e2e_generation.shared import (
  get_quadrant_generation,
  get_quadrant_render,
  png_bytes_to_image,
  stitch_quadrants_to_tile,
)


def get_tile_images(
  conn: sqlite3.Connection,
  x: int,
  y: int,
) -> tuple[Image.Image | None, Image.Image | None]:
  """
  Get the stitched render and generation images for a 2x2 tile.

  Args:
    conn: Database connection
    x: Top-left quadrant x coordinate
    y: Top-left quadrant y coordinate

  Returns:
    Tuple of (render_image, generation_image). Either may be None if
    any quadrant is missing data.
  """
  # The 4 quadrants of a 2x2 tile
  quadrant_offsets = [
    (0, 0),  # top-left
    (1, 0),  # top-right
    (0, 1),  # bottom-left
    (1, 1),  # bottom-right
  ]

  # Collect render quadrants
  render_quadrants: dict[tuple[int, int], Image.Image] = {}
  render_missing = []

  for dx, dy in quadrant_offsets:
    qx, qy = x + dx, y + dy
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      render_quadrants[(dx, dy)] = png_bytes_to_image(render_bytes)
    else:
      render_missing.append((qx, qy))

  # Collect generation quadrants
  generation_quadrants: dict[tuple[int, int], Image.Image] = {}
  generation_missing = []

  for dx, dy in quadrant_offsets:
    qx, qy = x + dx, y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      generation_quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)
    else:
      generation_missing.append((qx, qy))

  # Stitch render if all quadrants present
  render_image = None
  if len(render_quadrants) == 4:
    render_image = stitch_quadrants_to_tile(render_quadrants)
  elif render_missing:
    print(f"⚠️  Missing render data for quadrants: {render_missing}")

  # Stitch generation if all quadrants present
  generation_image = None
  if len(generation_quadrants) == 4:
    generation_image = stitch_quadrants_to_tile(generation_quadrants)
  elif generation_missing:
    print(f"⚠️  Missing generation data for quadrants: {generation_missing}")

  return render_image, generation_image


def main():
  parser = argparse.ArgumentParser(
    description="Extract tile data (render and generation) for a 2x2 tile.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Top-left quadrant x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Top-left quadrant y coordinate",
  )
  parser.add_argument(
    "--output-dir",
    "-o",
    type=Path,
    default=None,
    help="Output directory for images (default: generation_dir/exports)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()
  output_dir = args.output_dir.resolve() if args.output_dir else generation_dir / "exports"

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Ensure output directory exists
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"📂 Generation directory: {generation_dir}")
  print(f"📂 Output directory: {output_dir}")
  print(f"🎯 Tile at ({args.x}, {args.y})")
  print(
    f"   Quadrants: ({args.x},{args.y}), ({args.x + 1},{args.y}), "
    f"({args.x},{args.y + 1}), ({args.x + 1},{args.y + 1})"
  )

  conn = sqlite3.connect(db_path)

  try:
    render_image, generation_image = get_tile_images(conn, args.x, args.y)

    saved_any = False

    if render_image:
      render_path = output_dir / f"render_{args.x}_{args.y}.png"
      render_image.save(render_path)
      print(f"✅ Saved render to: {render_path}")
      saved_any = True
    else:
      print("❌ Could not create render image (missing quadrant data)")

    if generation_image:
      generation_path = output_dir / f"generation_{args.x}_{args.y}.png"
      generation_image.save(generation_path)
      print(f"✅ Saved generation to: {generation_path}")
      saved_any = True
    else:
      print("❌ Could not create generation image (missing quadrant data)")

    return 0 if saved_any else 1

  finally:
    conn.close()


if __name__ == "__main__":
  exit(main())

```

Done. The script now saves to `<generation_dir>/exports/` by default. You can still override with `--output-dir` if needed.

