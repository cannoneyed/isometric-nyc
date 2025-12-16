# Image download and generation retries
_Exported on 12/16/2025 at 10:16:56 EST from Cursor (2.1.39)_

---

**User**

For the generation in @src/isometric_nyc/e2e_generation/app.py , I'm seeing a lot of errors when trying to download the image that's been generated. Can we add two things:



First, if we encounter an error for downloading the image, can you wait 10s, then try again, and repeat this 3 times?



```

# Example error

raise HTTPError(http_error_msg, response=self)

requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://hub.oxen.ai/api/repos/cannoneyed/playground/file/main/cannoneyed-rural-rose-dingo/images/images_1765821943_cpgpoxgpi5s3c.png?oxen_expires=1765825599&oxen_signature=5zLrq4-s1jKmzO6GQWVotvRNskQTLPrwdlMQixZ4LXA%3D

```



If the error persists, then please try the generation again, don't just skip ahead to the next generation automatically. Let's also retry generations 3 times



---

**Cursor**

I'll help you add retry logic for image downloads and generation failures. First, let me find where the image download happens in the codebase.

Let me read the full `generate_omni.py` file to understand the complete generation flow:

Now I understand the flow. I need to:

1. Add retry logic to `download_image_to_pil` with 3 retries and 10-second waits
2. Add retry logic to the generation process in `app.py` to retry the entire generation up to 3 times if it fails

Let me make these changes:

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
  additional_prompt: str | None = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image
      model_config: Optional model configuration (ModelConfig from model_config.py).
        If not provided, uses defaults.
      additional_prompt: Optional additional text to append to the base prompt

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

  # Build prompt - base prompt plus any additional text
  base_prompt = (
    "Fill in the outlined section with the missing pixels corresponding to "
    "the <isometric nyc pixel art> style, removing the border and exactly "
    "following the shape/style/structure of the surrounding image (if present)."
  )

  if additional_prompt:
    prompt = f"{base_prompt} {additional_prompt}"
    print(f"   📝 Using additional prompt: {additional_prompt}")
  else:
    prompt = base_prompt

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

  # Log the response structure for debugging
  print(f"   📥 API response keys: {list(result.keys())}")

  # Try various response formats
  if "images" in result and len(result["images"]) > 0:
    image_data = result["images"][0]
    print(
      f"   📥 Image data keys: {list(image_data.keys()) if isinstance(image_data, dict) else type(image_data)}"
    )

    # Try different possible keys for the image URL
    if isinstance(image_data, dict):
      if "url" in image_data:
        return image_data["url"]
      elif "image_url" in image_data:
        return image_data["image_url"]
      elif "data" in image_data:
        # Some APIs return base64 data - we'd need to handle this differently
        raise ValueError(
          f"API returned base64 data instead of URL: {list(image_data.keys())}"
        )
      else:
        raise ValueError(
          f"Image data missing 'url' key. Available keys: {list(image_data.keys())}"
        )
    elif isinstance(image_data, str):
      # Direct URL string
      return image_data
    else:
      raise ValueError(f"Unexpected image data type: {type(image_data)}")
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  elif "error" in result:
    raise ValueError(f"API returned error: {result['error']}")
  elif "message" in result:
    raise ValueError(f"API returned message: {result['message']}")
  else:
    raise ValueError(
      f"Unexpected API response format. Keys: {list(result.keys())}, Full response: {result}"
    )


def download_image_to_pil(
  url: str,
  max_retries: int = 3,
  retry_delay: float = 10.0,
) -> Image.Image:
  """
  Download an image from a URL and return as PIL Image.

  Includes retry logic for transient errors (e.g., 403 Forbidden when
  the image is not yet available).

  Args:
      url: URL of the image to download
      max_retries: Maximum number of retry attempts (default: 3)
      retry_delay: Seconds to wait between retries (default: 10.0)

  Returns:
      PIL Image object

  Raises:
      requests.HTTPError: If all retry attempts fail
  """
  import time

  last_error = None

  for attempt in range(1, max_retries + 1):
    try:
      response = requests.get(url, timeout=120)
      response.raise_for_status()
      return Image.open(BytesIO(response.content))
    except requests.exceptions.HTTPError as e:
      last_error = e
      if attempt < max_retries:
        print(
          f"   ⚠️  Download failed (attempt {attempt}/{max_retries}): {e}"
        )
        print(f"   ⏳ Waiting {retry_delay}s before retrying...")
        time.sleep(retry_delay)
      else:
        print(f"   ❌ Download failed after {max_retries} attempts: {e}")
    except requests.exceptions.RequestException as e:
      last_error = e
      if attempt < max_retries:
        print(
          f"   ⚠️  Download error (attempt {attempt}/{max_retries}): {e}"
        )
        print(f"   ⏳ Waiting {retry_delay}s before retrying...")
        time.sleep(retry_delay)
      else:
        print(f"   ❌ Download failed after {max_retries} attempts: {e}")

  # If we get here, all retries failed
  if last_error:
    raise last_error
  raise RuntimeError("Download failed with no error captured")


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
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
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
      context_quadrants: Optional list of (x, y) quadrant coordinates to use as
        context. These quadrants provide surrounding pixel art context for the
        generation. If a context quadrant has a generation, that will be used;
        otherwise the render will be used.
      prompt: Optional additional prompt text for generation

  Returns:
      Dict with:
          - success: bool
          - message: str (on success)
          - error: str (on failure)
          - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Create helper functions for validation
  # These are modified to treat context quadrants as "generated" if they have
  # either a generation or a render
  def has_generation_in_db(qx: int, qy: int) -> bool:
    # Check if this quadrant has an actual generation
    gen = shared_get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True

    # For context quadrants, treat them as "generated" if they have a render
    # This allows context quadrants to provide surrounding content even if
    # they don't have generations yet
    if (qx, qy) in context_set:
      render = shared_get_quadrant_render(conn, qx, qy)
      return render is not None

    return False

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
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    # This allows context quadrants to provide surrounding pixel art context
    # even if they only have renders
    if (qx, qy) in context_set:
      render_bytes = shared_get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

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
    print(f"   Template path: {template_path}")
    print(f"   Template size: {template_image.size[0]}x{template_image.size[1]}")
    image_url = upload_to_gcs(template_path, bucket_name)
    print(f"   Uploaded URL: {image_url}")

    update_status("generating", "Calling AI model (this may take a minute)...")
    print("🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url, model_config, prompt)

    update_status("saving", "Downloading and saving results...")
    print("📥 Downloading generated image...")
    print(f"   Generated URL: {generated_url}")
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

Now let me add retry logic to the generation process in `app.py`. I'll modify the `process_queue_item_from_db` function to retry the generation up to 3 times:

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
  """Get info about a quadrant including whether it has data, is flagged, and is water."""
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": bool(row[2]),
      }
    return {"has_data": False, "flagged": False, "is_water": False}
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

  # Check which tiles have data, flagged status, and water status
  tiles = {}
  flagged_tiles = {}
  water_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      water_tiles[(dx, dy)] = info["is_water"]

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
    water_tiles=water_tiles,
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
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt
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
      # Clear the render column (set to NULL) but keep the row
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL
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


@app.route("/api/water", methods=["POST"])
def api_water():
  """API endpoint to mark/unmark selected quadrants as water tiles."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get water value (default to True/1 for marking as water, False/0 for unmarking)
  water_value = 1 if data.get("is_water", True) else 0

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

    action = "Marked as water" if water_value else "Unmarked as water"
    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "is_water": bool(water_value),
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

```

Let me check for any linting errors in both files:

Let me also check if there's a copy of `download_image_to_pil` in `automatic_generation.py` that needs the same update:

Let me read the `automatic_generation.py` to see how it's used:

Let me search for where `download_image_to_pil` is used in `automatic_generation.py`:

Let me update the `download_image_to_pil` in `automatic_generation.py` with the same retry logic:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles OFFSET from center (with a gap to avoid seams)
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  # Generate a plan (does not execute):
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y>

  # Execute an existing plan:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --plan-json <path_to_plan.json>

Example:
  # Create plan:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20

  # Execute plan:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --plan-json generations/test_generation/automatic_plan.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
  """State of a quadrant in the grid."""

  EMPTY = "empty"  # Not generated
  GENERATED = "generated"  # Already has generation
  SELECTED = "selected"  # Selected for generation in current step


class StepStatus(Enum):
  """Status of a generation step."""

  PENDING = "pending"
  GENERATED = "generated"
  ERROR = "error"


@dataclass(frozen=True)
class Point:
  """A 2D point representing a quadrant coordinate."""

  x: int
  y: int

  def __str__(self) -> str:
    return f"({self.x}, {self.y})"

  def __add__(self, other: Point) -> Point:
    return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
  """Bounding box defined by top-left and bottom-right corners."""

  top_left: Point
  bottom_right: Point

  @property
  def width(self) -> int:
    return self.bottom_right.x - self.top_left.x + 1

  @property
  def height(self) -> int:
    return self.bottom_right.y - self.top_left.y + 1

  @property
  def area(self) -> int:
    return self.width * self.height

  def contains(self, p: Point) -> bool:
    return (
      self.top_left.x <= p.x <= self.bottom_right.x
      and self.top_left.y <= p.y <= self.bottom_right.y
    )

  def all_points(self) -> list[Point]:
    """Return all points within the bounding box."""
    return [
      Point(x, y)
      for y in range(self.top_left.y, self.bottom_right.y + 1)
      for x in range(self.top_left.x, self.bottom_right.x + 1)
    ]


@dataclass
class GenerationStep:
  """A single step in the generation plan."""

  step_number: int
  quadrants: list[Point]  # List of quadrant positions to generate
  description: str
  status: StepStatus = StepStatus.PENDING
  error_message: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    return {
      "step_number": self.step_number,
      "quadrants": [{"x": q.x, "y": q.y} for q in self.quadrants],
      "description": self.description,
      "status": self.status.value,
      "error_message": self.error_message,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> GenerationStep:
    """Create from JSON dict."""
    return cls(
      step_number=data["step_number"],
      quadrants=[Point(q["x"], q["y"]) for q in data["quadrants"]],
      description=data["description"],
      status=StepStatus(data.get("status", "pending")),
      error_message=data.get("error_message"),
    )


@dataclass
class GenerationPlan:
  """A complete generation plan with metadata."""

  created_at: str
  bounds: BoundingBox
  steps: list[GenerationStep]
  generation_dir: str

  def to_dict(self) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    return {
      "created_at": self.created_at,
      "bounds": {
        "top_left": {"x": self.bounds.top_left.x, "y": self.bounds.top_left.y},
        "bottom_right": {
          "x": self.bounds.bottom_right.x,
          "y": self.bounds.bottom_right.y,
        },
      },
      "generation_dir": self.generation_dir,
      "steps": [step.to_dict() for step in self.steps],
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> GenerationPlan:
    """Create from JSON dict."""
    bounds = BoundingBox(
      top_left=Point(data["bounds"]["top_left"]["x"], data["bounds"]["top_left"]["y"]),
      bottom_right=Point(
        data["bounds"]["bottom_right"]["x"], data["bounds"]["bottom_right"]["y"]
      ),
    )
    return cls(
      created_at=data["created_at"],
      bounds=bounds,
      generation_dir=data["generation_dir"],
      steps=[GenerationStep.from_dict(s) for s in data["steps"]],
    )

  def save(self, path: Path) -> None:
    """Save plan to JSON file."""
    with open(path, "w") as f:
      json.dump(self.to_dict(), f, indent=2)
    print(f"📄 Saved plan to {path}")

  @classmethod
  def load(cls, path: Path) -> GenerationPlan:
    """Load plan from JSON file."""
    with open(path) as f:
      data = json.load(f)
    return cls.from_dict(data)

  def update_step_status(
    self, step_number: int, status: StepStatus, error_message: str | None = None
  ) -> None:
    """Update the status of a specific step."""
    for step in self.steps:
      if step.step_number == step_number:
        step.status = status
        step.error_message = error_message
        break

  def get_pending_steps(self) -> list[GenerationStep]:
    """Get all steps that are still pending."""
    return [s for s in self.steps if s.status == StepStatus.PENDING]

  def get_summary(self) -> dict[str, int]:
    """Get count of steps by status."""
    summary: dict[str, int] = {"pending": 0, "generated": 0, "error": 0}
    for step in self.steps:
      summary[step.status.value] += 1
    return summary


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
  """
  In-memory grid of quadrant states.

  This class manages the state of all quadrants within a bounding box
  and helps construct generation plans.
  """

  def __init__(self, bounds: BoundingBox):
    self.bounds = bounds
    self._states: dict[Point, QuadrantState] = {}

    # Initialize all quadrants as empty
    for p in bounds.all_points():
      self._states[p] = QuadrantState.EMPTY

  def get_state(self, p: Point) -> QuadrantState:
    """Get the state of a quadrant."""
    return self._states.get(p, QuadrantState.EMPTY)

  def set_state(self, p: Point, state: QuadrantState) -> None:
    """Set the state of a quadrant."""
    if p in self._states:
      self._states[p] = state

  def is_generated(self, p: Point) -> bool:
    """Check if a quadrant has been generated."""
    return self.get_state(p) == QuadrantState.GENERATED

  def mark_generated(self, p: Point) -> None:
    """Mark a quadrant as generated."""
    self.set_state(p, QuadrantState.GENERATED)

  def mark_multiple_generated(self, points: list[Point]) -> None:
    """Mark multiple quadrants as generated."""
    for p in points:
      self.mark_generated(p)

  def get_all_generated(self) -> list[Point]:
    """Get all generated quadrant positions."""
    return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

  def get_all_empty(self) -> list[Point]:
    """Get all empty (not generated) quadrant positions."""
    return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

  def get_generated_bounds(self) -> BoundingBox | None:
    """Get the bounding box of all generated quadrants."""
    generated = self.get_all_generated()
    if not generated:
      return None

    min_x = min(p.x for p in generated)
    max_x = max(p.x for p in generated)
    min_y = min(p.y for p in generated)
    max_y = max(p.y for p in generated)

    return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

  def has_generated_neighbor(self, p: Point) -> bool:
    """Check if a quadrant has any generated neighbors (4-connected)."""
    neighbors = [
      Point(p.x - 1, p.y),
      Point(p.x + 1, p.y),
      Point(p.x, p.y - 1),
      Point(p.x, p.y + 1),
    ]
    return any(self.is_generated(n) for n in neighbors)

  def count_generated_neighbors(self, p: Point) -> int:
    """Count how many generated neighbors a quadrant has (4-connected)."""
    neighbors = [
      Point(p.x - 1, p.y),
      Point(p.x + 1, p.y),
      Point(p.x, p.y - 1),
      Point(p.x, p.y + 1),
    ]
    return sum(1 for n in neighbors if self.is_generated(n))

  def visualize(
    self,
    highlight: list[Point] | None = None,
    step_number: int | None = None,
  ) -> str:
    """
    Create an ASCII visualization of the grid.

    Legend:
      G = Generated
      . = Empty
      S = Selected (highlighted)
    """
    lines = []
    if step_number is not None:
      lines.append(f"Step {step_number}:")
    else:
      lines.append("Current state:")

    highlight_set = set(highlight or [])

    # Header with x coordinates
    x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
    header = "   " + " ".join(f"{x:2d}" for x in x_range)
    lines.append(header)
    lines.append("   " + "-" * (len(x_range) * 3 - 1))

    for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
      row = f"{y:2d}|"
      for x in x_range:
        p = Point(x, y)
        if p in highlight_set:
          char = " S"
        elif self.is_generated(p):
          char = " G"
        else:
          char = " ."
        row += char + " "
      lines.append(row)

    return "\n".join(lines)


# =============================================================================
# Database Operations
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


# =============================================================================
# Generation Step Execution
# =============================================================================


def run_generation_step(
  conn: sqlite3.Connection,
  config: dict,
  quadrant_tuples: list[tuple[int, int]],
  generation_dir: Path,
  port: int,
  bucket_name: str = "isometric-nyc-infills",
) -> dict:
  """
  Run a single generation step for the given quadrants.

  This uses the flexible TemplateBuilder approach that can handle
  1, 2, or 4 quadrant selections.

  Returns dict with success status and message/error.
  """
  import os
  import tempfile

  from dotenv import load_dotenv
  from PIL import Image

  from isometric_nyc.e2e_generation.infill_template import (
    QUADRANT_SIZE,
    InfillRegion,
    TemplateBuilder,
    validate_quadrant_selection,
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
    upload_to_gcs,
  )

  load_dotenv()

  # Check for API key
  api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
  if not api_key:
    return {
      "success": False,
      "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
    }

  # Helper functions
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = shared_get_quadrant_generation(conn, qx, qy)
    return gen is not None

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
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

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    quadrant_tuples, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    return {"success": False, "error": msg}

  print(f"   ✅ Validation: {msg}")

  # Get primary quadrants (the ones we selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(quadrant_tuples)

  # Build the template
  print("   🎨 Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    return {"success": False, "error": error_msg}

  template_image, placement = result

  # Save template to temp file and upload to GCS
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = Path(tmp.name)
    template_image.save(template_path)

  try:
    print("   📤 Uploading template to GCS...")
    image_url = upload_to_gcs(template_path, bucket_name)

    print("   🤖 Calling Oxen API...")
    generated_url = call_oxen_api(image_url, api_key)

    print("   📥 Downloading generated image...")
    generated_image = download_image_to_pil(generated_url)

    # Extract quadrants from generated image and save to database
    print("   💾 Saving generated quadrants to database...")

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
          print(f"      ✓ Saved generation for ({qx}, {qy})")
          saved_count += 1
        else:
          print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
      else:
        print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": primary_quadrants,
    }

  finally:
    # Clean up temp file
    template_path.unlink(missing_ok=True)


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  qx: int,
  qy: int,
  port: int,
) -> bytes | None:
  """Render a single quadrant using the web server."""
  from urllib.parse import urlencode

  from PIL import Image
  from playwright.sync_api import sync_playwright

  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    image_to_png_bytes,
    save_quadrant_render,
    split_tile_into_quadrants,
  )

  # Ensure the quadrant exists in the database
  ensure_quadrant_exists(conn, config, qx, qy)

  # Find the top-left of the tile containing this quadrant
  # Quadrants come in 2x2 tiles, so we need to find which tile this is in
  tile_x = (qx // 2) * 2
  tile_y = (qy // 2) * 2

  # Get the top-left quadrant of the tile
  tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

  # Build URL for rendering
  params = {
    "export": "true",
    "lat": tl_quadrant["lat"],
    "lon": tl_quadrant["lng"],
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
      print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

    screenshot = page.screenshot()

    page.close()
    context.close()
    browser.close()

  # Convert to PIL Image
  from io import BytesIO

  tile_image = Image.open(BytesIO(screenshot))

  # Split into quadrants
  quadrant_images = split_tile_into_quadrants(tile_image)

  # Save all 4 quadrants
  for (dx, dy), quad_img in quadrant_images.items():
    qx_save, qy_save = tile_x + dx, tile_y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

  # Return the specific quadrant we wanted
  dx = qx - tile_x
  dy = qy - tile_y
  return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
  """Call the Oxen API to generate pixel art."""
  import requests

  endpoint = "https://hub.oxen.ai/api/images/edit"
  model = "cannoneyed-gentle-gold-antlion"
  prompt = (
    "Fill in the outlined section with the missing pixels corresponding to "
    "the <isometric nyc pixel art> style, removing the border and exactly "
    "following the shape/style/structure of the surrounding image (if present)."
  )

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": model,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": 28,
  }

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


def download_image_to_pil(
  url: str,
  max_retries: int = 3,
  retry_delay: float = 10.0,
):
  """
  Download an image from a URL and return as PIL Image.

  Includes retry logic for transient errors (e.g., 403 Forbidden when
  the image is not yet available).

  Args:
      url: URL of the image to download
      max_retries: Maximum number of retry attempts (default: 3)
      retry_delay: Seconds to wait between retries (default: 10.0)

  Returns:
      PIL Image object

  Raises:
      requests.HTTPError: If all retry attempts fail
  """
  import time
  from io import BytesIO

  import requests
  from PIL import Image

  last_error = None

  for attempt in range(1, max_retries + 1):
    try:
      response = requests.get(url, timeout=120)
      response.raise_for_status()
      return Image.open(BytesIO(response.content))
    except requests.exceptions.HTTPError as e:
      last_error = e
      if attempt < max_retries:
        print(
          f"   ⚠️  Download failed (attempt {attempt}/{max_retries}): {e}"
        )
        print(f"   ⏳ Waiting {retry_delay}s before retrying...")
        time.sleep(retry_delay)
      else:
        print(f"   ❌ Download failed after {max_retries} attempts: {e}")
    except requests.exceptions.RequestException as e:
      last_error = e
      if attempt < max_retries:
        print(
          f"   ⚠️  Download error (attempt {attempt}/{max_retries}): {e}"
        )
        print(f"   ⏳ Waiting {retry_delay}s before retrying...")
        time.sleep(retry_delay)
      else:
        print(f"   ❌ Download failed after {max_retries} attempts: {e}")

  # If we get here, all retries failed
  if last_error:
    raise last_error
  raise RuntimeError("Download failed with no error captured")


# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
  """
  Find empty quadrants that are inside the generated bounds.

  These need to be filled before expanding outward.
  """
  gen_bounds = grid.get_generated_bounds()
  if gen_bounds is None:
    return []

  gaps = []
  for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
    for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
      p = Point(x, y)
      if not grid.is_generated(p) and grid.bounds.contains(p):
        gaps.append(p)

  return gaps


def can_generate_2x2(
  grid: QuadrantGrid, top_left: Point, require_gap: bool = True
) -> bool:
  """
  Check if a 2x2 tile starting at top_left can be generated.

  A 2x2 tile can be generated if:
  - All 4 quadrants are empty (not already generated)
  - All 4 quadrants are within bounds
  - If require_gap is True (default): NO quadrant has a direct neighbor that's generated
    (to avoid seams), but there is generated content within 2 tiles
  - If require_gap is False: At least one quadrant has a generated neighbor

  The gap requirement prevents seams at tile boundaries. 2x2 tiles with gaps are
  bridged back using 1x2/2x1 tiles which handle seams better.
  """
  quadrants = [
    top_left,
    Point(top_left.x + 1, top_left.y),
    Point(top_left.x, top_left.y + 1),
    Point(top_left.x + 1, top_left.y + 1),
  ]

  # All must be empty
  if any(grid.is_generated(q) for q in quadrants):
    return False

  # All must be within bounds
  if not all(grid.bounds.contains(q) for q in quadrants):
    return False

  quadrant_set = set(quadrants)

  if require_gap:
    # Check that NO quadrant has a direct generated neighbor (to avoid seams)
    for q in quadrants:
      direct_neighbors = [
        Point(q.x - 1, q.y),
        Point(q.x + 1, q.y),
        Point(q.x, q.y - 1),
        Point(q.x, q.y + 1),
      ]
      for n in direct_neighbors:
        if n not in quadrant_set and grid.is_generated(n):
          # Has direct neighbor - would create seam
          return False

    # Check that there IS generated content within 2 tiles (so we can bridge later)
    for q in quadrants:
      # Check neighbors at distance 2
      two_away = [
        Point(q.x - 2, q.y),
        Point(q.x + 2, q.y),
        Point(q.x, q.y - 2),
        Point(q.x, q.y + 2),
        Point(q.x - 1, q.y - 1),
        Point(q.x + 1, q.y - 1),
        Point(q.x - 1, q.y + 1),
        Point(q.x + 1, q.y + 1),
      ]
      for n in two_away:
        if n not in quadrant_set and grid.is_generated(n):
          return True
    return False
  else:
    # Original behavior: at least one must have a generated neighbor (outside the 2x2)
    for q in quadrants:
      neighbors = [
        Point(q.x - 1, q.y),
        Point(q.x + 1, q.y),
        Point(q.x, q.y - 1),
        Point(q.x, q.y + 1),
      ]
      for n in neighbors:
        if n not in quadrant_set and grid.is_generated(n):
          return True
    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
  """
  Check if a 1x2 horizontal tile can be generated.

  Layout: [left][right]

  IMPORTANT: Generated neighbors can ONLY be on the LONG sides (top/bottom),
  NOT on the SHORT sides (left end of left, right end of right).
  This prevents seams since short-side pixels aren't included in the template.
  """
  right = Point(left.x + 1, left.y)

  if grid.is_generated(left) or grid.is_generated(right):
    return False
  if not grid.bounds.contains(left) or not grid.bounds.contains(right):
    return False

  # SHORT sides (ends) - these CANNOT have generated neighbors
  short_side_neighbors = [
    Point(left.x - 1, left.y),  # Left of left quadrant
    Point(right.x + 1, right.y),  # Right of right quadrant
  ]
  for n in short_side_neighbors:
    if grid.is_generated(n):
      return False  # Would create seam

  # LONG sides (top/bottom) - at least one must have generated neighbor
  long_side_neighbors = [
    Point(left.x, left.y - 1),  # Above left
    Point(left.x, left.y + 1),  # Below left
    Point(right.x, right.y - 1),  # Above right
    Point(right.x, right.y + 1),  # Below right
  ]
  for n in long_side_neighbors:
    if grid.is_generated(n):
      return True

  return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
  """
  Check if a 2x1 vertical tile can be generated.

  Layout:
    [top]
    [bottom]

  IMPORTANT: Generated neighbors can ONLY be on the LONG sides (left/right),
  NOT on the SHORT sides (above top, below bottom).
  This prevents seams since short-side pixels aren't included in the template.
  """
  bottom = Point(top.x, top.y + 1)

  if grid.is_generated(top) or grid.is_generated(bottom):
    return False
  if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
    return False

  # SHORT sides (ends) - these CANNOT have generated neighbors
  short_side_neighbors = [
    Point(top.x, top.y - 1),  # Above top quadrant
    Point(bottom.x, bottom.y + 1),  # Below bottom quadrant
  ]
  for n in short_side_neighbors:
    if grid.is_generated(n):
      return False  # Would create seam

  # LONG sides (left/right) - at least one must have generated neighbor
  long_side_neighbors = [
    Point(top.x - 1, top.y),  # Left of top
    Point(top.x + 1, top.y),  # Right of top
    Point(bottom.x - 1, bottom.y),  # Left of bottom
    Point(bottom.x + 1, bottom.y),  # Right of bottom
  ]
  for n in long_side_neighbors:
    if grid.is_generated(n):
      return True

  return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
  """Check if a single quadrant can be generated."""
  if grid.is_generated(p):
    return False
  if not grid.bounds.contains(p):
    return False
  return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(
  grid: QuadrantGrid, direction: str, require_gap: bool = True
) -> list[Point]:
  """
  Find all valid 2x2 tile positions along a direction from the generated region.

  Args:
      grid: The quadrant grid
      direction: One of "top", "bottom", "left", "right"
      require_gap: If True, 2x2 tiles must have a gap from existing content

  Returns list of top-left corners for valid 2x2 tiles.
  """
  gen_bounds = grid.get_generated_bounds()
  if gen_bounds is None:
    return []

  valid_positions = []

  # When require_gap is True, we look for tiles 2 rows/cols away (with 1 row/col gap)
  # When require_gap is False, we look for tiles 1 row/col away (directly adjacent)
  offset = 2 if require_gap else 1

  if direction == "top":
    # Look for 2x2 tiles above the current bounds
    y = gen_bounds.top_left.y - offset - 1  # -1 because 2x2 tile has height 2
    for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
        valid_positions.append(tl)

  elif direction == "bottom":
    # Look for 2x2 tiles below the current bounds
    y = gen_bounds.bottom_right.y + offset
    for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
        valid_positions.append(tl)

  elif direction == "left":
    # Look for 2x2 tiles to the left of current bounds
    x = gen_bounds.top_left.x - offset - 1  # -1 because 2x2 tile has width 2
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
        valid_positions.append(tl)

  elif direction == "right":
    # Look for 2x2 tiles to the right of current bounds
    x = gen_bounds.bottom_right.x + offset
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
        valid_positions.append(tl)

  return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
  """Get all 4 quadrants for a 2x2 tile starting at top_left."""
  return [
    top_left,
    Point(top_left.x + 1, top_left.y),
    Point(top_left.x, top_left.y + 1),
    Point(top_left.x + 1, top_left.y + 1),
  ]


def get_1x2_quadrants(left: Point) -> list[Point]:
  """Get both quadrants for a 1x2 horizontal tile."""
  return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
  """Get both quadrants for a 2x1 vertical tile."""
  return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
  """
  Create an optimal generation plan to fill all empty quadrants.

  Strategy:
  1. Fill any interior gaps first (within current generated bounds)
  2. Expand outward in spiral pattern (top, right, bottom, left)
  3. For each direction:
     a. Generate 2x2 tiles offset from the edge
     b. Bridge the offset tiles to the center
     c. Bridge the offset tiles to each other
     d. Fill remaining single-quadrant gaps
  """
  steps: list[GenerationStep] = []
  step_num = 1

  # Phase 1: Fill interior gaps
  interior_gaps = find_interior_gaps(grid)
  if interior_gaps:
    # Try to fill gaps efficiently using largest possible tiles
    gap_steps = fill_gaps_efficiently(grid, interior_gaps)
    for quadrants, desc in gap_steps:
      steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
      grid.mark_multiple_generated(quadrants)
      step_num += 1

  # Phase 2: Spiral expansion
  directions = ["top", "right", "bottom", "left"]
  direction_idx = 0
  max_iterations = 1000  # Safety limit

  while grid.get_all_empty() and max_iterations > 0:
    max_iterations -= 1
    made_progress = False

    # Try all 4 directions in order
    for _ in range(4):
      direction = directions[direction_idx]
      direction_idx = (direction_idx + 1) % 4

      # Step A: Generate offset 2x2 tiles in this direction
      offset_2x2 = find_best_2x2_tiles(grid, direction)
      for tl in offset_2x2:
        quadrants = get_2x2_quadrants(tl)
        steps.append(
          GenerationStep(step_num, quadrants, f"Offset 2x2 ({direction}): {tl}")
        )
        grid.mark_multiple_generated(quadrants)
        step_num += 1
        made_progress = True

    # Step B: Fill remaining gaps (bridges and single quadrants)
    # IMPORTANT: 2x2 tiles can NEVER touch existing generated content.
    # Only use 1x2, 2x1, or single tiles for bridging.
    remaining_empty = grid.get_all_empty()
    if remaining_empty:
      # Sort by how many generated neighbors they have (more = better)
      remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

      for p in remaining_empty:
        if grid.is_generated(p):
          continue

        # Try 1x2 horizontal
        for dx in [0, -1]:
          left = Point(p.x + dx, p.y)
          if can_generate_1x2_horizontal(grid, left):
            quadrants = get_1x2_quadrants(left)
            steps.append(
              GenerationStep(step_num, quadrants, f"Bridge 1x2 horizontal: {left}")
            )
            grid.mark_multiple_generated(quadrants)
            step_num += 1
            made_progress = True
            break
        else:
          # Try 2x1 vertical
          for dy in [0, -1]:
            top = Point(p.x, p.y + dy)
            if can_generate_2x1_vertical(grid, top):
              quadrants = get_2x1_quadrants(top)
              steps.append(
                GenerationStep(
                  step_num,
                  quadrants,
                  f"Bridge 2x1 vertical: {top}",
                )
              )
              grid.mark_multiple_generated(quadrants)
              step_num += 1
              made_progress = True
              break
          else:
            # Single quadrant as last resort
            if can_generate_single(grid, p):
              steps.append(GenerationStep(step_num, [p], f"Single quadrant: {p}"))
              grid.mark_generated(p)
              step_num += 1
              made_progress = True

    if not made_progress:
      # Check if we have disconnected empty regions
      empty = grid.get_all_empty()
      if empty:
        # Find an empty quadrant adjacent to the generated region
        for p in empty:
          if grid.has_generated_neighbor(p):
            # Generate it as a single
            steps.append(
              GenerationStep(step_num, [p], f"Single quadrant (fallback): {p}")
            )
            grid.mark_generated(p)
            step_num += 1
            made_progress = True
            break

        if not made_progress:
          # Truly stuck - there may be disconnected regions
          print(
            f"Warning: {len(empty)} quadrants cannot be reached from generated region"
          )
          break

  return steps


def fill_gaps_efficiently(
  grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
  """
  Fill interior gaps using 1x2, 2x1, or single tiles ONLY.

  IMPORTANT:
  - 2x2 tiles can NEVER touch existing generated content.
  - 1x2 tiles can only have generated neighbors on LONG sides (top/bottom)
  - 2x1 tiles can only have generated neighbors on LONG sides (left/right)
  - Interior gaps are often surrounded by generated content, so many
    1x2/2x1 configurations won't be valid due to short-side constraints.

  Returns list of (quadrants, description) tuples.
  """
  result = []
  gap_set = set(gaps)

  while gap_set:
    found = False

    # Sort gaps by how many generated neighbors they have (prioritize well-connected gaps)
    sorted_gaps = sorted(gap_set, key=lambda p: -grid.count_generated_neighbors(p))

    for p in sorted_gaps:
      if p not in gap_set:
        continue

      # Try 1x2 horizontal (use proper validation with short-side constraint)
      right = Point(p.x + 1, p.y)
      if right in gap_set and can_generate_1x2_horizontal(grid, p):
        result.append(([p, right], f"1x2 at {p}"))
        grid.mark_multiple_generated([p, right])
        gap_set -= {p, right}
        found = True
        break

      # Try 2x1 vertical (use proper validation with short-side constraint)
      bottom = Point(p.x, p.y + 1)
      if bottom in gap_set and can_generate_2x1_vertical(grid, p):
        result.append(([p, bottom], f"2x1 at {p}"))
        grid.mark_multiple_generated([p, bottom])
        gap_set -= {p, bottom}
        found = True
        break

      # Single quadrant as last resort
      if grid.has_generated_neighbor(p):
        result.append(([p], f"single at {p}"))
        grid.mark_generated(p)
        gap_set.discard(p)
        found = True
        break

    if not found:
      # Stuck - remaining gaps are unreachable
      print(f"Warning: {len(gap_set)} interior gaps unreachable")
      break

  return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
  """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
  parts = s.strip().replace(" ", "").split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid coordinate format: {s}")
  return Point(int(parts[0]), int(parts[1]))


def create_and_save_plan(
  conn: sqlite3.Connection,
  bounds: BoundingBox,
  generation_dir: Path,
  visualize: bool = False,
) -> GenerationPlan | None:
  """
  Create a generation plan and save it to automatic_plan.json.

  Returns the plan, or None if no generation is needed.
  """
  # Load current state
  generated = load_generated_quadrants(conn)
  print(f"\n📊 Currently generated: {len(generated)} quadrants")

  # Create grid
  grid = QuadrantGrid(bounds)
  for p in generated:
    if bounds.contains(p):
      grid.mark_generated(p)

  generated_in_bounds = len(grid.get_all_generated())
  empty_in_bounds = len(grid.get_all_empty())
  print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

  if empty_in_bounds == 0:
    print("\n✅ All quadrants in bounding box are already generated!")
    return None

  # Show initial state
  if visualize:
    print("\n" + grid.visualize())

  # Create plan
  print("\n🔧 Creating generation plan...")

  # Create a copy of the grid for planning
  plan_grid = QuadrantGrid(bounds)
  for p in generated:
    if bounds.contains(p):
      plan_grid.mark_generated(p)

  steps = create_generation_plan(plan_grid)

  print(f"\n📋 Generation plan: {len(steps)} steps")

  # Show plan summary
  total_quadrants = sum(len(step.quadrants) for step in steps)
  by_size: dict[int, int] = {}
  for step in steps:
    size = len(step.quadrants)
    by_size[size] = by_size.get(size, 0) + 1

  print(f"   Total quadrants to generate: {total_quadrants}")
  print("   Steps by tile size:")
  for size in sorted(by_size.keys(), reverse=True):
    label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
    print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

  # Create and save plan
  plan = GenerationPlan(
    created_at=datetime.now().isoformat(),
    bounds=bounds,
    steps=steps,
    generation_dir=str(generation_dir),
  )

  plan_path = generation_dir / "automatic_plan.json"
  plan.save(plan_path)

  # Show plan details if visualizing
  if visualize:
    print("\n🔍 Generation plan details:\n")
    viz_grid = QuadrantGrid(bounds)
    for p in generated:
      if bounds.contains(p):
        viz_grid.mark_generated(p)

    for step in steps:
      coords = ", ".join(str(q) for q in step.quadrants)
      print(f"Step {step.step_number}: {step.description}")
      print(f"  Quadrants: [{coords}]")
      print(viz_grid.visualize(step.quadrants, step.step_number))
      viz_grid.mark_multiple_generated(step.quadrants)
      print()

  return plan


def execute_plan(
  conn: sqlite3.Connection,
  plan: GenerationPlan,
  plan_path: Path,
  port: int,
  bucket: str,
  no_start_server: bool,
  max_steps: int | None = None,
) -> int:
  """
  Execute a generation plan, updating status as we go.

  Stops on first error.

  Returns exit code (0 for success, 1 for error).
  """
  # Import here to avoid circular imports
  from isometric_nyc.e2e_generation.shared import (
    WEB_DIR,
    get_generation_config,
    start_web_server,
  )

  # Get pending steps
  pending_steps = plan.get_pending_steps()
  if not pending_steps:
    print("\n✅ All steps in the plan are already complete!")
    summary = plan.get_summary()
    print(f"   Summary: {summary['generated']} generated, {summary['error']} errors")
    return 0

  if max_steps is not None:
    pending_steps = pending_steps[:max_steps]

  print("\n🚀 Executing generation plan...")
  print(f"   {len(pending_steps)} steps to execute")

  web_server = None
  generation_dir = Path(plan.generation_dir)

  try:
    if not no_start_server:
      web_server = start_web_server(WEB_DIR, port)

    config = get_generation_config(conn)

    for step in pending_steps:
      print(f"\n{'=' * 60}")
      print(f"Step {step.step_number}/{len(plan.steps)}: {step.description}")
      print("=" * 60)

      # Convert Points to tuples for the generation API
      quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

      try:
        result = run_generation_step(
          conn,
          config,
          quadrant_tuples,
          generation_dir,
          port,
          bucket,
        )
        if result.get("success"):
          print(f"✅ Step {step.step_number} complete: {result.get('message')}")
          plan.update_step_status(step.step_number, StepStatus.GENERATED)
          plan.save(plan_path)
        else:
          error_msg = result.get("error", "Unknown error")
          print(f"❌ Step {step.step_number} failed: {error_msg}")
          plan.update_step_status(step.step_number, StepStatus.ERROR, error_msg)
          plan.save(plan_path)
          print("\n⛔ Stopping execution due to error.")
          return 1
      except Exception as e:
        error_msg = str(e)
        print(f"❌ Step {step.step_number} failed: {error_msg}")
        plan.update_step_status(step.step_number, StepStatus.ERROR, error_msg)
        plan.save(plan_path)
        print("\n⛔ Stopping execution due to error.")
        return 1

  finally:
    if web_server:
      print("\n🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()

  # Print summary
  summary = plan.get_summary()
  print(f"\n{'=' * 60}")
  print("✅ Plan execution complete!")
  print(
    f"   Generated: {summary['generated']}, Pending: {summary['pending']}, Errors: {summary['error']}"
  )
  print("=" * 60)

  return 0


def main():
  parser = argparse.ArgumentParser(
    description="Automatically generate tiles to fill a bounding box.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )

  # Plan creation arguments (mutually exclusive with --plan-json)
  plan_group = parser.add_argument_group("plan creation")
  plan_group.add_argument(
    "--top-left",
    type=str,
    help="Top-left corner of bounding box (x,y)",
  )
  plan_group.add_argument(
    "--bottom-right",
    type=str,
    help="Bottom-right corner of bounding box (x,y)",
  )
  plan_group.add_argument(
    "--visualize",
    action="store_true",
    help="Show ASCII visualization of the plan",
  )

  # Plan execution arguments
  exec_group = parser.add_argument_group("plan execution")
  exec_group.add_argument(
    "--plan-json",
    type=Path,
    help="Path to an existing plan JSON file to execute",
  )
  exec_group.add_argument(
    "--port",
    type=int,
    default=5173,
    help="Web server port (default: 5173)",
  )
  exec_group.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  exec_group.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum number of steps to execute",
  )
  exec_group.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )

  args = parser.parse_args()

  # Load database
  generation_dir = args.generation_dir.resolve()
  db_path = generation_dir / "quadrants.db"

  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  conn = sqlite3.connect(db_path)

  try:
    # Mode 1: Execute existing plan
    if args.plan_json:
      plan_path = args.plan_json.resolve()
      if not plan_path.exists():
        print(f"❌ Error: Plan file not found: {plan_path}")
        return 1

      print(f"📄 Loading plan from {plan_path}")
      plan = GenerationPlan.load(plan_path)

      print(f"📦 Bounding box: {plan.bounds.top_left} to {plan.bounds.bottom_right}")
      print(f"📋 Total steps: {len(plan.steps)}")
      summary = plan.get_summary()
      print(
        f"   Status: {summary['pending']} pending, {summary['generated']} generated, {summary['error']} errors"
      )

      return execute_plan(
        conn,
        plan,
        plan_path,
        args.port,
        args.bucket,
        args.no_start_server,
        args.max_steps,
      )

    # Mode 2: Create new plan
    if not args.top_left or not args.bottom_right:
      print(
        "❌ Error: Either --plan-json or both --top-left and --bottom-right are required"
      )
      parser.print_help()
      return 1

    # Parse coordinates
    try:
      top_left = parse_coordinate(args.top_left)
      bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
      print(f"❌ Error: {e}")
      return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
      print("❌ Error: top-left must be above and to the left of bottom-right")
      return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    plan = create_and_save_plan(conn, bounds, generation_dir, args.visualize)

    if plan is None:
      return 0

    print("\n" + "=" * 60)
    print("📄 Plan created and saved to:")
    print(f"   {generation_dir / 'automatic_plan.json'}")
    print("\nTo execute the plan, run:")
    print(
      "   uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\"
    )
    print(f"     {generation_dir} \\")
    print(f"     --plan-json {generation_dir / 'automatic_plan.json'}")
    print("=" * 60)

    return 0

  finally:
    conn.close()


if __name__ == "__main__":
  exit(main())

```

I've made all the necessary changes. Here's a summary of what was implemented:

## Changes Made

### 1. Image Download Retry Logic (`generate_omni.py` and `automatic_generation.py`)

Updated `download_image_to_pil` in both files to:
- **Retry up to 3 times** when download fails
- **Wait 10 seconds** between each retry attempt
- Handle both `HTTPError` (like the 403 Forbidden) and general `RequestException`
- Log each retry attempt and the final failure

### 2. Generation Retry Logic (`app.py`)

Updated `process_queue_item_from_db` to:
- **Retry the entire generation up to 3 times** if it fails
- **Wait 5 seconds** between generation retries
- Update the status to "retrying" so the UI reflects the current state
- Log each retry attempt and the final failure

## How It Works

When a generation is triggered:

1. **First**, the image download will be attempted with retry logic:
   - If download fails → wait 10s → retry (up to 3 times)
   
2. **If download still fails** (or any other error in the generation pipeline):
   - The entire generation will be retried
   - Wait 5s → retry generation (up to 3 times)

This means in the worst case, you get:
- 3 download attempts × 3 generation attempts = 9 total download attempts before giving up

The 403 error you were seeing should now be handled gracefully with the 10-second wait giving the Oxen AI hub time to make the generated image available.

