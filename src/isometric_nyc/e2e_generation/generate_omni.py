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


def call_oxen_api(image_url: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image

  Returns:
      URL of the generated image

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  model_id = OMNI_WATER_MODEL_ID
  api_key = os.getenv("OXEN_OMNI_v04_WATER_API_KEY")

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
    "num_inference_steps": 28,
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
