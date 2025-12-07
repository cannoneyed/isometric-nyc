# Implement task 008: e2e pixel generation
_Exported on 12/6/2025 at 22:02:21 EST from Cursor (2.1.39)_

---

**User**

Implement the plan in @tasks/008_e2e_generartion_pixels.md 

---

**Cursor**

I'll implement the plan from the task file. Let me start by reading the relevant source files to understand the existing logic.

Let me also read the seed_tiles.py and check_tiles.py to understand the database schema and existing utilities:

Now I have a good understanding of the codebase. Let me create a plan and implement the solution.

```python
"""
Shared utilities for e2e generation scripts.

Contains common database operations, web server management, and image utilities.
"""

import io
import json
import math
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


# =============================================================================
# Web Server Management
# =============================================================================


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


# =============================================================================
# Database Operations
# =============================================================================


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render, generation
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "has_render": row[5] is not None,
    "has_generation": row[6] is not None,
    "render": row[5],
    "generation": row[6],
  }


def get_quadrant_render(conn: sqlite3.Connection, x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row else None


def get_quadrant_generation(conn: sqlite3.Connection, x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row else None


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def check_all_quadrants_generated(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have generations.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_generation"]:
      return False

  return True


def save_quadrant_render(
  conn: sqlite3.Connection, config: dict, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Creates the quadrant if it doesn't exist.
  Returns True if successful.
  """
  # Ensure the quadrant exists first
  ensure_quadrant_exists(conn, config, x, y)

  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def save_quadrant_generation(
  conn: sqlite3.Connection, config: dict, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save generation bytes for a quadrant at position (x, y).

  Creates the quadrant if it doesn't exist.
  Returns True if successful.
  """
  # Ensure the quadrant exists first
  ensure_quadrant_exists(conn, config, x, y)

  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET generation = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def ensure_quadrant_exists(
  conn: sqlite3.Connection, config: dict, x: int, y: int
) -> dict:
  """
  Ensure a quadrant exists at position (x, y), creating it if necessary.

  Returns the quadrant data.
  """
  cursor = conn.cursor()

  # Check if quadrant already exists
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render, generation
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()

  if row:
    return {
      "lat": row[0],
      "lng": row[1],
      "tile_row": row[2],
      "tile_col": row[3],
      "quadrant_index": row[4],
      "has_render": row[5] is not None,
      "has_generation": row[6] is not None,
    }

  # Quadrant doesn't exist - create it
  lat, lng = calculate_quadrant_lat_lng(config, x, y)

  # Calculate tile_row, tile_col, and quadrant_index
  # For a quadrant at (x, y), it could belong to multiple tiles due to overlap
  # We'll use the tile where this quadrant is the TL (index 0)
  tile_col = x
  tile_row = y
  quadrant_index = 0  # TL of its "primary" tile

  cursor.execute(
    """
    INSERT INTO quadrants (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (x, y, lat, lng, tile_row, tile_col, quadrant_index),
  )
  conn.commit()

  print(f"   📝 Created quadrant ({x}, {y}) at {lat:.6f}, {lng:.6f}")

  return {
    "lat": lat,
    "lng": lng,
    "tile_row": tile_row,
    "tile_col": tile_col,
    "quadrant_index": quadrant_index,
    "has_render": False,
    "has_generation": False,
  }


# =============================================================================
# Coordinate Calculations
# =============================================================================


def calculate_offset(
  lat_center: float,
  lon_center: float,
  shift_x_px: float,
  shift_y_px: float,
  view_height_meters: float,
  viewport_height_px: int,
  azimuth_deg: float,
  elevation_deg: float,
) -> tuple[float, float]:
  """
  Calculate the new lat/lon center after shifting the view by shift_x_px and shift_y_px.
  """
  meters_per_pixel = view_height_meters / viewport_height_px

  shift_right_meters = shift_x_px * meters_per_pixel
  shift_up_meters = shift_y_px * meters_per_pixel

  elev_rad = math.radians(elevation_deg)
  sin_elev = math.sin(elev_rad)

  if abs(sin_elev) < 1e-6:
    raise ValueError(f"Elevation {elevation_deg} is too close to 0/180.")

  delta_rot_x = shift_right_meters
  delta_rot_y = -shift_up_meters / sin_elev

  azimuth_rad = math.radians(azimuth_deg)
  cos_a = math.cos(azimuth_rad)
  sin_a = math.sin(azimuth_rad)

  delta_east_meters = delta_rot_x * cos_a + delta_rot_y * sin_a
  delta_north_meters = -delta_rot_x * sin_a + delta_rot_y * cos_a

  delta_lat = delta_north_meters / 111111.0
  delta_lon = delta_east_meters / (111111.0 * math.cos(math.radians(lat_center)))

  return lat_center + delta_lat, lon_center + delta_lon


def calculate_quadrant_lat_lng(
  config: dict, quadrant_x: int, quadrant_y: int
) -> tuple[float, float]:
  """
  Calculate the lat/lng anchor for a quadrant at position (quadrant_x, quadrant_y).

  The anchor is the bottom-right corner of the quadrant.
  For the TL quadrant (0, 0) of the seed tile, the anchor equals the seed lat/lng.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant step is half a tile (with tile_step=0.5, one tile step = one quadrant)
  # But quadrant positions are in quadrant units, not tile units
  # quadrant_x = tile_col + dx, quadrant_y = tile_row + dy
  # So shift in pixels = quadrant position * (tile_size * tile_step)
  quadrant_step_x_px = width_px * tile_step
  quadrant_step_y_px = height_px * tile_step

  shift_x_px = quadrant_x * quadrant_step_x_px
  shift_y_px = -quadrant_y * quadrant_step_y_px  # Negative because y increases downward

  return calculate_offset(
    seed_lat,
    seed_lng,
    shift_x_px,
    shift_y_px,
    view_height_meters,
    height_px,
    azimuth,
    elevation,
  )


# =============================================================================
# Image Utilities
# =============================================================================


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def png_bytes_to_image(png_bytes: bytes) -> Image.Image:
  """Convert PNG bytes to a PIL Image."""
  return Image.open(io.BytesIO(png_bytes))


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def stitch_quadrants_to_tile(
  quadrants: dict[tuple[int, int], Image.Image],
) -> Image.Image:
  """
  Stitch 4 quadrant images into a single tile image.

  Args:
    quadrants: Dict mapping (dx, dy) offset to the quadrant image:
      (0, 0) = top-left
      (1, 0) = top-right
      (0, 1) = bottom-left
      (1, 1) = bottom-right

  Returns:
    Combined tile image
  """
  # Get dimensions from one of the quadrants
  sample_quad = next(iter(quadrants.values()))
  quad_w, quad_h = sample_quad.size

  # Create combined image
  tile = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Place quadrants
  placements = {
    (0, 0): (0, 0),  # TL at top-left
    (1, 0): (quad_w, 0),  # TR at top-right
    (0, 1): (0, quad_h),  # BL at bottom-left
    (1, 1): (quad_w, quad_h),  # BR at bottom-right
  }

  for offset, pos in placements.items():
    if offset in quadrants:
      tile.paste(quadrants[offset], pos)

  return tile


def get_neighboring_generated_quadrants(
  conn: sqlite3.Connection, x: int, y: int
) -> dict[str, bytes | None]:
  """
  Get generated quadrant images from neighboring tiles.

  For a tile at (x, y), the neighbors are:
  - left: quadrants at (x-1, y) and (x-1, y+1)
  - above: quadrants at (x, y-1) and (x+1, y-1)
  - above-left: quadrant at (x-1, y-1)

  Returns dict with keys: 'left_top', 'left_bottom', 'above_left', 'above_right', 'corner'
  Values are PNG bytes or None if not available.
  """
  neighbors = {
    "left_top": get_quadrant_generation(conn, x - 1, y),
    "left_bottom": get_quadrant_generation(conn, x - 1, y + 1),
    "above_left": get_quadrant_generation(conn, x, y - 1),
    "above_right": get_quadrant_generation(conn, x + 1, y - 1),
    "corner": get_quadrant_generation(conn, x - 1, y - 1),
  }
  return neighbors


def has_any_neighbor_generations(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """Check if there are any generated neighbor quadrants for a tile at (x, y)."""
  neighbors = get_neighboring_generated_quadrants(conn, x, y)
  return any(v is not None for v in neighbors.values())

```

Now let me update `render_tile.py` to import from `shared.py` and create the `generate_tile.py` script:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import sqlite3
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  image_to_png_bytes,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    # Ensure the quadrant exists (creates it if necessary)
    quadrant = ensure_quadrant_exists(conn, config, x, y)

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
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
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

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

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Failed to save quadrant ({qx}, {qy})")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Now let me create the `generate_tile.py` script:

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_neighboring_generated_quadrants,
  get_quadrant,
  get_quadrant_render,
  has_any_neighbor_generations,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)


# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print(f"   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print(f"   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def create_infill_image_from_scratch(
  conn: sqlite3.Connection, x: int, y: int
) -> Image.Image:
  """
  Create an infill image when there are no neighbor generations.

  Simply stitches together the 4 rendered quadrants.
  """
  quadrants = {}

  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes is None:
        raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
      quadrants[(dx, dy)] = png_bytes_to_image(render_bytes)

  return stitch_quadrants_to_tile(quadrants)


def create_infill_image_with_neighbors(
  conn: sqlite3.Connection, x: int, y: int
) -> Image.Image:
  """
  Create an infill image using neighbor generations.

  The infill image is a 2x2 tile where:
  - Parts that have generated neighbors are filled with those generations
  - Parts that need generation are filled with renders

  For a tile at (x, y), the neighbor positions are:
  - Left neighbor: tile at (x-1, y) provides quadrants at (x-1, y) and (x-1, y+1)
  - Above neighbor: tile at (x, y-1) provides quadrants at (x, y-1) and (x+1, y-1)
  - Diagonal: tile at (x-1, y-1) provides corner at (x-1, y-1)

  The infill image should show:
  - Left edge: generated TR quadrants from left neighbor
  - Top edge: generated BL/BR quadrants from above neighbor
  - Center: rendered quadrants from current tile
  """
  neighbors = get_neighboring_generated_quadrants(conn, x, y)

  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # The infill image needs to be larger to include neighbor context
  # We'll create a 3x3 quadrant grid (1.5x the tile size in each direction)
  # to provide context from neighbors
  infill_w = quad_w * 3
  infill_h = quad_h * 3

  infill = Image.new("RGBA", (infill_w, infill_h), (0, 0, 0, 0))

  # Fill in neighbor generations where available
  # Top-left corner (x-1, y-1)
  if neighbors["corner"]:
    img = png_bytes_to_image(neighbors["corner"])
    infill.paste(img, (0, 0))

  # Top row: above neighbors (x, y-1) and (x+1, y-1)
  if neighbors["above_left"]:
    img = png_bytes_to_image(neighbors["above_left"])
    infill.paste(img, (quad_w, 0))

  if neighbors["above_right"]:
    img = png_bytes_to_image(neighbors["above_right"])
    infill.paste(img, (quad_w * 2, 0))

  # Left column: left neighbors (x-1, y) and (x-1, y+1)
  if neighbors["left_top"]:
    img = png_bytes_to_image(neighbors["left_top"])
    infill.paste(img, (0, quad_h))

  if neighbors["left_bottom"]:
    img = png_bytes_to_image(neighbors["left_bottom"])
    infill.paste(img, (0, quad_h * 2))

  # Center area: current tile renders (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (quad_w * (1 + dx), quad_h * (1 + dy)))

  return infill


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (with or without neighbors)
  4. Upload to GCS and call Oxen API
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist

  Returns:
    Path to the generated image, or None if skipped
  """
  load_dotenv()

  api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
  if not api_key:
    raise ValueError(
      "OXEN_INFILL_V02_API_KEY environment variable not set. "
      "Please add it to your .env file."
    )

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🎯 Generating tile at ({x}, {y})")
    print(f"{'='*60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip
    if not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    has_neighbors = has_any_neighbor_generations(conn, x, y)

    if has_neighbors:
      print("   Using neighbor generations for infill context")
      infill_image = create_infill_image_with_neighbors(conn, x, y)
    else:
      print("   No neighbors - generating from scratch")
      infill_image = create_infill_image_from_scratch(conn, x, y)

    # Save infill image to disk for upload
    infill_path = renders_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # Also save the stitched render
    render_path = renders_dir / f"{x}_{y}.png"
    render_image = create_infill_image_from_scratch(conn, x, y)
    render_image.save(render_path)
    print(f"   ✓ Saved stitched render to {render_path.name}")

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)
    generated_url = call_oxen_api(image_url, api_key)

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # If we used neighbors, we need to crop out just the center tile
    if has_neighbors:
      # The infill is 3x3 quadrants, we want the center 2x2
      sample_render = get_quadrant_render(conn, x, y)
      sample_img = png_bytes_to_image(sample_render)
      quad_w, quad_h = sample_img.size

      # Crop center tile from generation
      # The center starts at (quad_w, quad_h) and is 2*quad_w x 2*quad_h
      center_box = (quad_w, quad_h, quad_w * 3, quad_h * 3)
      generation_image = generation_image.crop(center_box)

      # Save the cropped generation
      generation_image.save(generation_path)
      print(f"   ✓ Cropped generation to tile size")

    # Split into quadrants and save
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
      else:
        print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")

    print(f"\n{'='*60}")
    print(f"✅ Generation complete!")
    print(f"   Render: {render_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'='*60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Now let me check for any linting errors in the files I created/modified:

Let me verify the updated `render_tile.py` is complete and correct:

Let me run a quick check to make sure the modules can be imported correctly:

All modules import correctly. Let me also update `check_tiles.py` to use the shared utilities for consistency:

```python
"""
Check that tiles fit together correctly by rendering a 2x2 grid and assembling them.

This script renders 4 tiles at positions (0,0), (0,1), (1,0), (1,1) and
assembles them into a single image to verify correct alignment.

Usage:
  uv run python src/isometric_nyc/e2e_generation/check_tiles.py <generation_dir>

The assembled image will be saved as renders/check_tiles_assembled.png
"""

import argparse
import sqlite3
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  get_quadrant,
  start_web_server,
)


def render_tile(
  page,
  quadrant: dict,
  config: dict,
  output_path: Path,
  port: int,
) -> bool:
  """Render a single tile using Playwright."""
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

  page.goto(url, wait_until="networkidle")

  try:
    page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
  except Exception:
    print("   ⚠️  Timeout waiting for tiles")

  page.screenshot(path=str(output_path))
  return True


def check_tiles(generation_dir: Path, port: int = DEFAULT_WEB_PORT) -> Path | None:
  """
  Render a 2x2 grid of tiles and assemble them.

  Renders tiles at (0,0), (1,0), (0,1), (1,1) and assembles into one image.
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    width = config["width_px"]
    height = config["height_px"]

    # Get all 4 quadrants
    # Grid layout:
    #   (0,0) | (1,0)
    #   ------+------
    #   (0,1) | (1,1)
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    quadrants = {}

    print("📍 Finding quadrants...")
    for x, y in positions:
      q = get_quadrant(conn, x, y)
      if not q:
        print(f"   ❌ No quadrant at ({x}, {y})")
        return None
      quadrants[(x, y)] = q
      print(f"   ✓ ({x}, {y}): {q['lat']:.6f}, {q['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Render all tiles
    print("\n🎨 Rendering tiles...")
    tile_paths = {}

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
        viewport={"width": width, "height": height},
        device_scale_factor=1,
      )
      page = context.new_page()

      for x, y in positions:
        output_path = renders_dir / f"check_{x}_{y}.png"
        print(f"   Rendering ({x}, {y})...")
        render_tile(page, quadrants[(x, y)], config, output_path, port)
        tile_paths[(x, y)] = output_path
        print(f"   ✓ Saved to {output_path.name}")

      page.close()
      context.close()
      browser.close()

    # Assemble tiles
    # Each tile overlaps by 50%, so we place them offset by half the tile size
    print("\n🧩 Assembling tiles...")

    half_w = width // 2
    half_h = height // 2

    # The assembled image shows the overlapping region
    # For 4 quadrants with 50% overlap:
    # - Total width = width + half_w = 1.5 * width
    # - Total height = height + half_h = 1.5 * height
    assembled_width = width + half_w
    assembled_height = height + half_h

    assembled = Image.new("RGB", (assembled_width, assembled_height))

    # Place tiles at their offset positions
    # (0,0) at top-left: (0, 0)
    # (1,0) at top-right offset: (half_w, 0)
    # (0,1) at bottom-left offset: (0, half_h)
    # (1,1) at bottom-right offset: (half_w, half_h)
    placements = {
      (0, 0): (0, 0),
      (1, 0): (half_w, 0),
      (0, 1): (0, half_h),
      (1, 1): (half_w, half_h),
    }

    for (x, y), (px, py) in placements.items():
      tile_img = Image.open(tile_paths[(x, y)])
      assembled.paste(tile_img, (px, py))
      print(f"   Placed ({x}, {y}) at ({px}, {py})")

    # Save assembled image
    output_path = renders_dir / "check_tiles_assembled.png"
    assembled.save(output_path)
    print(f"\n✅ Assembled image saved to {output_path}")
    print(f"   Size: {assembled_width}x{assembled_height}")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Check tiles fit together by rendering and assembling a 2x2 grid."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
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

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = check_tiles(generation_dir, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Let me verify the check_tiles.py works and also read it to make sure it's complete:

Let me verify all modules still work correctly:

Let me run the linter to make sure everything is properly formatted:

I've implemented the plan from `tasks/008_e2e_generartion_pixels.md`. Here's what was created:

## Summary

### 1. Created `shared.py` with common utilities

This file contains shared functionality extracted from `render_tile.py` and used across multiple scripts:

- **Web server management**: `start_web_server()`, `wait_for_server()`
- **Database operations**: `get_generation_config()`, `get_quadrant()`, `ensure_quadrant_exists()`, `save_quadrant_render()`, `save_quadrant_generation()`, `check_all_quadrants_rendered()`, `check_all_quadrants_generated()`
- **Coordinate calculations**: `calculate_offset()`, `calculate_quadrant_lat_lng()`
- **Image utilities**: `image_to_png_bytes()`, `png_bytes_to_image()`, `split_tile_into_quadrants()`, `stitch_quadrants_to_tile()`
- **Neighbor utilities**: `get_neighboring_generated_quadrants()`, `has_any_neighbor_generations()`

### 2. Updated `render_tile.py`

Refactored to import shared utilities from `shared.py`, reducing code duplication.

### 3. Created `generate_tile.py`

The main new script that handles the full pixel art generation pipeline:

**Usage:**
```bash
uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>
```

**Logic flow:**
1. **Check/render quadrants**: If any of the 4 quadrants for the tile don't have renders, it renders the tile using the web server and saves quadrant renders to the database
2. **Check generations**: If all quadrants already have generations (and not using `--overwrite`), skip
3. **Create infill image**:
   - If no neighbor generations exist → stitch together the 4 rendered quadrants
   - If neighbor generations exist → create a 3x3 quadrant infill image with neighbor context
4. **Call Oxen API**: Upload infill to GCS, call the Oxen model, download result
5. **Save generations**: Split the generation into quadrants and save to both database and filesystem

**Options:**
- `--port`: Web server port (default: 5173)
- `--bucket`: GCS bucket name (default: isometric-nyc-infills)
- `--no-start-server`: Don't start web server (assume it's running)
- `--overwrite`: Regenerate even if generations already exist

### 4. Updated `check_tiles.py`

Refactored to use shared utilities for consistency.

All files pass linting and import checks. The implementation follows the existing patterns in the codebase and integrates with the quadrants database schema.

```shellscript
uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>
```

---

**User**

OK - while we wait for the models to load, can we create a simple web app that loads then displays a 2x2 grid of *generated* tiles starting at a given x y? If no generated data exists, it should show a blank light gray placeholder in its place. Just a simple python server / web app please

---

**Cursor**

I'll create a simple Python web app to display generated tiles. Let me check if Flask is available and create the app:

Now let me create the web app:

```python
"""
Simple web app to view generated tiles in a 2x2 grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.
"""

import argparse
import base64
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .grid-container {
      display: inline-block;
      border: 2px solid #333;
      border-radius: 8px;
      overflow: hidden;
    }
    
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 2px;
      background: #333;
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
  </div>
  
  <div class="grid-container">
    <div class="grid">
      {% for dy in range(2) %}
        {% for dx in range(2) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing quadrants ({{ x }}, {{ y }}) through ({{ x + 1 }}, {{ y + 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function goTo() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      window.location.href = `?x=${x}&y=${y}`;
    }
    
    function navigate(dx, dy) {
      const x = parseInt(document.getElementById('x').value) + dx;
      const y = parseInt(document.getElementById('y').value) + dy;
      window.location.href = `?x=${x}&y=${y}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing 2x2 grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)

  # Check which tiles have generations
  tiles = {}
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(
    description="View generated tiles in a 2x2 grid."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print(f"🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print(f"   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

Let me verify it imports correctly:

The app is ready. Let me start it for you:

The server is running! Let me check how it looks:

The web app is running at **http://127.0.0.1:8080/**. As you can see, it's showing a 2x2 grid of placeholders since no generated tiles exist yet.

**Features:**
- 🎯 **Coordinate inputs** - Enter X and Y to jump to any position
- 🔘 **Navigation buttons** - Left, Right, Up, Down to move around the grid
- ⌨️ **Keyboard navigation** - Arrow keys also work
- 📍 **Coordinate labels** - Each tile shows its (x, y) position
- 🟪 **Placeholders** - Gray/purple boxes for missing generations, images for generated tiles

**Usage:**
```bash
uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>
```

Once you have generated tiles in the database, they'll display as actual images instead of the placeholder boxes. The server is currently running on port 8080 pointing to `generations/test_generation`.

```shellscript
uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>
```

---

**User**

Amazing! Can you add two details to the tiles viewer app -



1. A url parameter/control for how many n by n quadrants to show (default 2)

2. A toggle for turning on/off the lines/xy coords?

---

**Cursor**

```python
"""
Simple web app to view generated tiles in a 2x2 grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
      background: #333;
    }
    
    .grid-container.show-lines .grid {
      gap: 2px;
    }
    
    .grid-container:not(.show-lines) .grid {
      gap: 0;
      background: transparent;
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-lines) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines/Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing 2x2 grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)

  # Check which tiles have generations
  tiles = {}
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

Now let me update the route handler to support the new parameters:

```python
"""
Simple web app to view generated tiles in a 2x2 grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
      background: #333;
    }
    
    .grid-container.show-lines .grid {
      gap: 2px;
    }
    
    .grid-container:not(.show-lines) .grid {
      gap: 0;
      background: transparent;
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-lines) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines/Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines and coords: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines/coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
      background: #333;
    }
    
    .grid-container.show-lines .grid {
      gap: 2px;
    }
    
    .grid-container:not(.show-lines) .grid {
      gap: 0;
      background: transparent;
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-lines) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines/Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

Let me restart the server for the changes to take effect:

The generated tiles are showing! Let me enable the lines/coords to show how that toggle works:

The updates are working. Here's what was added:

**New Features:**

1. **N×N Grid Size Control**
   - New `N:` input field in the controls
   - URL parameter: `?n=3` (default: 2, max: 10)
   - Shows any grid size from 1×1 to 10×10

2. **Lines/Coords Toggle**
   - Checkbox to show/hide grid lines and coordinate labels
   - URL parameter: `?lines=1` (on) or `?lines=0` (off)
   - Keyboard shortcut: Press `L` to toggle
   - When off: seamless tile view with no borders or labels
   - When on: shows grid borders and (x, y) coordinates on each tile

**Example URLs:**
- `http://localhost:8080/?x=0&y=0&n=4&lines=1` - 4×4 grid with lines
- `http://localhost:8080/?x=0&y=0&n=3&lines=0` - 3×3 grid, seamless

As you can see in the screenshot, you already have some rendered tiles in the database! Tiles (0,0), (1,0), (0,1), (1,1) have data, while (2,0), (2,1), etc. show as gray placeholders.

---

**User**

Ok - it seems like the logic for generating the infill image with a neighbor isn't right. In the case of running e.g. generate_tile 0 1, the output image isn't correct - it needs to make an image consisting of any generated quadrants, and then rendered quadrants for the rest, with a 1px red border around the entire rendered part. Can you fix the behavior and also add a param to the script that bypasses the actual model inference call and just saves the infill so we can debug?

---

**Cursor**

Let me read the current generate_tile.py to understand the logic and fix it:

I see the issue. The current logic is looking at quadrants *outside* the tile for neighbors, but with overlapping tiles, some quadrants *inside* the current tile may already have generations. Let me fix this:

Now let me rewrite the infill creation logic and add a debug mode:

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_neighboring_generated_quadrants,
  get_quadrant_render,
  has_any_neighbor_generations,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (with or without neighbors)
  4. Upload to GCS and call Oxen API
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist

  Returns:
    Path to the generated image, or None if skipped
  """
  load_dotenv()

  api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
  if not api_key:
    raise ValueError(
      "OXEN_INFILL_V02_API_KEY environment variable not set. "
      "Please add it to your .env file."
    )

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})")
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip
    if not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    has_neighbors = has_any_neighbor_generations(conn, x, y)

    if has_neighbors:
      print("   Using neighbor generations for infill context")
      infill_image = create_infill_image_with_neighbors(conn, x, y)
    else:
      print("   No neighbors - generating from scratch")
      infill_image = create_infill_image_from_scratch(conn, x, y)

    # Save infill image to disk for upload
    infill_path = renders_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # Also save the stitched render
    render_path = renders_dir / f"{x}_{y}.png"
    render_image = create_infill_image_from_scratch(conn, x, y)
    render_image.save(render_path)
    print(f"   ✓ Saved stitched render to {render_path.name}")

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)
    generated_url = call_oxen_api(image_url, api_key)

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # If we used neighbors, we need to crop out just the center tile
    if has_neighbors:
      # The infill is 3x3 quadrants, we want the center 2x2
      sample_render = get_quadrant_render(conn, x, y)
      sample_img = png_bytes_to_image(sample_render)
      quad_w, quad_h = sample_img.size

      # Crop center tile from generation
      # The center starts at (quad_w, quad_h) and is 2*quad_w x 2*quad_h
      center_box = (quad_w, quad_h, quad_w * 3, quad_h * 3)
      generation_image = generation_image.crop(center_box)

      # Save the cropped generation
      generation_image.save(generation_path)
      print("   ✓ Cropped generation to tile size")

    # Split into quadrants and save
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
      else:
        print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Render: {render_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Now let me update the main `generate_tile` function to use the new logic and add the debug flag:

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  has_any_neighbor_generations,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(f"\n   Drawing red border around {len(render_positions)} render quadrant(s)")
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = renders_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)
    generated_url = call_oxen_api(image_url, api_key)

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Now let me add the `--debug` argument to main():

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  has_any_neighbor_generations,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(f"\n   Drawing red border around {len(render_positions)} render quadrant(s)")
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = renders_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)
    generated_url = call_oxen_api(image_url, api_key)

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode: skip model inference and just save the infill image",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  has_any_neighbor_generations,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(f"\n   Drawing red border around {len(render_positions)} render quadrant(s)")
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = renders_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)
    generated_url = call_oxen_api(image_url, api_key)

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode: skip model inference and just save the infill image",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
      args.debug,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Let me clean up the imports since some are no longer used:

```python
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)
```

Let me verify the current state of the file:

Let me verify the module still imports correctly:

Let me test the debug mode:

All quadrants at (0, 1) already have generations. Let me try a position that needs generation:

---

**User**

This is wrong - the top two quadrant generations *should* exist... and even if they don't, then the missing quadrants need to be filled with the web render.

---

**Cursor**

I see the issue. Looking at the image, only the top-left quadrant has content - the rest is black. This suggests that quadrants (2,0), (1,1), and (2,1) don't have renders in the database, and with `--no-start-server` we can't render them.

Let me fix the logic to:
1. Better handle the case where renders are missing
2. Show a clearer error when renders are needed but server isn't running

Let me check what's actually in the database and renders directory:

---

**User**

OK - I just looked into it and somehow the db has been corrupted... the quadrant in 0, 1 is wrong... we need clear out these tiles. Can you write a script that removes generations from specific coordinate generations?

---

**Cursor**

```python
"""
Clear generations from specific quadrant coordinates.

Usage:
  # Clear a single quadrant
  uv run python src/isometric_nyc/e2e_generation/clear_generations.py <generation_dir> <x> <y>

  # Clear a tile (all 4 quadrants)
  uv run python src/isometric_nyc/e2e_generation/clear_generations.py <generation_dir> <x> <y> --tile

  # Clear all generations
  uv run python src/isometric_nyc/e2e_generation/clear_generations.py <generation_dir> --all

  # Dry run (show what would be cleared)
  uv run python src/isometric_nyc/e2e_generation/clear_generations.py <generation_dir> <x> <y> --dry-run
"""

import argparse
import sqlite3
from pathlib import Path


def get_quadrant_info(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get info about a quadrant."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT quadrant_x, quadrant_y, 
           render IS NOT NULL as has_render,
           generation IS NOT NULL as has_gen
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None
  return {
    "x": row[0],
    "y": row[1],
    "has_render": bool(row[2]),
    "has_generation": bool(row[3]),
  }


def clear_generation(
  conn: sqlite3.Connection, x: int, y: int, dry_run: bool = False
) -> bool:
  """Clear generation for a single quadrant."""
  info = get_quadrant_info(conn, x, y)
  if not info:
    print(f"   ⚠️  Quadrant ({x}, {y}) not found in database")
    return False

  if not info["has_generation"]:
    print(f"   ⏭️  Quadrant ({x}, {y}) has no generation to clear")
    return False

  if dry_run:
    print(f"   🔍 Would clear generation for quadrant ({x}, {y})")
    return True

  cursor = conn.cursor()
  cursor.execute(
    "UPDATE quadrants SET generation = NULL WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  conn.commit()
  print(f"   ✓ Cleared generation for quadrant ({x}, {y})")
  return True


def clear_tile_generations(
  conn: sqlite3.Connection, x: int, y: int, dry_run: bool = False
) -> int:
  """Clear generations for all 4 quadrants in a tile."""
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
  cleared = 0
  for qx, qy in positions:
    if clear_generation(conn, qx, qy, dry_run):
      cleared += 1
  return cleared


def clear_all_generations(conn: sqlite3.Connection, dry_run: bool = False) -> int:
  """Clear all generations in the database."""
  cursor = conn.cursor()

  # Count how many have generations
  cursor.execute("SELECT COUNT(*) FROM quadrants WHERE generation IS NOT NULL")
  count = cursor.fetchone()[0]

  if count == 0:
    print("   ⏭️  No generations to clear")
    return 0

  if dry_run:
    print(f"   🔍 Would clear {count} generation(s)")
    return count

  cursor.execute("UPDATE quadrants SET generation = NULL")
  conn.commit()
  print(f"   ✓ Cleared {count} generation(s)")
  return count


def show_status(conn: sqlite3.Connection) -> None:
  """Show the current status of all quadrants."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT quadrant_x, quadrant_y,
           render IS NOT NULL as has_render,
           generation IS NOT NULL as has_gen
    FROM quadrants
    ORDER BY quadrant_y, quadrant_x
    """
  )
  rows = cursor.fetchall()

  print("\n📊 Quadrant Status:")
  print("   " + "-" * 40)
  print(f"   {'Coord':<10} {'Render':<10} {'Generation':<10}")
  print("   " + "-" * 40)

  for row in rows:
    x, y, has_render, has_gen = row
    render_str = "✓" if has_render else "✗"
    gen_str = "✓" if has_gen else "✗"
    print(f"   ({x}, {y}){'':<4} {render_str:<10} {gen_str:<10}")

  print("   " + "-" * 40)

  # Summary
  cursor.execute("SELECT COUNT(*) FROM quadrants")
  total = cursor.fetchone()[0]
  cursor.execute("SELECT COUNT(*) FROM quadrants WHERE render IS NOT NULL")
  renders = cursor.fetchone()[0]
  cursor.execute("SELECT COUNT(*) FROM quadrants WHERE generation IS NOT NULL")
  gens = cursor.fetchone()[0]

  print(f"\n   Total quadrants: {total}")
  print(f"   With renders: {renders}")
  print(f"   With generations: {gens}")


def main():
  parser = argparse.ArgumentParser(
    description="Clear generations from specific quadrant coordinates."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    nargs="?",
    help="Quadrant x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    nargs="?",
    help="Quadrant y coordinate",
  )
  parser.add_argument(
    "--tile",
    action="store_true",
    help="Clear all 4 quadrants for the tile starting at (x, y)",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Clear ALL generations in the database",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be cleared without actually clearing",
  )
  parser.add_argument(
    "--status",
    action="store_true",
    help="Show status of all quadrants",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  conn = sqlite3.connect(db_path)

  try:
    # Show status if requested
    if args.status:
      show_status(conn)
      return 0

    # Clear all if requested
    if args.all:
      print(f"\n🗑️  Clearing ALL generations...")
      if args.dry_run:
        print("   (DRY RUN - no changes will be made)")
      clear_all_generations(conn, args.dry_run)
      return 0

    # Need coordinates for single/tile clear
    if args.x is None or args.y is None:
      print("❌ Error: x and y coordinates required (or use --all)")
      return 1

    if args.tile:
      print(f"\n🗑️  Clearing tile at ({args.x}, {args.y})...")
      print(f"   Quadrants: ({args.x},{args.y}), ({args.x+1},{args.y}), "
            f"({args.x},{args.y+1}), ({args.x+1},{args.y+1})")
      if args.dry_run:
        print("   (DRY RUN - no changes will be made)")
      cleared = clear_tile_generations(conn, args.x, args.y, args.dry_run)
      print(f"\n   Total cleared: {cleared}")
    else:
      print(f"\n🗑️  Clearing quadrant ({args.x}, {args.y})...")
      if args.dry_run:
        print("   (DRY RUN - no changes will be made)")
      clear_generation(conn, args.x, args.y, args.dry_run)

    return 0

  finally:
    conn.close()


if __name__ == "__main__":
  exit(main())

```

Done! Here's the new `clear_generations.py` script:

**Usage examples:**

```bash
# Show status of all quadrants (renders/generations)
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation --status

# Clear a single quadrant's generation
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation 0 1

# Clear all 4 quadrants for a tile (e.g., tile at 0,0 covers quadrants 0,0 / 1,0 / 0,1 / 1,1)
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation 0 0 --tile

# Clear ALL generations in the database
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation --all

# Dry run (see what would be cleared without actually clearing)
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation 0 1 --dry-run
```

You can use `--status` first to see the current state of all quadrants, then clear the specific ones that are corrupted.

```shellscript
# Show status of all quadrants (renders/generations)
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation --status

# Clear a single quadrant's generation
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation 0 1

# Clear all 4 quadrants for a tile (e.g., tile at 0,0 covers quadrants 0,0 / 1,0 / 0,1 / 1,1)
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation 0 0 --tile

# Clear ALL generations in the database
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation --all

# Dry run (see what would be cleared without actually clearing)
uv run python src/isometric_nyc/e2e_generation/clear_generations.py generations/test_generation 0 1 --dry-run
```

---

**User**

OK - now I need a new script that takes a previously generated generation png and inserts the four quadrants into a specific tile address

---

**Cursor**

```python
"""
Import a generation PNG into the database at a specific tile address.

Takes a previously generated PNG image and splits it into 4 quadrants,
inserting them into the database at the specified tile coordinates.

Usage:
  uv run python src/isometric_nyc/e2e_generation/import_generation.py <generation_dir> <image_path> <x> <y>

Example:
  uv run python src/isometric_nyc/e2e_generation/import_generation.py generations/test_generation output.png 0 0
"""

import argparse
import io
import sqlite3
from pathlib import Path

from PIL import Image


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def split_into_quadrants(
  image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split an image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): image.crop((0, 0, half_w, half_h)),
    (1, 0): image.crop((half_w, 0, width, half_h)),
    (0, 1): image.crop((0, half_h, half_w, height)),
    (1, 1): image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def get_quadrant_info(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get info about a quadrant."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT quadrant_x, quadrant_y, 
           render IS NOT NULL as has_render,
           generation IS NOT NULL as has_gen
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None
  return {
    "x": row[0],
    "y": row[1],
    "has_render": bool(row[2]),
    "has_generation": bool(row[3]),
  }


def save_quadrant_generation(
  conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes
) -> bool:
  """Save generation bytes for a quadrant."""
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET generation = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def import_generation(
  generation_dir: Path,
  image_path: Path,
  x: int,
  y: int,
  overwrite: bool = False,
  dry_run: bool = False,
) -> bool:
  """
  Import a generation image into the database.

  Args:
    generation_dir: Path to the generation directory
    image_path: Path to the PNG image to import
    x: Tile x coordinate (top-left quadrant)
    y: Tile y coordinate (top-left quadrant)
    overwrite: If True, overwrite existing generations
    dry_run: If True, don't actually write to database

  Returns:
    True if successful
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

  # Load and split the image
  print(f"📷 Loading image: {image_path}")
  image = Image.open(image_path)
  print(f"   Size: {image.size[0]}x{image.size[1]}")

  if image.size[0] != image.size[1]:
    print(f"   ⚠️  Warning: Image is not square")

  quadrant_images = split_into_quadrants(image)
  quad_w, quad_h = quadrant_images[(0, 0)].size
  print(f"   Quadrant size: {quad_w}x{quad_h}")

  conn = sqlite3.connect(db_path)

  try:
    print(f"\n🎯 Importing to tile ({x}, {y})")
    print(f"   Quadrants: ({x},{y}), ({x+1},{y}), ({x},{y+1}), ({x+1},{y+1})")

    if dry_run:
      print("   (DRY RUN - no changes will be made)\n")

    # Process each quadrant
    success_count = 0
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy

      # Check if quadrant exists
      info = get_quadrant_info(conn, qx, qy)
      if not info:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database - skipping")
        continue

      # Check if it already has a generation
      if info["has_generation"] and not overwrite:
        print(f"   ⏭️  Quadrant ({qx}, {qy}) already has generation - skipping (use --overwrite)")
        continue

      # Convert to bytes
      png_bytes = image_to_png_bytes(quad_img)

      if dry_run:
        status = "overwrite" if info["has_generation"] else "new"
        print(f"   🔍 Would import quadrant ({qx}, {qy}) [{status}] - {len(png_bytes)} bytes")
        success_count += 1
      else:
        if save_quadrant_generation(conn, qx, qy, png_bytes):
          status = "overwrote" if info["has_generation"] else "imported"
          print(f"   ✓ {status.capitalize()} quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
          success_count += 1
        else:
          print(f"   ❌ Failed to save quadrant ({qx}, {qy})")

    print(f"\n{'🔍 Would import' if dry_run else '✅ Imported'} {success_count}/4 quadrants")
    return success_count > 0

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Import a generation PNG into the database at a specific tile address."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "image_path",
    type=Path,
    help="Path to the PNG image to import",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate (top-left quadrant)",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate (top-left quadrant)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing generations",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be imported without actually importing",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()
  image_path = args.image_path.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  try:
    result = import_generation(
      generation_dir,
      image_path,
      args.x,
      args.y,
      args.overwrite,
      args.dry_run,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

Done! Here's the new `import_generation.py` script:

**Usage:**

```bash
# Import a generation PNG to tile (0, 0)
uv run python src/isometric_nyc/e2e_generation/import_generation.py generations/test_generation path/to/image.png 0 0

# Overwrite existing generations
uv run python src/isometric_nyc/e2e_generation/import_generation.py generations/test_generation path/to/image.png 0 0 --overwrite

# Dry run (see what would be imported)
uv run python src/isometric_nyc/e2e_generation/import_generation.py generations/test_generation path/to/image.png 0 0 --dry-run
```

The script will:
1. Load the PNG image
2. Split it into 4 quadrants (top-left, top-right, bottom-left, bottom-right)
3. Insert each quadrant into the database at the corresponding positions:
   - `(x, y)`, `(x+1, y)`, `(x, y+1)`, `(x+1, y+1)`
4. Skip quadrants that already have generations (unless `--overwrite` is used)

```shellscript
# Import a generation PNG to tile (0, 0)
uv run python src/isometric_nyc/e2e_generation/import_generation.py generations/test_generation path/to/image.png 0 0

# Overwrite existing generations
uv run python src/isometric_nyc/e2e_generation/import_generation.py generations/test_generation path/to/image.png 0 0 --overwrite

# Dry run (see what would be imported)
uv run python src/isometric_nyc/e2e_generation/import_generation.py generations/test_generation path/to/image.png 0 0 --dry-run
```

---

**User**

Please modify the @generate_tile.py script to save infill template images to an `infills` directory instead of to the `renders` directory

---

**Cursor**

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    infills_dir = generation_dir / "infills"
    infills_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(
        f"\n   Drawing red border around {len(render_positions)} render quadrant(s)"
      )
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = renders_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)
    generated_url = call_oxen_api(image_url, api_key)

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode: skip model inference and just save the infill image",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
      args.debug,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def call_oxen_api(image_url: str, api_key: str) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-odd-blue-marmot",  # V04 generation model
    "input_image": image_url,
    "prompt": "Convert the input image to <isometric nyc pixel art>",
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    infills_dir = generation_dir / "infills"
    infills_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(
        f"\n   Drawing red border around {len(render_positions)} render quadrant(s)"
      )
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = infills_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)
    generated_url = call_oxen_api(image_url, api_key)

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode: skip model inference and just save the infill image",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
      args.debug,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Done! Infill images will now be saved to `<generation_dir>/infills/` instead of `<generation_dir>/renders/`.

---

**User**

For @src/isometric_nyc/generate_tile.py @src/isometric_nyc/e2e_generation/generate_tile.py:102-103 make the prompt for infill models use the logic to determine which prompt variant it should use (same logic from @src/isometric_nyc/generate_tile_oxen_infill.py )

---

**Cursor**

I'll add the region description logic from `generate_tile_oxen_infill.py` to `generate_tile.py`:

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def get_region_description(render_positions: list[tuple[int, int]]) -> str:
  """
  Get a human-readable description of the rendered region(s).

  Args:
    render_positions: List of (dx, dy) positions that need generation.
      (0, 0) = TL, (1, 0) = TR, (0, 1) = BL, (1, 1) = BR

  Returns:
    Description like "right half", "bottom left quadrant", etc.
  """
  # Convert (dx, dy) to quadrant indices: 0=TL, 1=TR, 2=BL, 3=BR
  rendered_indices = {dy * 2 + dx for dx, dy in render_positions}

  # Check for half patterns
  if rendered_indices == {0, 2}:
    return "left half"
  if rendered_indices == {1, 3}:
    return "right half"
  if rendered_indices == {0, 1}:
    return "top half"
  if rendered_indices == {2, 3}:
    return "bottom half"

  # Check for single quadrants
  if rendered_indices == {0}:
    return "top left quadrant"
  if rendered_indices == {1}:
    return "top right quadrant"
  if rendered_indices == {2}:
    return "bottom left quadrant"
  if rendered_indices == {3}:
    return "bottom right quadrant"

  # Three quadrants (L-shapes)
  if rendered_indices == {0, 1, 2}:
    return "top half and bottom left quadrant"
  if rendered_indices == {0, 1, 3}:
    return "top half and bottom right quadrant"
  if rendered_indices == {0, 2, 3}:
    return "left half and bottom right quadrant"
  if rendered_indices == {1, 2, 3}:
    return "right half and bottom left quadrant"

  # Multiple non-adjacent quadrants (diagonal)
  if rendered_indices == {0, 3}:
    return "top left and bottom right quadrants"
  if rendered_indices == {1, 2}:
    return "top right and bottom left quadrants"

  # Fallback for any other combination
  quadrant_names = {
    0: "top left",
    1: "top right",
    2: "bottom left",
    3: "bottom right",
  }
  names = [quadrant_names[i] for i in sorted(rendered_indices)]
  return " and ".join(names) + " quadrants"


def call_oxen_api(
  image_url: str,
  api_key: str,
  *,
  use_infill_model: bool = False,
  region_description: str | None = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key
    use_infill_model: If True, use the infill model instead of generation model
    region_description: For infill model, describes which region to convert

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  if use_infill_model:
    model = "cannoneyed-satisfactory-harlequin-minnow"  # V04 infill model
    if region_description:
      prompt = (
        f"Convert the {region_description} of the image to isometric nyc pixel art "
        f"in precisely the style of the other part of the image."
      )
    else:
      prompt = "Convert the input image to <isometric nyc pixel art>"
  else:
    model = "cannoneyed-odd-blue-marmot"  # V04 generation model
    prompt = "Convert the input image to <isometric nyc pixel art>"

  payload = {
    "model": model,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Prompt: {payload['prompt']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    infills_dir = generation_dir / "infills"
    infills_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(
        f"\n   Drawing red border around {len(render_positions)} render quadrant(s)"
      )
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = infills_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)

    # We use the generation model if *all* quadrants must be generated (i.e. # len(render_positions) == 4)
    use_infill_model = len(render_positions) != 4
    generated_url = call_oxen_api(
      image_url=image_url,
      api_key=api_key,
      use_infill_model=use_infill_model,
    )

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode: skip model inference and just save the infill image",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
      args.debug,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Now let me update the call to pass the region description:

```python
"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def get_region_description(render_positions: list[tuple[int, int]]) -> str:
  """
  Get a human-readable description of the rendered region(s).

  Args:
    render_positions: List of (dx, dy) positions that need generation.
      (0, 0) = TL, (1, 0) = TR, (0, 1) = BL, (1, 1) = BR

  Returns:
    Description like "right half", "bottom left quadrant", etc.
  """
  # Convert (dx, dy) to quadrant indices: 0=TL, 1=TR, 2=BL, 3=BR
  rendered_indices = {dy * 2 + dx for dx, dy in render_positions}

  # Check for half patterns
  if rendered_indices == {0, 2}:
    return "left half"
  if rendered_indices == {1, 3}:
    return "right half"
  if rendered_indices == {0, 1}:
    return "top half"
  if rendered_indices == {2, 3}:
    return "bottom half"

  # Check for single quadrants
  if rendered_indices == {0}:
    return "top left quadrant"
  if rendered_indices == {1}:
    return "top right quadrant"
  if rendered_indices == {2}:
    return "bottom left quadrant"
  if rendered_indices == {3}:
    return "bottom right quadrant"

  # Three quadrants (L-shapes)
  if rendered_indices == {0, 1, 2}:
    return "top half and bottom left quadrant"
  if rendered_indices == {0, 1, 3}:
    return "top half and bottom right quadrant"
  if rendered_indices == {0, 2, 3}:
    return "left half and bottom right quadrant"
  if rendered_indices == {1, 2, 3}:
    return "right half and bottom left quadrant"

  # Multiple non-adjacent quadrants (diagonal)
  if rendered_indices == {0, 3}:
    return "top left and bottom right quadrants"
  if rendered_indices == {1, 2}:
    return "top right and bottom left quadrants"

  # Fallback for any other combination
  quadrant_names = {
    0: "top left",
    1: "top right",
    2: "bottom left",
    3: "bottom right",
  }
  names = [quadrant_names[i] for i in sorted(rendered_indices)]
  return " and ".join(names) + " quadrants"


def call_oxen_api(
  image_url: str,
  api_key: str,
  *,
  use_infill_model: bool = False,
  region_description: str | None = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key
    use_infill_model: If True, use the infill model instead of generation model
    region_description: For infill model, describes which region to convert

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  if use_infill_model:
    model = "cannoneyed-satisfactory-harlequin-minnow"  # V04 infill model
    if region_description:
      prompt = (
        f"Convert the {region_description} of the image to isometric nyc pixel art "
        f"in precisely the style of the other part of the image."
      )
    else:
      prompt = "Convert the input image to <isometric nyc pixel art>"
  else:
    model = "cannoneyed-odd-blue-marmot"  # V04 generation model
    prompt = "Convert the input image to <isometric nyc pixel art>"

  payload = {
    "model": model,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Prompt: {payload['prompt']}")
  print(f"      Image: {image_url}")

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


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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
  output_path = renders_dir / f"render_{x}_{y}.png"

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

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    infills_dir = generation_dir / "infills"
    infills_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(
        f"\n   Drawing red border around {len(render_positions)} render quadrant(s)"
      )
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = infills_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)

    # We use the generation model if *all* quadrants must be generated (i.e. len(render_positions) == 4)
    use_infill_model = len(render_positions) != 4

    # Get region description for infill model prompt
    region_description = None
    if use_infill_model:
      region_description = get_region_description(render_positions)
      print(f"   Region to generate: {region_description}")

    generated_url = call_oxen_api(
      image_url=image_url,
      api_key=api_key,
      use_infill_model=use_infill_model,
      region_description=region_description,
    )

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode: skip model inference and just save the infill image",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
      args.debug,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Done! I've added the `get_region_description` function and updated the API call. Now when using the infill model, the prompt will be dynamic based on which quadrants need generation:

**Examples of generated prompts:**
- If bottom half needs generation: `"Convert the bottom half of the image to isometric nyc pixel art in precisely the style of the other part of the image."`
- If top left quadrant needs generation: `"Convert the top left quadrant of the image to isometric nyc pixel art in precisely the style of the other part of the image."`
- If right half needs generation: `"Convert the right half of the image to isometric nyc pixel art in precisely the style of the other part of the image."`

The logic handles:
- Half patterns (left/right/top/bottom half)
- Single quadrants
- Three quadrants (L-shapes)
- Diagonal quadrants
- Fallback for any other combination

---

**User**

Can you modify @src/isometric_nyc/e2e_generation/view_generations.py to make two separate toggles for lines/coords?

---

**Cursor**

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
      background: #333;
    }
    
    .grid-container.show-lines .grid {
      gap: 2px;
    }
    
    .grid-container:not(.show-lines) .grid {
      gap: 0;
      background: transparent;
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-lines) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines/Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-lines) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines/Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines/Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      return { x, y, n, showLines };
    }
    
    function goTo() {
      const { x, y, n, showLines } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const params = getParams();
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

Done! The viewer now has two separate toggles:

**URL Parameters:**
- `lines` - Show grid lines: 1=on, 0=off (default: 1)
- `coords` - Show coordinates: 1=on, 0=off (default: 1)

**Keyboard shortcuts:**
- `L` - Toggle lines
- `C` - Toggle coords

You can now independently control whether to show the grid borders between tiles and whether to show the (x, y) coordinate labels on each tile.

---

**User**

Add a key handler so that the "l" key toggles on/off lines and "c" key toggles on/off coords

---

**Cursor**

Those keyboard shortcuts are already implemented! In the last update, I added:

- `L` key - toggles lines
- `C` key - toggles coords

Let me verify they're in place:

The keyboard shortcuts are already implemented from the previous update:

```javascript
case 'l': case 'L':
  document.getElementById('showLines').click();
  break;
case 'c': case 'C':
  document.getElementById('showCoords').click();
  break;
```

Just press `L` to toggle lines and `C` to toggle coords when viewing the page (make sure you're not focused on an input field).

```javascript
case 'l': case 'L':
  document.getElementById('showLines').click();
  break;
case 'c': case 'C':
  document.getElementById('showCoords').click();
  break;
```

---

**User**

Can you make there be two separate checkboxes for lines/coords?

---

**Cursor**

That's already implemented from the earlier update! Let me verify:

The two separate checkboxes are already there from the earlier changes:

```html
<label>
  <input type="checkbox" id="showLines" ... onchange="toggleLines()">
  Lines
</label>
<label>
  <input type="checkbox" id="showCoords" ... onchange="toggleCoords()">
  Coords
</label>
```

You should see two checkboxes in the viewer:
- **Lines** - toggles grid borders between tiles
- **Coords** - toggles coordinate labels (x, y) on each tile

You may need to restart the Flask server to see the changes if it was running before the update. Would you like me to restart it?

```html
<label>
  <input type="checkbox" id="showLines" ... onchange="toggleLines()">
  Lines
</label>
<label>
  <input type="checkbox" id="showCoords" ... onchange="toggleCoords()">
  Coords
</label>
```

---

**User**

It seems like the generation for quadrant 0, -1 and 1, -1 were marked as saved but the @view_generations.py  server is returning a 404 for them...

---

**Cursor**

Let me help debug this. Let's check what's actually in the database for those quadrants:

```python
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
```

```python
@app.route("/tile/<int:x>/<int:y>")
def tile(x: int, y: int):
  """Serve a tile image."""
  gen = get_quadrant_generation(x, y)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")
```

The database shows the data IS there:

```
0|-1|1|1|478988   (has_gen=1, 478KB)
1|-1|1|1|471712   (has_gen=1, 471KB)
```

The quadrants (0, -1) and (1, -1) have generations saved. The issue might be:

1. **Server needs restart** - The Flask server might be running old code
2. **URL routing issue** - Flask's `<int:x>` should handle negatives, but let's verify

Let me restart the viewer server:

```plaintext
0|-1|1|1|478988   (has_gen=1, 478KB)
1|-1|1|1|471712   (has_gen=1, 471KB)
```

---

**User**

wait - the generation bytes size is 0 in those two, right?

---

**Cursor**

```plaintext
-1|-1|1|0|
0|-1|1|1|478988
1|-1|1|1|471712
```

```plaintext
-1|-1|1|0|
```

```plaintext
0|-1|1|1|478988
```

No, looking at the output again:

```
-1|-1|1|0|           <- This is quadrant (-1, -1): has_gen=0, size=NULL
0|-1|1|1|478988      <- This is quadrant (0, -1): has_gen=1, size=478,988 bytes
1|-1|1|1|471712      <- This is quadrant (1, -1): has_gen=1, size=471,712 bytes
```

The format is: `quadrant_x | quadrant_y | has_render | has_gen | gen_size`

Quadrants (0, -1) and (1, -1) DO have generations (~470KB each). The one with `0` for generation is (-1, -1).

So the data is definitely in the database. The 404 must be coming from somewhere else. Have you tried restarting the viewer server? Or try accessing the tile directly in your browser:

```
http://localhost:8080/tile/0/-1
```

If that still 404s, there might be an issue with how Flask handles the negative number in the URL path.

```plaintext
-1|-1|1|0|           <- This is quadrant (-1, -1): has_gen=0, size=NULL
0|-1|1|1|478988      <- This is quadrant (0, -1): has_gen=1, size=478,988 bytes
1|-1|1|1|471712      <- This is quadrant (1, -1): has_gen=1, size=471,712 bytes
```

```plaintext
http://localhost:8080/tile/0/-1
```

---

**User**

yeah looks like it's something to do with flask and the negative - the plain url just 404s

---

**Cursor**

Ah yes, Flask's default `<int:x>` converter doesn't handle negative numbers. Let me fix that:

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  gen = get_quadrant_generation(qx, qy)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

Fixed! The issue was that Flask's `<int:x>` converter only matches positive integers. I changed it to accept any string and convert manually with `int()`, which handles negatives.

Restart the viewer server and it should work now for coordinates like `/tile/0/-1`.

---

**User**

Can you add another checkmark toggle (which changes via the "r" key, that switches from visualizing the "generation" png data to serving and visualizing the "render" png data

---

**Cursor**

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  gen = get_quadrant_generation(qx, qy)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  gen = get_quadrant_generation(qx, qy)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  gen = get_quadrant_generation(qx, qy)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      const showRender = document.getElementById('showRender').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords, showRender };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleRender() {
      // This requires a page reload to fetch different data
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
        case 'r': case 'R':
          document.getElementById('showRender').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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


@app.route("/")
def index():
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  gen = get_quadrant_generation(qx, qy)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      const showRender = document.getElementById('showRender').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords, showRender };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleRender() {
      // This requires a page reload to fetch different data
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
        case 'r': case 'R':
          document.getElementById('showRender').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have generations
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      gen = get_quadrant_generation(qx, qy)
      tiles[(dx, dy)] = gen is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  gen = get_quadrant_generation(qx, qy)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      const showRender = document.getElementById('showRender').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords, showRender };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleRender() {
      // This requires a page reload to fetch different data
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
        case 'r': case 'R':
          document.getElementById('showRender').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  gen = get_quadrant_generation(qx, qy)
  if gen is None:
    return Response("Not found", status=404)

  return Response(gen, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      const showRender = document.getElementById('showRender').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords, showRender };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleRender() {
      // This requires a page reload to fetch different data
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
        case 'r': case 'R':
          document.getElementById('showRender').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
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


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

Done! Added a new "Renders" toggle:

**Features:**
- **Checkbox** - "Renders" checkbox in the toggle group
- **Keyboard** - Press `R` to toggle between render/generation mode
- **URL param** - `render=1` shows renders, `render=0` shows generations (default)

When "Renders" is checked, the viewer will display the raw 3D renders from the database instead of the AI-generated pixel art. This is useful for comparing the input vs output or debugging issues.

Restart the Flask server to see the changes.

---

**User**

can you change the render/generation toggle key to "g" instead of "r"?

---

**Cursor**

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  R          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      const showRender = document.getElementById('showRender').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords, showRender };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleRender() {
      // This requires a page reload to fetch different data
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
        case 'g': case 'G':
          document.getElementById('showRender').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
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


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Simple web app to view generated tiles in an n×n grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  n      - Grid size n×n (default: 2, max: 10)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .nav-buttons {
      display: flex;
      gap: 8px;
    }
    
    .nav-buttons button {
      padding: 8px 12px;
      background: #333;
      color: #fff;
    }
    
    .nav-buttons button:hover {
      background: #444;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
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
      grid-template-columns: repeat({{ n }}, 1fr);
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
      min-width: 256px;
      min-height: 256px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>N: <input type="number" id="n" value="{{ n }}" min="1" max="10"></label>
    <button onclick="goTo()">Go</button>
    
    <div class="nav-buttons">
      <button onclick="navigate(-1, 0)">← Left</button>
      <button onclick="navigate(1, 0)">Right →</button>
      <button onclick="navigate(0, -1)">↑ Up</button>
      <button onclick="navigate(0, 1)">Down ↓</button>
    </div>
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(n) %}
        {% for dx in range(n) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ n }}×{{ n }} quadrants from ({{ x }}, {{ y }}) through ({{ x + n - 1 }}, {{ y + n - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const n = document.getElementById('n').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      const showRender = document.getElementById('showRender').checked ? '1' : '0';
      return { x, y, n, showLines, showCoords, showRender };
    }
    
    function goTo() {
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&n=${params.n}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleRender() {
      // This requires a page reload to fetch different data
      const { x, y, n, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&n=${n}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
        case 'g': case 'G':
          document.getElementById('showRender').click();
          break;
      }
    });
  </script>
</body>
</html>
"""


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
  """Main page showing n×n grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  n = request.args.get("n", 2, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp n to reasonable bounds
  n = max(1, min(n, 10))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(n):
    for dy in range(n):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    n=n,
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


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a 2x2 grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=False)
  return 0


if __name__ == "__main__":
  exit(main())

```

Done! The keyboard shortcut is now `G` to toggle between render/generation mode.

