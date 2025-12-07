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
