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
import io
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


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
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
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
  }


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


def save_quadrant_render(
  conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Returns True if successful, False if quadrant not found.
  """
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


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


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

    # Get the quadrant at position (x, y) - this will be the TL of the rendered tile
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

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

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

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
