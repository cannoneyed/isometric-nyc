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

from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  CHROMIUM_ARGS,
  DEFAULT_WEB_PORT,
  WEB_DIR,
  build_tile_render_url,
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
  url = build_tile_render_url(
    port=port,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    azimuth=config["camera_azimuth_degrees"],
    elevation=config["camera_elevation_degrees"],
    view_height=config.get("view_height_meters", 200),
  )

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
      browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)

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
