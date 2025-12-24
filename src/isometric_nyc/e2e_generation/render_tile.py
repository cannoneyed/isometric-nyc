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

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  build_tile_render_url,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  image_to_png_bytes,
  render_url_to_image,
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

    # Build URL using shared utility
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

    # Render using shared utility
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    tile_image = render_url_to_image(url, config["width_px"], config["height_px"])

    # Save full tile
    tile_image.save(output_path)
    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
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
