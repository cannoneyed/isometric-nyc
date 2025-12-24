"""
Get web-rendered tiles for exported generation tiles.

This script takes an exports directory containing PNG files named with
tl/br quadrant coordinates, and renders the corresponding tiles from the
web viewer, saving them in a 'render' subdirectory.

Export filename format: export_tl_X1_Y1_br_X2_Y2.png
  - X1, Y1: top-left quadrant coordinates
  - X2, Y2: bottom-right quadrant coordinates

Usage:
  uv run python src/isometric_nyc/e2e_generation/get_renders_for_exports.py <exports_dir>
  uv run python src/isometric_nyc/e2e_generation/get_renders_for_exports.py generations/v01/exports/terrain
"""

import argparse
import re
import sqlite3
from pathlib import Path

from PIL import Image

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  build_tile_render_url,
  ensure_quadrant_exists,
  get_generation_config,
  render_url_to_image,
  split_tile_into_quadrants,
  start_web_server,
)

# Regex pattern to parse export filenames
# e.g., export_tl_-1_77_br_0_78.png
EXPORT_PATTERN = re.compile(
  r"export_tl_(?P<tl_x>-?\d+)_(?P<tl_y>-?\d+)_br_(?P<br_x>-?\d+)_(?P<br_y>-?\d+)\.png"
)


def parse_export_filename(filename: str) -> tuple[int, int, int, int] | None:
  """
  Parse an export filename to extract quadrant coordinates.

  Args:
      filename: Filename like "export_tl_-1_77_br_0_78.png"

  Returns:
      Tuple of (tl_x, tl_y, br_x, br_y) or None if parsing failed
  """
  match = EXPORT_PATTERN.match(filename)
  if not match:
    return None

  return (
    int(match.group("tl_x")),
    int(match.group("tl_y")),
    int(match.group("br_x")),
    int(match.group("br_y")),
  )


def render_tile_region(
  conn: sqlite3.Connection,
  config: dict,
  tl_x: int,
  tl_y: int,
  br_x: int,
  br_y: int,
  port: int,
) -> Image.Image | None:
  """
  Render a tile region spanning from (tl_x, tl_y) to (br_x, br_y).

  For a standard 2x2 tile where br = tl + (1, 1), this renders a single tile.
  For larger regions, this stitches multiple tiles together.

  Args:
      conn: Database connection
      config: Generation config dict
      tl_x, tl_y: Top-left quadrant coordinates
      br_x, br_y: Bottom-right quadrant coordinates
      port: Web server port

  Returns:
      Rendered PIL Image or None if failed
  """
  # Calculate size of the region in quadrants
  width_quadrants = br_x - tl_x + 1
  height_quadrants = br_y - tl_y + 1

  # Each tile renders a 2x2 region of quadrants
  # We need to figure out how many tiles to render
  quadrant_width_px = config["width_px"] // 2
  quadrant_height_px = config["height_px"] // 2

  # Total output size
  output_width = width_quadrants * quadrant_width_px
  output_height = height_quadrants * quadrant_height_px

  # Create output canvas
  output_image = Image.new("RGB", (output_width, output_height))

  # Render tiles - each tile covers a 2x2 region starting at even coordinates
  # We need to render tiles that cover our region
  tiles_rendered = set()

  # Iterate over each quadrant in the region
  for qy in range(tl_y, br_y + 1):
    for qx in range(tl_x, br_x + 1):
      # Find the tile that contains this quadrant
      # Tiles are rendered at even coordinates
      tile_x = (qx // 2) * 2
      tile_y = (qy // 2) * 2

      # Skip if we've already rendered this tile
      if (tile_x, tile_y) in tiles_rendered:
        continue

      tiles_rendered.add((tile_x, tile_y))

      # Ensure the quadrant exists to get lat/lng
      quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

      # Build URL and render using shared utility
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

      print(f"   🎨 Rendering tile at ({tile_x}, {tile_y})...")
      tile_image = render_url_to_image(url, config["width_px"], config["height_px"])

      # Split into quadrants and paste relevant ones into output
      quadrant_images = split_tile_into_quadrants(tile_image)
      for (dx, dy), quad_img in quadrant_images.items():
        src_qx = tile_x + dx
        src_qy = tile_y + dy

        # Check if this quadrant is in our target region
        if tl_x <= src_qx <= br_x and tl_y <= src_qy <= br_y:
          # Calculate destination position
          dst_x = (src_qx - tl_x) * quadrant_width_px
          dst_y = (src_qy - tl_y) * quadrant_height_px
          output_image.paste(quad_img, (dst_x, dst_y))

  return output_image


def process_exports(
  exports_dir: Path,
  generation_dir: Path,
  port: int,
  dry_run: bool = False,
) -> int:
  """
  Process all export files and generate corresponding renders.

  Args:
      exports_dir: Directory containing export PNG files
      generation_dir: Generation directory with quadrants.db
      port: Web server port
      dry_run: If True, only print what would be done

  Returns:
      Number of renders created
  """
  # Find export files
  export_files = sorted(exports_dir.glob("export_tl_*.png"))

  if not export_files:
    print(f"❌ No export files found in {exports_dir}")
    return 0

  print(f"📁 Found {len(export_files)} export files")

  # Create render output directory
  render_dir = exports_dir / "render"
  if not dry_run:
    render_dir.mkdir(exist_ok=True)

  # Connect to database
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)
  config = get_generation_config(conn)

  renders_created = 0

  try:
    for export_file in export_files:
      coords = parse_export_filename(export_file.name)
      if coords is None:
        print(f"⚠️  Skipping {export_file.name} - couldn't parse coordinates")
        continue

      tl_x, tl_y, br_x, br_y = coords
      output_name = f"render_tl_{tl_x}_{tl_y}_br_{br_x}_{br_y}.png"
      output_path = render_dir / output_name

      print(f"\n📍 {export_file.name}")
      print(f"   Region: ({tl_x}, {tl_y}) to ({br_x}, {br_y})")
      print(f"   Output: {output_name}")

      if dry_run:
        renders_created += 1
        continue

      # Render the tile region
      rendered_image = render_tile_region(conn, config, tl_x, tl_y, br_x, br_y, port)

      if rendered_image:
        rendered_image.save(output_path)
        print(f"   ✅ Saved to {output_path}")
        renders_created += 1
      else:
        print("   ❌ Failed to render")

  finally:
    conn.close()

  return renders_created


def main():
  parser = argparse.ArgumentParser(
    description="Get web-rendered tiles for exported generation tiles."
  )
  parser.add_argument(
    "exports_dir",
    type=Path,
    help="Directory containing export PNG files",
  )
  parser.add_argument(
    "--generation-dir",
    type=Path,
    default=Path("generations/v01"),
    help="Generation directory with quadrants.db (default: generations/v01)",
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
    "--dry-run",
    action="store_true",
    help="Print what would be done without rendering",
  )

  args = parser.parse_args()

  exports_dir = args.exports_dir.resolve()
  generation_dir = args.generation_dir.resolve()

  if not exports_dir.exists():
    print(f"❌ Exports directory not found: {exports_dir}")
    return 1

  if not generation_dir.exists():
    print(f"❌ Generation directory not found: {generation_dir}")
    return 1

  print("=" * 60)
  print("🖼️  GET RENDERS FOR EXPORTS")
  print(f"   Exports: {exports_dir}")
  print(f"   Generation: {generation_dir}")
  print(f"   Port: {args.port}")
  if args.dry_run:
    print("   Mode: DRY RUN")
  print("=" * 60)

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server and not args.dry_run:
      print("\n🌐 Starting web server...")
      web_server = start_web_server(WEB_DIR, args.port)

    # Process exports
    count = process_exports(exports_dir, generation_dir, args.port, args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
      print(f"✨ Would create {count} renders")
    else:
      print(f"✨ Created {count} renders")
    print("=" * 60)

    return 0

  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"\n❌ Error: {e}")
    raise
  finally:
    if web_server:
      print("\n🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())
