"""
Extract tile data (render and generation) for a 2x2 tile.

This script retrieves the render and generation images for a 2x2 tile
defined by its top-left quadrant coordinate and saves them as PNG files.

Usage:
  uv run python src/isometric_nyc/generation/get_tile_data.py \\
    <generation_dir> <x> <y>

  # Specify custom output directory:
  uv run python src/isometric_nyc/generation/get_tile_data.py \\
    <generation_dir> <x> <y> --output-dir ./my_exports

Output (saved to <generation_dir>/exports/ by default):
  - render_<x>_<y>.png: The stitched 2x2 render image
  - generation_<x>_<y>.png: The stitched 2x2 generation image
"""

import argparse
import sqlite3
from pathlib import Path

from PIL import Image

from isometric_nyc.generation.shared import (
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
  output_dir = (
    args.output_dir.resolve() if args.output_dir else generation_dir / "exports"
  )

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
