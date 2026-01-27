"""
Export a full tile (2x2 quadrant grid) from a generation directory as a PNG file.

A tile consists of 4 quadrants arranged in a 2x2 grid:
  (x, y)     (x+1, y)
  (x, y+1)   (x+1, y+1)

Usage:
  uv run python src/isometric_nyc/generation/export_tile.py <generation_dir> -x X -y Y
  uv run python src/isometric_nyc/generation/export_tile.py <generation_dir> -x X -y Y --render

Examples:
  # Export generation tile starting at quadrant (0, 0)
  uv run python src/isometric_nyc/generation/export_tile.py generations/test_generation -x 0 -y 0

  # Export render tile starting at quadrant (1, 2)
  uv run python src/isometric_nyc/generation/export_tile.py generations/test_generation -x 1 -y 2 --render

Output:
  Creates <x>_<y>.png in the current directory (or specified output dir).
"""

import argparse
import io
import sqlite3
import sys
from pathlib import Path

from PIL import Image


def get_quadrant_data(
  db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
  """
  Get the image bytes for a quadrant at position (x, y).

  Args:
      db_path: Path to the quadrants.db file.
      x: X coordinate of the quadrant.
      y: Y coordinate of the quadrant.
      use_render: If True, get render bytes; otherwise get generation bytes.

  Returns:
      PNG bytes or None if not found.
  """
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def png_bytes_to_image(png_bytes: bytes) -> Image.Image:
  """Convert PNG bytes to a PIL Image."""
  return Image.open(io.BytesIO(png_bytes))


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
      Combined tile image.
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


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export a full tile (2x2 quadrant grid) from a generation directory as a PNG file."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "-x",
    type=int,
    required=True,
    help="X coordinate of the top-left quadrant of the tile",
  )
  parser.add_argument(
    "-y",
    type=int,
    required=True,
    help="Y coordinate of the top-left quadrant of the tile",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Export render pixels instead of generation pixels",
  )
  parser.add_argument(
    "-o",
    "--output-dir",
    type=Path,
    default=Path("."),
    help="Output directory for the PNG file (default: current directory)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()
  output_dir = args.output_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

  # Define the 4 quadrant positions for this tile
  quadrant_positions = [
    (0, 0),  # TL
    (1, 0),  # TR
    (0, 1),  # BL
    (1, 1),  # BR
  ]

  data_type = "render" if args.render else "generation"
  print(f"📤 Exporting {data_type} tile at ({args.x}, {args.y})...")

  # Collect all quadrant images
  quadrants: dict[tuple[int, int], Image.Image] = {}
  missing_quadrants = []

  for dx, dy in quadrant_positions:
    qx, qy = args.x + dx, args.y + dy
    data = get_quadrant_data(db_path, qx, qy, use_render=args.render)

    if data is None:
      missing_quadrants.append((qx, qy))
    else:
      quadrants[(dx, dy)] = png_bytes_to_image(data)
      print(f"   ✓ Quadrant ({qx}, {qy})")

  if missing_quadrants:
    print(f"❌ Error: Missing {data_type} for quadrants: {missing_quadrants}")
    return 1

  # Stitch quadrants into a tile
  tile_image = stitch_quadrants_to_tile(quadrants)

  # Write to file
  output_filename = f"{args.x}_{args.y}.png"
  output_path = output_dir / output_filename

  tile_image.save(output_path, "PNG")

  print(f"✅ Saved to: {output_path}")
  print(f"   Tile size: {tile_image.size[0]}x{tile_image.size[1]} pixels")

  return 0


if __name__ == "__main__":
  sys.exit(main())
