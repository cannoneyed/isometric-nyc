"""
Export or import a 2x2 generation tile to/from the generation directory's exports folder.

If the export file (<generation_dir>/exports/export_X_Y.png) does NOT exist:
  - Exports the tile from the database to <generation_dir>/exports/export_X_Y.png

If the export file DOES exist:
  - Imports the four quadrants from the file back into the database

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py <generation_dir> -x X -y Y

Examples:
  # Export/import tile starting at quadrant (0, 0)
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 -x 0 -y 0

  # With --render flag to export/import render pixels instead of generation pixels
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 -x 2 -y 2 --render

  # Force export even if file exists
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 -x 0 -y 0 --force-export

  # Force import even if file doesn't exist
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 -x 0 -y 0 --force-import
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


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


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
  sample_quad = next(iter(quadrants.values()))
  quad_w, quad_h = sample_quad.size

  tile = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  placements = {
    (0, 0): (0, 0),
    (1, 0): (quad_w, 0),
    (0, 1): (0, quad_h),
    (1, 1): (quad_w, quad_h),
  }

  for offset, pos in placements.items():
    if offset in quadrants:
      tile.paste(quadrants[offset], pos)

  return tile


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

  return {
    (0, 0): image.crop((0, 0, half_w, half_h)),
    (1, 0): image.crop((half_w, 0, width, half_h)),
    (0, 1): image.crop((0, half_h, half_w, height)),
    (1, 1): image.crop((half_w, half_h, width, height)),
  }


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


def save_quadrant_data(
  conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes, use_render: bool = False
) -> bool:
  """Save generation or render bytes for a quadrant."""
  cursor = conn.cursor()
  column = "render" if use_render else "generation"
  cursor.execute(
    f"""
        UPDATE quadrants
        SET {column} = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def export_tile(
  db_path: Path,
  x: int,
  y: int,
  output_path: Path,
  use_render: bool = False,
) -> bool:
  """
  Export a 2x2 tile from the database to a PNG file.

  Args:
      db_path: Path to the quadrants.db file.
      x: X coordinate of the top-left quadrant.
      y: Y coordinate of the top-left quadrant.
      output_path: Path to save the output PNG.
      use_render: If True, export render pixels; otherwise export generation pixels.

  Returns:
      True if successful, False otherwise.
  """
  quadrant_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
  data_type = "render" if use_render else "generation"

  print(f"📤 Exporting {data_type} tile at ({x}, {y})...")

  quadrants: dict[tuple[int, int], Image.Image] = {}
  missing_quadrants = []

  for dx, dy in quadrant_positions:
    qx, qy = x + dx, y + dy
    data = get_quadrant_data(db_path, qx, qy, use_render=use_render)

    if data is None:
      missing_quadrants.append((qx, qy))
    else:
      quadrants[(dx, dy)] = png_bytes_to_image(data)
      print(f"   ✓ Quadrant ({qx}, {qy})")

  if missing_quadrants:
    print(f"❌ Error: Missing {data_type} for quadrants: {missing_quadrants}")
    return False

  tile_image = stitch_quadrants_to_tile(quadrants)
  tile_image.save(output_path, "PNG")

  print(f"✅ Exported to: {output_path}")
  print(f"   Tile size: {tile_image.size[0]}x{tile_image.size[1]} pixels")

  return True


def import_tile(
  db_path: Path,
  x: int,
  y: int,
  input_path: Path,
  use_render: bool = False,
  overwrite: bool = False,
) -> bool:
  """
  Import a tile PNG into the database as 4 quadrants.

  Args:
      db_path: Path to the quadrants.db file.
      x: X coordinate of the top-left quadrant.
      y: Y coordinate of the top-left quadrant.
      input_path: Path to the input PNG file.
      use_render: If True, import as render pixels; otherwise as generation pixels.
      overwrite: If True, overwrite existing data.

  Returns:
      True if successful, False otherwise.
  """
  data_type = "render" if use_render else "generation"

  print(f"📥 Importing {data_type} tile at ({x}, {y}) from {input_path.name}...")

  image = Image.open(input_path)
  print(f"   Image size: {image.size[0]}x{image.size[1]}")

  if image.size[0] != image.size[1]:
    print("   ⚠️  Warning: Image is not square")

  quadrant_images = split_into_quadrants(image)
  quad_w, quad_h = quadrant_images[(0, 0)].size
  print(f"   Quadrant size: {quad_w}x{quad_h}")

  conn = sqlite3.connect(db_path)

  try:
    success_count = 0
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy

      info = get_quadrant_info(conn, qx, qy)
      if not info:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database - skipping")
        continue

      has_data = info["has_render"] if use_render else info["has_generation"]
      if has_data and not overwrite:
        print(
          f"   ⏭️  Quadrant ({qx}, {qy}) already has {data_type} - skipping (use --overwrite)"
        )
        continue

      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_data(conn, qx, qy, png_bytes, use_render=use_render):
        status = "overwrote" if has_data else "imported"
        print(
          f"   ✓ {status.capitalize()} quadrant ({qx}, {qy}) - {len(png_bytes)} bytes"
        )
        success_count += 1
      else:
        print(f"   ❌ Failed to save quadrant ({qx}, {qy})")

    print(f"\n✅ Imported {success_count}/4 quadrants")
    return success_count > 0

  finally:
    conn.close()


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export or import a 2x2 generation tile to/from the exports directory."
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
    help="Export/import render pixels instead of generation pixels",
  )
  parser.add_argument(
    "--exports-dir",
    type=Path,
    default=None,
    help="Exports directory (default: <generation_dir>/exports)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="When importing, overwrite existing data",
  )
  parser.add_argument(
    "--force-export",
    action="store_true",
    help="Force export even if the file already exists",
  )
  parser.add_argument(
    "--force-import",
    action="store_true",
    help="Force import mode (error if file doesn't exist)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()
  exports_dir = (args.exports_dir or generation_dir / "exports").resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  exports_dir.mkdir(parents=True, exist_ok=True)

  export_filename = f"export_{args.x}_{args.y}.png"
  export_path = exports_dir / export_filename

  print(f"🎯 Tile: ({args.x}, {args.y})")
  print(f"   Generation dir: {generation_dir}")
  print(f"   Export path: {export_path}")
  print()

  if args.force_import:
    if not export_path.exists():
      print(f"❌ Error: Export file not found for import: {export_path}")
      return 1
    success = import_tile(
      db_path,
      args.x,
      args.y,
      export_path,
      use_render=args.render,
      overwrite=args.overwrite,
    )
  elif args.force_export or not export_path.exists():
    success = export_tile(
      db_path,
      args.x,
      args.y,
      export_path,
      use_render=args.render,
    )
  else:
    print(f"📁 Export file exists: {export_path}")
    success = import_tile(
      db_path,
      args.x,
      args.y,
      export_path,
      use_render=args.render,
      overwrite=args.overwrite,
    )

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())
