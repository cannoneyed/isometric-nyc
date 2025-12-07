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
    print("   ⚠️  Warning: Image is not square")

  quadrant_images = split_into_quadrants(image)
  quad_w, quad_h = quadrant_images[(0, 0)].size
  print(f"   Quadrant size: {quad_w}x{quad_h}")

  conn = sqlite3.connect(db_path)

  try:
    print(f"\n🎯 Importing to tile ({x}, {y})")
    print(f"   Quadrants: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")

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
        print(
          f"   ⏭️  Quadrant ({qx}, {qy}) already has generation - skipping (use --overwrite)"
        )
        continue

      # Convert to bytes
      png_bytes = image_to_png_bytes(quad_img)

      if dry_run:
        status = "overwrite" if info["has_generation"] else "new"
        print(
          f"   🔍 Would import quadrant ({qx}, {qy}) [{status}] - {len(png_bytes)} bytes"
        )
        success_count += 1
      else:
        if save_quadrant_generation(conn, qx, qy, png_bytes):
          status = "overwrote" if info["has_generation"] else "imported"
          print(
            f"   ✓ {status.capitalize()} quadrant ({qx}, {qy}) - {len(png_bytes)} bytes"
          )
          success_count += 1
        else:
          print(f"   ❌ Failed to save quadrant ({qx}, {qy})")

    print(
      f"\n{'🔍 Would import' if dry_run else '✅ Imported'} {success_count}/4 quadrants"
    )
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
