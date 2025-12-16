# Export/import generation tile rectangle arguments
_Exported on 12/16/2025 at 10:16:44 EST from Cursor (2.1.39)_

---

**User**

OK - can you modify @src/isometric_nyc/e2e_generation/export_import_generation_tile.py to accept a tl / br arguments that imports + exports rectangles of the specified top left / bottom right? Two notes:

1. The rectangle must have all generated quadrants

2. The png should be saved as `export_tl_<tl>_br_<br>.png`



Ensure that the quadrant sizes are not changed

---

**Cursor**

Looking at the file, I need to modify it to support rectangular regions specified by top-left and bottom-right coordinates. Let me update the script:

```python
"""
Export or import a rectangular region of generation tiles to/from the exports folder.

If the export file (<generation_dir>/exports/export_tl_X_Y_br_X_Y.png) does NOT exist:
  - Exports the region from the database to the export file

If the export file DOES exist:
  - Imports the quadrants from the file back into the database

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Export/import rectangular region from (0,0) to (3,3) (4x4 quadrants)
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3

  # Export/import a single 2x2 tile
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 1,1

  # With --render flag to export/import render pixels instead of generation pixels
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 2,2 --br 5,5 --render

  # Force export even if file exists
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-export

  # Force import even if file doesn't exist
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-import
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

```

```python
"""
Export or import a rectangular region of generation tiles to/from the exports folder.

If the export file (<generation_dir>/exports/export_tl_X_Y_br_X_Y.png) does NOT exist:
  - Exports the region from the database to the export file

If the export file DOES exist:
  - Imports the quadrants from the file back into the database

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Export/import rectangular region from (0,0) to (3,3) (4x4 quadrants)
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3

  # Export/import a single 2x2 tile
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 1,1

  # With --render flag to export/import render pixels instead of generation pixels
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 2,2 --br 5,5 --render

  # Force export even if file exists
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-export

  # Force import even if file doesn't exist
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-import
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
  width_count: int,
  height_count: int,
) -> Image.Image:
  """
  Stitch quadrant images into a single tile image.

  Args:
      quadrants: Dict mapping (dx, dy) offset to the quadrant image.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Combined tile image.
  """
  sample_quad = next(iter(quadrants.values()))
  quad_w, quad_h = sample_quad.size

  tile = Image.new("RGBA", (quad_w * width_count, quad_h * height_count))

  for (dx, dy), quad_img in quadrants.items():
    pos = (dx * quad_w, dy * quad_h)
    tile.paste(quad_img, pos)

  return tile


def split_into_quadrants(
  image: Image.Image,
  width_count: int,
  height_count: int,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split an image into a grid of quadrant images.

  Args:
      image: The source image to split.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Dict mapping (dx, dy) offset to the quadrant image.
  """
  width, height = image.size
  quad_w = width // width_count
  quad_h = height // height_count

  quadrants = {}
  for dy in range(height_count):
    for dx in range(width_count):
      left = dx * quad_w
      top = dy * quad_h
      right = left + quad_w
      bottom = top + quad_h
      quadrants[(dx, dy)] = image.crop((left, top, right, bottom))

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

```

```python
"""
Export or import a rectangular region of generation tiles to/from the exports folder.

If the export file (<generation_dir>/exports/export_tl_X_Y_br_X_Y.png) does NOT exist:
  - Exports the region from the database to the export file

If the export file DOES exist:
  - Imports the quadrants from the file back into the database

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Export/import rectangular region from (0,0) to (3,3) (4x4 quadrants)
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3

  # Export/import a single 2x2 tile
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 1,1

  # With --render flag to export/import render pixels instead of generation pixels
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 2,2 --br 5,5 --render

  # Force export even if file exists
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-export

  # Force import even if file doesn't exist
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-import
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
  width_count: int,
  height_count: int,
) -> Image.Image:
  """
  Stitch quadrant images into a single tile image.

  Args:
      quadrants: Dict mapping (dx, dy) offset to the quadrant image.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Combined tile image.
  """
  sample_quad = next(iter(quadrants.values()))
  quad_w, quad_h = sample_quad.size

  tile = Image.new("RGBA", (quad_w * width_count, quad_h * height_count))

  for (dx, dy), quad_img in quadrants.items():
    pos = (dx * quad_w, dy * quad_h)
    tile.paste(quad_img, pos)

  return tile


def split_into_quadrants(
  image: Image.Image,
  width_count: int,
  height_count: int,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split an image into a grid of quadrant images.

  Args:
      image: The source image to split.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Dict mapping (dx, dy) offset to the quadrant image.
  """
  width, height = image.size
  quad_w = width // width_count
  quad_h = height // height_count

  quadrants = {}
  for dy in range(height_count):
    for dx in range(width_count):
      left = dx * quad_w
      top = dy * quad_h
      right = left + quad_w
      bottom = top + quad_h
      quadrants[(dx, dy)] = image.crop((left, top, right, bottom))

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
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  use_render: bool = False,
) -> bool:
  """
  Export a rectangular region of quadrants from the database to a PNG file.

  Args:
      db_path: Path to the quadrants.db file.
      tl: (x, y) coordinates of the top-left quadrant.
      br: (x, y) coordinates of the bottom-right quadrant (inclusive).
      output_path: Path to save the output PNG.
      use_render: If True, export render pixels; otherwise export generation pixels.

  Returns:
      True if successful, False otherwise.
  """
  tl_x, tl_y = tl
  br_x, br_y = br
  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1
  data_type = "render" if use_render else "generation"

  print(
    f"📤 Exporting {data_type} region from ({tl_x},{tl_y}) to ({br_x},{br_y}) "
    f"({width_count}x{height_count} quadrants)..."
  )

  quadrants: dict[tuple[int, int], Image.Image] = {}
  missing_quadrants = []

  for dy in range(height_count):
    for dx in range(width_count):
      qx, qy = tl_x + dx, tl_y + dy
      data = get_quadrant_data(db_path, qx, qy, use_render=use_render)

      if data is None:
        missing_quadrants.append((qx, qy))
      else:
        quadrants[(dx, dy)] = png_bytes_to_image(data)
        print(f"   ✓ Quadrant ({qx}, {qy})")

  if missing_quadrants:
    print(f"❌ Error: Missing {data_type} for quadrants: {missing_quadrants}")
    return False

  tile_image = stitch_quadrants_to_tile(quadrants, width_count, height_count)
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

```

```python
"""
Export or import a rectangular region of generation tiles to/from the exports folder.

If the export file (<generation_dir>/exports/export_tl_X_Y_br_X_Y.png) does NOT exist:
  - Exports the region from the database to the export file

If the export file DOES exist:
  - Imports the quadrants from the file back into the database

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Export/import rectangular region from (0,0) to (3,3) (4x4 quadrants)
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3

  # Export/import a single 2x2 tile
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 1,1

  # With --render flag to export/import render pixels instead of generation pixels
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 2,2 --br 5,5 --render

  # Force export even if file exists
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-export

  # Force import even if file doesn't exist
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-import
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
  width_count: int,
  height_count: int,
) -> Image.Image:
  """
  Stitch quadrant images into a single tile image.

  Args:
      quadrants: Dict mapping (dx, dy) offset to the quadrant image.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Combined tile image.
  """
  sample_quad = next(iter(quadrants.values()))
  quad_w, quad_h = sample_quad.size

  tile = Image.new("RGBA", (quad_w * width_count, quad_h * height_count))

  for (dx, dy), quad_img in quadrants.items():
    pos = (dx * quad_w, dy * quad_h)
    tile.paste(quad_img, pos)

  return tile


def split_into_quadrants(
  image: Image.Image,
  width_count: int,
  height_count: int,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split an image into a grid of quadrant images.

  Args:
      image: The source image to split.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Dict mapping (dx, dy) offset to the quadrant image.
  """
  width, height = image.size
  quad_w = width // width_count
  quad_h = height // height_count

  quadrants = {}
  for dy in range(height_count):
    for dx in range(width_count):
      left = dx * quad_w
      top = dy * quad_h
      right = left + quad_w
      bottom = top + quad_h
      quadrants[(dx, dy)] = image.crop((left, top, right, bottom))

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
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  use_render: bool = False,
) -> bool:
  """
  Export a rectangular region of quadrants from the database to a PNG file.

  Args:
      db_path: Path to the quadrants.db file.
      tl: (x, y) coordinates of the top-left quadrant.
      br: (x, y) coordinates of the bottom-right quadrant (inclusive).
      output_path: Path to save the output PNG.
      use_render: If True, export render pixels; otherwise export generation pixels.

  Returns:
      True if successful, False otherwise.
  """
  tl_x, tl_y = tl
  br_x, br_y = br
  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1
  data_type = "render" if use_render else "generation"

  print(
    f"📤 Exporting {data_type} region from ({tl_x},{tl_y}) to ({br_x},{br_y}) "
    f"({width_count}x{height_count} quadrants)..."
  )

  quadrants: dict[tuple[int, int], Image.Image] = {}
  missing_quadrants = []

  for dy in range(height_count):
    for dx in range(width_count):
      qx, qy = tl_x + dx, tl_y + dy
      data = get_quadrant_data(db_path, qx, qy, use_render=use_render)

      if data is None:
        missing_quadrants.append((qx, qy))
      else:
        quadrants[(dx, dy)] = png_bytes_to_image(data)
        print(f"   ✓ Quadrant ({qx}, {qy})")

  if missing_quadrants:
    print(f"❌ Error: Missing {data_type} for quadrants: {missing_quadrants}")
    return False

  tile_image = stitch_quadrants_to_tile(quadrants, width_count, height_count)
  tile_image.save(output_path, "PNG")

  print(f"✅ Exported to: {output_path}")
  print(f"   Tile size: {tile_image.size[0]}x{tile_image.size[1]} pixels")

  return True


def import_tile(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  input_path: Path,
  use_render: bool = False,
  overwrite: bool = False,
) -> bool:
  """
  Import a tile PNG into the database as quadrants.

  Args:
      db_path: Path to the quadrants.db file.
      tl: (x, y) coordinates of the top-left quadrant.
      br: (x, y) coordinates of the bottom-right quadrant (inclusive).
      input_path: Path to the input PNG file.
      use_render: If True, import as render pixels; otherwise as generation pixels.
      overwrite: If True, overwrite existing data.

  Returns:
      True if successful, False otherwise.
  """
  tl_x, tl_y = tl
  br_x, br_y = br
  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1
  total_quadrants = width_count * height_count
  data_type = "render" if use_render else "generation"

  print(
    f"📥 Importing {data_type} region from ({tl_x},{tl_y}) to ({br_x},{br_y}) "
    f"({width_count}x{height_count} quadrants) from {input_path.name}..."
  )

  image = Image.open(input_path)
  print(f"   Image size: {image.size[0]}x{image.size[1]}")

  quadrant_images = split_into_quadrants(image, width_count, height_count)
  quad_w, quad_h = quadrant_images[(0, 0)].size
  print(f"   Quadrant size: {quad_w}x{quad_h}")

  conn = sqlite3.connect(db_path)

  try:
    success_count = 0
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = tl_x + dx, tl_y + dy

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

    print(f"\n✅ Imported {success_count}/{total_quadrants} quadrants")
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

```

```python
"""
Export or import a rectangular region of generation tiles to/from the exports folder.

If the export file (<generation_dir>/exports/export_tl_X_Y_br_X_Y.png) does NOT exist:
  - Exports the region from the database to the export file

If the export file DOES exist:
  - Imports the quadrants from the file back into the database

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Export/import rectangular region from (0,0) to (3,3) (4x4 quadrants)
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3

  # Export/import a single 2x2 tile
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 1,1

  # With --render flag to export/import render pixels instead of generation pixels
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 2,2 --br 5,5 --render

  # Force export even if file exists
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-export

  # Force import even if file doesn't exist
  uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3 --force-import
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
  width_count: int,
  height_count: int,
) -> Image.Image:
  """
  Stitch quadrant images into a single tile image.

  Args:
      quadrants: Dict mapping (dx, dy) offset to the quadrant image.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Combined tile image.
  """
  sample_quad = next(iter(quadrants.values()))
  quad_w, quad_h = sample_quad.size

  tile = Image.new("RGBA", (quad_w * width_count, quad_h * height_count))

  for (dx, dy), quad_img in quadrants.items():
    pos = (dx * quad_w, dy * quad_h)
    tile.paste(quad_img, pos)

  return tile


def split_into_quadrants(
  image: Image.Image,
  width_count: int,
  height_count: int,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split an image into a grid of quadrant images.

  Args:
      image: The source image to split.
      width_count: Number of quadrants horizontally.
      height_count: Number of quadrants vertically.

  Returns:
      Dict mapping (dx, dy) offset to the quadrant image.
  """
  width, height = image.size
  quad_w = width // width_count
  quad_h = height // height_count

  quadrants = {}
  for dy in range(height_count):
    for dx in range(width_count):
      left = dx * quad_w
      top = dy * quad_h
      right = left + quad_w
      bottom = top + quad_h
      quadrants[(dx, dy)] = image.crop((left, top, right, bottom))

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
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  use_render: bool = False,
) -> bool:
  """
  Export a rectangular region of quadrants from the database to a PNG file.

  Args:
      db_path: Path to the quadrants.db file.
      tl: (x, y) coordinates of the top-left quadrant.
      br: (x, y) coordinates of the bottom-right quadrant (inclusive).
      output_path: Path to save the output PNG.
      use_render: If True, export render pixels; otherwise export generation pixels.

  Returns:
      True if successful, False otherwise.
  """
  tl_x, tl_y = tl
  br_x, br_y = br
  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1
  data_type = "render" if use_render else "generation"

  print(
    f"📤 Exporting {data_type} region from ({tl_x},{tl_y}) to ({br_x},{br_y}) "
    f"({width_count}x{height_count} quadrants)..."
  )

  quadrants: dict[tuple[int, int], Image.Image] = {}
  missing_quadrants = []

  for dy in range(height_count):
    for dx in range(width_count):
      qx, qy = tl_x + dx, tl_y + dy
      data = get_quadrant_data(db_path, qx, qy, use_render=use_render)

      if data is None:
        missing_quadrants.append((qx, qy))
      else:
        quadrants[(dx, dy)] = png_bytes_to_image(data)
        print(f"   ✓ Quadrant ({qx}, {qy})")

  if missing_quadrants:
    print(f"❌ Error: Missing {data_type} for quadrants: {missing_quadrants}")
    return False

  tile_image = stitch_quadrants_to_tile(quadrants, width_count, height_count)
  tile_image.save(output_path, "PNG")

  print(f"✅ Exported to: {output_path}")
  print(f"   Tile size: {tile_image.size[0]}x{tile_image.size[1]} pixels")

  return True


def import_tile(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  input_path: Path,
  use_render: bool = False,
  overwrite: bool = False,
) -> bool:
  """
  Import a tile PNG into the database as quadrants.

  Args:
      db_path: Path to the quadrants.db file.
      tl: (x, y) coordinates of the top-left quadrant.
      br: (x, y) coordinates of the bottom-right quadrant (inclusive).
      input_path: Path to the input PNG file.
      use_render: If True, import as render pixels; otherwise as generation pixels.
      overwrite: If True, overwrite existing data.

  Returns:
      True if successful, False otherwise.
  """
  tl_x, tl_y = tl
  br_x, br_y = br
  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1
  total_quadrants = width_count * height_count
  data_type = "render" if use_render else "generation"

  print(
    f"📥 Importing {data_type} region from ({tl_x},{tl_y}) to ({br_x},{br_y}) "
    f"({width_count}x{height_count} quadrants) from {input_path.name}..."
  )

  image = Image.open(input_path)
  print(f"   Image size: {image.size[0]}x{image.size[1]}")

  quadrant_images = split_into_quadrants(image, width_count, height_count)
  quad_w, quad_h = quadrant_images[(0, 0)].size
  print(f"   Quadrant size: {quad_w}x{quad_h}")

  conn = sqlite3.connect(db_path)

  try:
    success_count = 0
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = tl_x + dx, tl_y + dy

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

    print(f"\n✅ Imported {success_count}/{total_quadrants} quadrants")
    return success_count > 0

  finally:
    conn.close()


def parse_coord(value: str) -> tuple[int, int]:
  """Parse a coordinate string like '0,0' into a tuple (x, y)."""
  try:
    parts = value.split(",")
    if len(parts) != 2:
      raise ValueError
    return (int(parts[0]), int(parts[1]))
  except ValueError:
    raise argparse.ArgumentTypeError(
      f"Invalid coordinate format: '{value}'. Expected format: X,Y (e.g., '0,0')"
    )


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export or import a rectangular region of generation tiles to/from the exports directory."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=parse_coord,
    required=True,
    help="Top-left coordinate of the region (format: X,Y, e.g., '0,0')",
  )
  parser.add_argument(
    "--br",
    type=parse_coord,
    required=True,
    help="Bottom-right coordinate of the region, inclusive (format: X,Y, e.g., '3,3')",
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

  tl_x, tl_y = args.tl
  br_x, br_y = args.br

  # Validate coordinates
  if br_x < tl_x or br_y < tl_y:
    print(
      f"❌ Error: Bottom-right ({br_x},{br_y}) must be >= top-left ({tl_x},{tl_y})"
    )
    return 1

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

  export_filename = f"export_tl_{tl_x}_{tl_y}_br_{br_x}_{br_y}.png"
  export_path = exports_dir / export_filename

  width_count = br_x - tl_x + 1
  height_count = br_y - tl_y + 1

  print(f"🎯 Region: ({tl_x},{tl_y}) to ({br_x},{br_y}) ({width_count}x{height_count} quadrants)")
  print(f"   Generation dir: {generation_dir}")
  print(f"   Export path: {export_path}")
  print()

  if args.force_import:
    if not export_path.exists():
      print(f"❌ Error: Export file not found for import: {export_path}")
      return 1
    success = import_tile(
      db_path,
      args.tl,
      args.br,
      export_path,
      use_render=args.render,
      overwrite=args.overwrite,
    )
  elif args.force_export or not export_path.exists():
    success = export_tile(
      db_path,
      args.tl,
      args.br,
      export_path,
      use_render=args.render,
    )
  else:
    print(f"📁 Export file exists: {export_path}")
    success = import_tile(
      db_path,
      args.tl,
      args.br,
      export_path,
      use_render=args.render,
      overwrite=args.overwrite,
    )

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(main())

```

Let me verify the changes by reading the updated file:

The changes are complete. Let me verify the script runs correctly:

The script is working correctly. Here's a summary of the changes:

**Key modifications:**

1. **New CLI arguments**: Replaced `-x` and `-y` with `--tl X,Y` and `--br X,Y` for specifying rectangular regions

2. **Updated `stitch_quadrants_to_tile`**: Now accepts `width_count` and `height_count` to handle arbitrary rectangular grids

3. **Updated `split_into_quadrants`**: Now splits images into arbitrary grids based on the specified dimensions, preserving quadrant sizes

4. **Updated `export_tile` and `import_tile`**: Both now accept `tl` and `br` tuples instead of single coordinates

5. **New filename format**: `export_tl_<tl_x>_<tl_y>_br_<br_x>_<br_y>.png`

6. **Validation**: Added check to ensure bottom-right coordinates are >= top-left coordinates

**Usage examples:**

```bash
# Export/import 4x4 quadrant region
uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3

# Single 2x2 tile (same as before)
uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 1,1
```

The quadrant sizes are preserved during import - the image is split evenly based on the grid dimensions specified by the tl/br coordinates.

```shellscript
# Export/import 4x4 quadrant region
uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 3,3

# Single 2x2 tile (same as before)
uv run python src/isometric_nyc/e2e_generation/export_import_generation_tile.py generations/v01 --tl 0,0 --br 1,1
```

