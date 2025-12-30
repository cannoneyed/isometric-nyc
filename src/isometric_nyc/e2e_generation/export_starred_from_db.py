"""
Export starred quadrants from the database as 2x2 tile images for dataset curation.

This script reads all starred quadrants from the database and exports each as a 2x2
tile image. The starred quadrant is placed in the top-left position of the tile,
and the adjacent quadrants (right, below, and diagonal) complete the 2x2 block.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_starred_from_db.py <generation_dir> --name <dataset_name>

Examples:
  # Export starred quadrants to a new dataset called "buildings_v1"
  uv run python src/isometric_nyc/e2e_generation/export_starred_from_db.py generations/v01 --name buildings_v1

  # List starred quadrants without exporting
  uv run python src/isometric_nyc/e2e_generation/export_starred_from_db.py generations/v01 --list

Output structure:
  synthetic_data/datasets/{name}/
    ├── generations/
    │   ├── tile_x0_y0.png  # 2x2 tile with starred (0,0) in top-left
    │   ├── tile_x5_y3.png  # 2x2 tile with starred (5,3) in top-left
    │   └── ...
    └── renders/
        ├── tile_x0_y0.png
        ├── tile_x5_y3.png
        └── ...
"""

import argparse
import csv
import io
import sqlite3
import sys
from pathlib import Path

from PIL import Image


def get_starred_quadrants(db_path: Path) -> list[dict]:
  """
  Get all starred quadrants from the database.

  Returns a list of dicts with x, y coordinates and data availability info.
  """
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()

    # First check if starred column exists
    cursor.execute("PRAGMA table_info(quadrants)")
    columns = [row[1] for row in cursor.fetchall()]
    if "starred" not in columns:
      print("⚠️  No 'starred' column found in database")
      return []

    cursor.execute(
      """
      SELECT quadrant_x, quadrant_y, 
             generation IS NOT NULL as has_gen,
             render IS NOT NULL as has_render
      FROM quadrants
      WHERE starred = 1
      ORDER BY quadrant_y, quadrant_x
      """
    )

    starred = []
    for row in cursor.fetchall():
      starred.append(
        {
          "x": row[0],
          "y": row[1],
          "has_generation": bool(row[2]),
          "has_render": bool(row[3]),
        }
      )

    return starred
  finally:
    conn.close()


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


def stitch_2x2_tile(
  quadrants: dict[tuple[int, int], Image.Image],
) -> Image.Image:
  """
  Stitch 4 quadrant images into a single 2x2 tile image.

  Args:
      quadrants: Dict mapping (dx, dy) offset (0 or 1) to the quadrant image.

  Returns:
      Combined 2x2 tile image.
  """
  sample_quad = next(iter(quadrants.values()))
  quad_w, quad_h = sample_quad.size

  tile = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  for (dx, dy), quad_img in quadrants.items():
    pos = (dx * quad_w, dy * quad_h)
    tile.paste(quad_img, pos)

  return tile


def export_starred_quadrant(
  db_path: Path,
  x: int,
  y: int,
  output_dir: Path,
  use_render: bool = False,
) -> bool:
  """
  Export a 2x2 tile with the starred quadrant in the top-left position.

  The tile includes:
    - (x, y) - starred quadrant (top-left)
    - (x+1, y) - right neighbor (top-right)
    - (x, y+1) - bottom neighbor (bottom-left)
    - (x+1, y+1) - diagonal neighbor (bottom-right)

  Args:
      db_path: Path to the quadrants.db file.
      x: X coordinate of the starred quadrant.
      y: Y coordinate of the starred quadrant.
      output_dir: Directory to save the output PNG.
      use_render: If True, export render pixels; otherwise export generation pixels.

  Returns:
      True if successful, False otherwise.
  """
  data_type = "render" if use_render else "generation"

  # Get all 4 quadrants for the 2x2 tile
  quadrant_coords = [
    (0, 0, x, y),  # top-left (starred)
    (1, 0, x + 1, y),  # top-right
    (0, 1, x, y + 1),  # bottom-left
    (1, 1, x + 1, y + 1),  # bottom-right
  ]

  quadrants: dict[tuple[int, int], Image.Image] = {}
  missing_quadrants = []

  for dx, dy, qx, qy in quadrant_coords:
    data = get_quadrant_data(db_path, qx, qy, use_render=use_render)

    if data is None:
      missing_quadrants.append((qx, qy))
    else:
      quadrants[(dx, dy)] = png_bytes_to_image(data)

  if missing_quadrants:
    print(f"   ⚠️  Missing {data_type} for: {missing_quadrants} - skipping")
    return False

  # Stitch into 2x2 tile
  tile_image = stitch_2x2_tile(quadrants)

  # Save to output directory
  output_path = output_dir / f"tile_x{x}_y{y}.png"
  tile_image.save(output_path, "PNG")

  print(f"   ✓ Exported ({x}, {y}) → {output_path.name}")
  return True


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export starred quadrants as 2x2 tile images for dataset curation."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="Name for the dataset directory (required unless --list is used)",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="Just list starred quadrants without exporting",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Override the output base directory (default: synthetic_data/datasets)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Get starred quadrants
  starred = get_starred_quadrants(db_path)

  if not starred:
    print("📋 No starred quadrants found in the database.")
    return 0

  print(f"📋 Found {len(starred)} starred quadrant(s):")
  for entry in starred:
    status_parts = []
    if entry["has_generation"]:
      status_parts.append("gen ✓")
    if entry["has_render"]:
      status_parts.append("render ✓")
    status = " | ".join(status_parts) if status_parts else "no data"
    print(f"   ⭐ ({entry['x']}, {entry['y']}) - {status}")

  if args.list:
    return 0

  # Check for required --name argument
  if not args.name:
    print("\n❌ Error: --name is required for export")
    print("   Usage: ... --name <dataset_name>")
    return 1

  # Set up output directories
  if args.output_dir:
    base_dir = args.output_dir.resolve()
  else:
    # Default to synthetic_data/datasets relative to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    base_dir = workspace_root / "synthetic_data" / "datasets"

  dataset_dir = base_dir / args.name
  generations_dir = dataset_dir / "generations"
  renders_dir = dataset_dir / "renders"

  # Check if dataset already exists
  if dataset_dir.exists():
    print(f"\n⚠️  Dataset directory already exists: {dataset_dir}")
    response = input("   Overwrite existing files? [y/N]: ")
    if response.lower() != "y":
      print("   Aborted.")
      return 1

  # Create directories
  generations_dir.mkdir(parents=True, exist_ok=True)
  renders_dir.mkdir(parents=True, exist_ok=True)

  print(f"\n📁 Output directory: {dataset_dir}")
  print(f"   generations/: {generations_dir}")
  print(f"   renders/: {renders_dir}")

  # Track successfully exported tiles for CSV
  exported_tiles: list[dict] = []

  # Export generation tiles
  print("\n🎨 Exporting generation tiles...")
  gen_success = 0
  gen_skip = 0
  for entry in starred:
    if not entry["has_generation"]:
      gen_skip += 1
      continue
    if export_starred_quadrant(
      db_path, entry["x"], entry["y"], generations_dir, use_render=False
    ):
      gen_success += 1
      # Track for CSV (only if both generation and render exist for complete tile)
      if entry["has_render"]:
        exported_tiles.append(
          {
            "x": entry["x"],
            "y": entry["y"],
            "name": f"tile_x{entry['x']}_y{entry['y']}",
          }
        )

  print(f"   ✅ Exported {gen_success} generation tile(s)")
  if gen_skip > 0:
    print(f"   ⏭️  Skipped {gen_skip} (no generation data)")

  # Export render tiles
  print("\n🖼️  Exporting render tiles...")
  render_success = 0
  render_skip = 0
  for entry in starred:
    if not entry["has_render"]:
      render_skip += 1
      continue
    if export_starred_quadrant(
      db_path, entry["x"], entry["y"], renders_dir, use_render=True
    ):
      render_success += 1

  print(f"   ✅ Exported {render_success} render tile(s)")
  if render_skip > 0:
    print(f"   ⏭️  Skipped {render_skip} (no render data)")

  # Write CSV file for use with create_omni_dataset.py
  csv_path = dataset_dir / "settings.csv"
  print("\n📋 Writing settings CSV...")

  # Sort tiles numerically by x, then y
  exported_tiles.sort(key=lambda t: (t["x"], t["y"]))

  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "n_variants", "prompt_suffix"])
    writer.writeheader()
    for tile in exported_tiles:
      writer.writerow(
        {
          "name": tile["name"],
          "n_variants": 2,
          "prompt_suffix": "",
        }
      )

  print(f"   ✅ Created {csv_path.name} with {len(exported_tiles)} entries")

  # Summary
  print(f"\n{'=' * 50}")
  print(f"✅ Dataset export complete: {args.name}")
  print(f"   Location: {dataset_dir}")
  print(f"   Generation tiles: {gen_success}")
  print(f"   Render tiles: {render_success}")
  print(f"   Settings CSV: {len(exported_tiles)} entries")
  print(f"{'=' * 50}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
