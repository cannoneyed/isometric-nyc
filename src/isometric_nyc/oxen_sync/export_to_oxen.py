"""
Export generations database to a local oxen directory structure.

Usage:
  uv run export-to-oxen
  uv run export-to-oxen --n_quadrants 100
  uv run export-to-oxen --oxen_dataset my-dataset --generations_dir nyc
"""

import argparse
import shutil
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from isometric_nyc.oxen_sync.utils import (
  compute_hash,
  format_filename,
  parse_filename,
  write_csv,
)

load_dotenv()


def get_db_path(generations_dir: str) -> Path:
  """Get the path to the quadrants database."""
  return Path("generations") / generations_dir / "quadrants.db"


def get_generations_config_path(generations_dir: str) -> Path:
  """Get the path to the generation_config.json."""
  return Path("generations") / generations_dir / "generation_config.json"


def get_bounds_path(generations_dir: str) -> Path:
  """Get the path to the bounds.json."""
  return Path("generations") / generations_dir / "bounds.json"


def get_existing_hashes(generations_out_dir: Path) -> dict[tuple[int, int], str]:
  """
  Get hashes of existing files in the output directory.

  Returns:
    Dict mapping (x, y) -> hash
  """
  result = {}

  if not generations_out_dir.exists():
    return result

  for png_file in generations_out_dir.glob("*.png"):
    parsed = parse_filename(png_file.name)
    if parsed:
      x, y, hash_str = parsed
      result[(x, y)] = hash_str

  return result


def export_to_oxen(
  generations_dir: str,
  oxen_dataset: str,
  n_quadrants: int | None = None,
) -> None:
  """
  Export quadrants from the local database to a local oxen directory structure.

  Saves files to oxen/<dataset>/generations/<xxx>_<yyy>_<hash>.png
  and creates data.csv in oxen/<dataset>/.

  Args:
    generations_dir: Name of the generations directory (e.g., "nyc")
    oxen_dataset: Name for the local oxen dataset directory
    n_quadrants: Optional limit on number of quadrants to export
  """
  db_path = get_db_path(generations_dir)
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Set up local output directories
  oxen_dir = Path("oxen") / "datasets" / oxen_dataset
  generations_out_dir = oxen_dir / "generations"
  generations_out_dir.mkdir(parents=True, exist_ok=True)

  print(f"📦 Exporting from {db_path} to {oxen_dir}")
  if n_quadrants:
    print(f"   Limiting to {n_quadrants} quadrants")

  # Copy generation_config.json
  config_src = get_generations_config_path(generations_dir)
  if config_src.exists():
    config_dst = oxen_dir / "generation_config.json"
    shutil.copy2(config_src, config_dst)
    print(f"📋 Copied generation_config.json to {config_dst}")
  else:
    print(f"⚠️  generation_config.json not found at {config_src}")

  # Copy bounds.json
  bounds_src = get_bounds_path(generations_dir)
  if bounds_src.exists():
    bounds_dst = oxen_dir / "bounds.json"
    shutil.copy2(bounds_src, bounds_dst)
    print(f"📋 Copied bounds.json to {bounds_dst}")
  else:
    print(f"⚠️  bounds.json not found at {bounds_src}")

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Count total quadrants first (fast query without blob data)
  print("🔍 Counting quadrants in database...")
  count_query = "SELECT COUNT(*) FROM quadrants WHERE generation IS NOT NULL"
  cursor.execute(count_query)
  total_in_db = cursor.fetchone()[0]
  print(f"   Found {total_in_db} quadrants with generation data")

  # Get existing hashes to determine what needs to be saved
  print("🔍 Scanning existing files...")
  existing_hashes = get_existing_hashes(generations_out_dir)
  print(f"   {len(existing_hashes)} files already exist")

  # Get coordinates and identify what needs to be saved (without loading blobs)
  print("📊 Identifying tiles to save...")
  coords_query = """
    SELECT quadrant_x, quadrant_y
    FROM quadrants
    WHERE generation IS NOT NULL
  """
  if n_quadrants:
    coords_query += f" LIMIT {n_quadrants}"

  cursor.execute(coords_query)
  all_coords = cursor.fetchall()

  # Determine which tiles need saving by checking against existing files
  tiles_to_save_coords = []
  tiles_unchanged = 0

  for x, y in all_coords:
    if (x, y) not in existing_hashes:
      # New tile, needs to be saved
      tiles_to_save_coords.append((x, y))
    else:
      # File exists - we'll assume it's unchanged since we can't check hash without loading blob
      # For a full hash check, we'd need to load the blob which defeats the purpose
      tiles_unchanged += 1

  total_to_save = len(tiles_to_save_coords)
  print(f"   {total_to_save} tiles to save (new)")
  print(f"   {tiles_unchanged} tiles already exist (skipping)")

  if total_to_save == 0:
    print("✅ No changes to export - all files are up to date")
    conn.close()
    return

  # Save generations with progress tracking (fetch blobs one at a time)
  print(f"💾 Saving {total_to_save} generations...")
  files_written = 0

  for x, y in tqdm(tiles_to_save_coords, desc="Saving"):
    # Fetch blob for this specific tile
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    if row is None or row[0] is None:
      continue

    gen_blob = row[0]
    gen_hash = compute_hash(gen_blob)

    # Remove old file if it exists with different hash
    for old_file in generations_out_dir.glob(f"{x:04d}_{y:04d}_*.png"):
      old_file.unlink()

    # Write generation PNG to local directory
    gen_filename = format_filename(x, y, gen_hash)
    gen_path = generations_out_dir / gen_filename
    gen_path.write_bytes(gen_blob)
    files_written += 1

  conn.close()

  # Build CSV from the files on disk (avoids reloading all blobs)
  print("📝 Writing generations.csv...")
  csv_rows = []
  for png_file in sorted(generations_out_dir.glob("*.png")):
    parsed = parse_filename(png_file.name)
    if parsed:
      x, y, hash_str = parsed
      csv_rows.append(
        {
          "x": x,
          "y": y,
          "quadrant": f"generations/{png_file.name}",
          "hash_quadrant": hash_str,
        }
      )

  csv_path = oxen_dir / "generations.csv"
  write_csv(csv_path, csv_rows)

  print(f"✅ Successfully exported to {oxen_dir}")
  print(f"   Total quadrants: {len(csv_rows)}")
  print(f"   Files written: {files_written}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Export generations database to a local oxen directory"
  )
  parser.add_argument(
    "--generations_dir",
    type=str,
    default="nyc",
    help="Name of the generations directory (default: nyc)",
  )
  parser.add_argument(
    "--oxen_dataset",
    type=str,
    default="isometric-nyc-tiles",
    help="Name for the local oxen dataset directory (default: isometric-nyc-tiles)",
  )
  parser.add_argument(
    "--n_quadrants",
    type=int,
    default=None,
    help="Limit number of quadrants to export (default: all)",
  )

  args = parser.parse_args()

  export_to_oxen(
    generations_dir=args.generations_dir,
    oxen_dataset=args.oxen_dataset,
    n_quadrants=args.n_quadrants,
  )


if __name__ == "__main__":
  main()
