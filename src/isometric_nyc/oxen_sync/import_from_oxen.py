"""
Import from a locally synced oxen dataset to the local generations database.

Usage:
  uv run python src/isometric_nyc/oxen_sync/import_from_oxen.py --generations_dir nyc --oxen_dataset oxen/datasets/isometric-nyc-tiles
"""

import argparse
import json
import re
import shutil
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from isometric_nyc.oxen_sync.utils import compute_hash

load_dotenv()


def get_db_path(generations_dir: str) -> Path:
  """Get the path to the quadrants database."""
  return Path("generations") / generations_dir / "quadrants.db"


def ensure_db_exists(db_path: Path) -> None:
  """Ensure the database exists with the proper schema."""
  is_new = not db_path.exists()

  if is_new:
    print(f"   Creating new database at {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create minimal schema for quadrants
  cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS quadrants (
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      lat REAL NOT NULL DEFAULT 0,
      lng REAL NOT NULL DEFAULT 0,
      tile_row INTEGER NOT NULL DEFAULT 0,
      tile_col INTEGER NOT NULL DEFAULT 0,
      quadrant_index INTEGER NOT NULL DEFAULT 0,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT,
      water_mask BLOB,
      water_type TEXT,
      dark_mode BLOB,
      PRIMARY KEY (quadrant_x, quadrant_y)
    )
    """
  )

  cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS metadata (
      key TEXT PRIMARY KEY,
      value TEXT
    )
    """
  )

  conn.commit()
  conn.close()


def get_local_hashes(
  conn: sqlite3.Connection,
) -> dict[tuple[int, int], str]:
  """
  Compute hashes for all quadrants in the local database.

  Returns:
    Dict mapping (x, y) -> generation_hash
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT quadrant_x, quadrant_y, generation
    FROM quadrants
    WHERE generation IS NOT NULL
    """
  )

  result = {}
  for x, y, gen_blob in cursor.fetchall():
    gen_hash = compute_hash(gen_blob) if gen_blob else ""
    result[(x, y)] = gen_hash

  return result


def update_quadrant(
  conn: sqlite3.Connection,
  x: int,
  y: int,
  generation_blob: bytes,
) -> None:
  """
  Update or insert a quadrant with new generation data.

  Args:
    conn: Database connection
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    generation_blob: PNG bytes for generation
  """
  cursor = conn.cursor()

  # Check if quadrant exists
  cursor.execute(
    "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  exists = cursor.fetchone() is not None

  if exists:
    cursor.execute(
      """
      UPDATE quadrants SET generation = ?
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (generation_blob, x, y),
    )
  else:
    cursor.execute(
      """
      INSERT INTO quadrants (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index, generation)
      VALUES (?, ?, 0, 0, 0, 0, 0, ?)
      """,
      (x, y, generation_blob),
    )


def import_bounds(
  oxen_dir: Path,
  generations_dir_path: Path,
) -> bool:
  """
  Import bounds.json from oxen directory to local generations directory.

  Args:
    oxen_dir: Path to the oxen directory
    generations_dir_path: Path to the local generations directory

  Returns:
    True if bounds was imported, False if not found
  """
  bounds_path = oxen_dir / "bounds.json"

  if not bounds_path.exists():
    print("   ⚠️  No bounds.json found in oxen directory")
    return False

  # Copy to local generations directory
  local_bounds_path = generations_dir_path / "bounds.json"
  shutil.copy(bounds_path, local_bounds_path)
  print(f"   📋 Copied bounds.json to {local_bounds_path}")

  return True


def import_generation_config(
  conn: sqlite3.Connection,
  oxen_dir: Path,
  generations_dir_path: Path,
) -> bool:
  """
  Import generation_config.json from oxen directory to local database.

  Copies the config file and stores it in the metadata table.

  Args:
    conn: Database connection
    oxen_dir: Path to the oxen directory
    generations_dir_path: Path to the local generations directory

  Returns:
    True if config was imported, False if not found
  """
  config_path = oxen_dir / "generation_config.json"

  if not config_path.exists():
    print("   ⚠️  No generation_config.json found in oxen directory")
    return False

  # Read the config
  with open(config_path) as f:
    config = json.load(f)

  # Copy to local generations directory
  local_config_path = generations_dir_path / "generation_config.json"
  shutil.copy(config_path, local_config_path)
  print(f"   📋 Copied generation_config.json to {local_config_path}")

  # Store in metadata table
  cursor = conn.cursor()
  cursor.execute(
    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
    ("generation_config", json.dumps(config)),
  )
  conn.commit()
  print("   📋 Stored generation_config in database metadata")

  return True


def parse_generation_filename(filename: str) -> tuple[int, int, str] | None:
  """
  Parse a generation filename in the format <xxx>_<yyy>_<hash>.png.

  Args:
    filename: The filename to parse (e.g., "123_456_abc123.png")

  Returns:
    Tuple of (x, y, hash) or None if filename doesn't match expected format
  """
  match = re.match(r"^(-?\d+)_(-?\d+)_([a-f0-9]+)\.png$", filename)
  if match:
    return int(match.group(1)), int(match.group(2)), match.group(3)
  return None


def import_from_oxen(
  generations_dir: str,
  oxen_dataset: str,
) -> None:
  """
  Import quadrants from a locally synced oxen dataset to the local database.

  Only imports files that have changed (based on content hash comparison).
  Scans the generations subdirectory for files named <xxx>_<yyy>_<hash>.png.

  Args:
    generations_dir: Name of the generations directory (e.g., "nyc")
    oxen_dataset: Path to the local oxen dataset directory (e.g., "oxen/isometric-nyc-tiles")
  """
  db_path = get_db_path(generations_dir)
  generations_dir_path = db_path.parent
  oxen_dir = Path(oxen_dataset)

  print(f"📥 Importing from {oxen_dir} to {db_path}")

  # Validate oxen directory exists
  if not oxen_dir.exists():
    raise FileNotFoundError(f"Oxen directory not found: {oxen_dir}")

  # Check for generations directory
  generations_path = oxen_dir / "generations"

  if not generations_path.exists():
    raise FileNotFoundError(f"generations directory not found in {oxen_dir}")

  # Ensure database exists
  ensure_db_exists(db_path)

  # Import generation config and bounds
  conn = sqlite3.connect(db_path)
  import_generation_config(conn, oxen_dir, generations_dir_path)
  conn.close()
  import_bounds(oxen_dir, generations_dir_path)

  # Scan generations directory for PNG files
  print("📋 Scanning generations directory...")
  png_files = list(generations_path.glob("*.png"))
  print(f"   Found {len(png_files)} PNG files")

  # Parse filenames and build list of remote files
  remote_files: list[tuple[int, int, str, Path]] = []
  skipped = 0
  for png_file in png_files:
    parsed = parse_generation_filename(png_file.name)
    if parsed:
      x, y, file_hash = parsed
      remote_files.append((x, y, file_hash, png_file))
    else:
      skipped += 1

  if skipped > 0:
    print(f"   ⚠️  Skipped {skipped} files with unexpected naming format")

  print(f"   Parsed {len(remote_files)} valid generation files")

  # Get local hashes
  print("🔍 Computing local hashes...")
  conn = sqlite3.connect(db_path)
  local_hashes = get_local_hashes(conn)
  print(f"   {len(local_hashes)} quadrants in local database")

  # Determine what needs to be imported
  to_import = []

  for x, y, remote_hash, file_path in remote_files:
    local_hash = local_hashes.get((x, y), "")

    if remote_hash != local_hash:
      to_import.append((x, y, file_path))

  print(f"   {len(to_import)} quadrants to import")

  if not to_import:
    print("✅ No changes to import - local database is up to date")
    conn.close()
    return

  # Import files
  print("📥 Importing changed files...")

  imported_count = 0
  for x, y, file_path in tqdm(to_import, desc="Importing"):
    if file_path.exists():
      gen_blob = file_path.read_bytes()
      update_quadrant(conn, x, y, gen_blob)
      imported_count += 1
    else:
      print(f"\n   ⚠️  File not found: {file_path}")

  conn.commit()
  conn.close()

  print(f"✅ Successfully imported from {oxen_dir}")
  print(f"   Quadrants imported: {imported_count}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Import from a locally synced oxen dataset to the local generations database"
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
    default="oxen/datasets/isometric-nyc-tiles",
    help="Path to the local oxen dataset directory (default: oxen/datasets/isometric-nyc-tiles)",
  )

  args = parser.parse_args()

  import_from_oxen(
    generations_dir=args.generations_dir,
    oxen_dataset=args.oxen_dataset,
  )


if __name__ == "__main__":
  main()
