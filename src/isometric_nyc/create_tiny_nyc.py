"""
Create a tiny subset of a generations directory for testing/development.

This script creates a new generations directory with:
1. A copy of the SQLite database with the same schema
2. Only the quadrants within a specified rectangle (by quadrant x,y coordinates)
3. A copy of generation_config.json

Usage:
  uv run python src/isometric_nyc/create_tiny_nyc.py \
    --source-dir generations/nyc \
    --dest-dir generations/tiny-nyc \
    --tl -10,-10 \
    --br 10,10
"""

import argparse
import shutil
import sqlite3
from pathlib import Path


def parse_coords(coord_str: str) -> tuple[int, int]:
  """Parse a coordinate string like '0,0' or '(0,0)' into (x, y) tuple."""
  # Strip parentheses if present
  coord_str = coord_str.strip("()")
  parts = coord_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid coordinate format: {coord_str}. Expected 'x,y'")
  return int(parts[0].strip()), int(parts[1].strip())


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> str | None:
  """Get the CREATE TABLE statement for a table."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
  )
  result = cursor.fetchone()
  return result[0] if result else None


def get_index_schemas(conn: sqlite3.Connection, table_name: str) -> list[str]:
  """Get all CREATE INDEX statements for a table."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
    (table_name,),
  )
  return [row[0] for row in cursor.fetchall()]


def create_tiny_db(
  source_db_path: Path,
  dest_db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
) -> int:
  """
  Create a subset database with only quadrants in the specified rectangle.

  Args:
      source_db_path: Path to source quadrants.db
      dest_db_path: Path for new database
      tl: Top-left corner (min_x, min_y)
      br: Bottom-right corner (max_x, max_y)

  Returns:
      Number of quadrants copied
  """
  min_x, min_y = tl
  max_x, max_y = br

  # Validate rectangle
  if min_x > max_x or min_y > max_y:
    raise ValueError(
      f"Invalid rectangle: tl=({min_x},{min_y}) must be <= br=({max_x},{max_y})"
    )

  source_conn = sqlite3.connect(source_db_path)
  dest_conn = sqlite3.connect(dest_db_path)

  source_cursor = source_conn.cursor()
  dest_cursor = dest_conn.cursor()

  # Get and create quadrants table schema
  quadrants_schema = get_table_schema(source_conn, "quadrants")
  if not quadrants_schema:
    raise ValueError("Source database missing 'quadrants' table")
  dest_cursor.execute(quadrants_schema)

  # Create indexes for quadrants
  for index_sql in get_index_schemas(source_conn, "quadrants"):
    dest_cursor.execute(index_sql)

  # Create metadata table schema
  metadata_schema = get_table_schema(source_conn, "metadata")
  if metadata_schema:
    dest_cursor.execute(metadata_schema)

  # Create generation_queue table schema (empty, just for compatibility)
  queue_schema = get_table_schema(source_conn, "generation_queue")
  if queue_schema:
    dest_cursor.execute(queue_schema)
    for index_sql in get_index_schemas(source_conn, "generation_queue"):
      dest_cursor.execute(index_sql)

  dest_conn.commit()

  # Get column names from source quadrants table
  source_cursor.execute("PRAGMA table_info(quadrants)")
  columns_info = source_cursor.fetchall()
  # Filter out generated columns (they can't be inserted directly)
  insertable_columns = [
    col[1]
    for col in columns_info
    if col[5] == 0  # col[5] is pk, not generated
  ]
  # Actually need to check for generated columns differently
  source_cursor.execute("SELECT sql FROM sqlite_master WHERE name='quadrants'")
  table_sql = source_cursor.fetchone()[0]
  generated_cols = set()
  if "GENERATED ALWAYS AS" in table_sql:
    # Parse out generated column names
    for col_info in columns_info:
      col_name = col_info[1]
      if f"{col_name} INTEGER GENERATED ALWAYS AS" in table_sql:
        generated_cols.add(col_name)

  insertable_columns = [col[1] for col in columns_info if col[1] not in generated_cols]

  columns_str = ", ".join(insertable_columns)
  placeholders = ", ".join(["?" for _ in insertable_columns])

  # Copy quadrants within rectangle
  source_cursor.execute(
    f"""
        SELECT {columns_str} FROM quadrants
        WHERE quadrant_x >= ? AND quadrant_x <= ?
          AND quadrant_y >= ? AND quadrant_y <= ?
        """,
    (min_x, max_x, min_y, max_y),
  )

  quadrants = source_cursor.fetchall()
  if quadrants:
    dest_cursor.executemany(
      f"INSERT INTO quadrants ({columns_str}) VALUES ({placeholders})",
      quadrants,
    )

  # Copy metadata
  source_cursor.execute("SELECT key, value FROM metadata")
  metadata = source_cursor.fetchall()
  if metadata:
    dest_cursor.executemany(
      "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
      metadata,
    )

  dest_conn.commit()

  source_conn.close()
  dest_conn.close()

  return len(quadrants)


def create_tiny_generation(
  source_dir: Path,
  dest_dir: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
) -> None:
  """
  Create a tiny generation directory with a subset of quadrants.

  Args:
      source_dir: Source generation directory (e.g., generations/nyc)
      dest_dir: Destination directory (e.g., generations/tiny-nyc)
      tl: Top-left corner (min_x, min_y) of rectangle
      br: Bottom-right corner (max_x, max_y) of rectangle
  """
  source_db = source_dir / "quadrants.db"
  source_config = source_dir / "generation_config.json"

  if not source_db.exists():
    raise FileNotFoundError(f"Source database not found: {source_db}")

  # Create destination directory
  dest_dir.mkdir(parents=True, exist_ok=True)

  # Copy generation_config.json if it exists
  if source_config.exists():
    dest_config = dest_dir / "generation_config.json"
    shutil.copy2(source_config, dest_config)
    print("  Copied generation_config.json")

  # Create the subset database
  dest_db = dest_dir / "quadrants.db"
  if dest_db.exists():
    dest_db.unlink()  # Remove existing db

  quadrant_count = create_tiny_db(source_db, dest_db, tl, br)

  print(f"  Created quadrants.db with {quadrant_count} quadrants")


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create a tiny subset of a generations directory.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Create tiny-nyc with quadrants from (-10,-10) to (10,10) = 21x21 grid
  uv run python src/isometric_nyc/create_tiny_nyc.py \\
    --source-dir generations/nyc \\
    --dest-dir generations/tiny-nyc \\
    --tl -10,-10 --br 10,10

  # Create a 10x10 subset starting at (10,10)
  uv run python src/isometric_nyc/create_tiny_nyc.py \\
    --source-dir generations/nyc \\
    --dest-dir generations/subset \\
    --tl 10,10 --br 19,19
        """,
  )
  parser.add_argument(
    "--source-dir",
    type=Path,
    required=True,
    help="Source generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--dest-dir",
    type=Path,
    required=True,
    help="Destination directory for the tiny subset",
  )
  parser.add_argument(
    "--tl",
    type=str,
    required=True,
    help="Top-left corner of rectangle as 'x,y' (e.g., '-10,-10')",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of rectangle as 'x,y' (e.g., '10,10')",
  )

  args = parser.parse_args()

  try:
    tl = parse_coords(args.tl)
    br = parse_coords(args.br)
  except ValueError as e:
    print(f"Error parsing coordinates: {e}")
    return 1

  source_dir = args.source_dir.resolve()
  dest_dir = args.dest_dir.resolve()

  if not source_dir.exists():
    print(f"Error: Source directory not found: {source_dir}")
    return 1

  if not source_dir.is_dir():
    print(f"Error: Not a directory: {source_dir}")
    return 1

  # Calculate expected grid size
  width = br[0] - tl[0] + 1
  height = br[1] - tl[1] + 1

  print("Creating tiny generation subset:")
  print(f"  Source: {source_dir}")
  print(f"  Destination: {dest_dir}")
  print(f"  Rectangle: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  print(f"  Expected grid: {width}x{height} = {width * height} quadrants")
  print()

  try:
    create_tiny_generation(source_dir, dest_dir, tl, br)
    print()
    print(f"Done! Tiny generation created at: {dest_dir}")
    return 0
  except FileNotFoundError as e:
    print(f"Error: {e}")
    return 1
  except ValueError as e:
    print(f"Error: {e}")
    return 1
  except Exception as e:
    print(f"Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())
