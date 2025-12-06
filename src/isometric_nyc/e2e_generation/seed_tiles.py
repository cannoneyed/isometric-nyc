"""
Seed the quadrant database for an e2e generation.

This script reads a generation_config.json and creates a SQLite database
populated with all quadrant entries for the generation. Quadrants are the
atomic units of generation - each tile is divided into a 2x2 grid of quadrants.

Usage:
  uv run python src/isometric_nyc/e2e_generation/seed_tiles.py <generation_dir>

The generation_dir should contain a generation_config.json file with the
generation parameters. The script will create a quadrants.db SQLite database
in the same directory.
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

# Schema version for migrations
SCHEMA_VERSION = 3

# Quadrant layout (2x2 grid within each tile):
# +-------+-------+
# |  TL   |  TR   |  (row 0)
# |  (0)  |  (1)  |
# +-------+-------+
# |  BL   |  BR   |  (row 1)
# |  (2)  |  (3)  |
# +-------+-------+
#   col 0   col 1
#
# Quadrant x/y indices are relative to the seed quadrant (TL of seed tile at 0,0).
# - quadrant_x: positive = right, negative = left
# - quadrant_y: positive = down, negative = up


def calculate_offset(
  lat_center: float,
  lon_center: float,
  shift_x_px: float,
  shift_y_px: float,
  view_height_meters: float,
  viewport_height_px: int,
  azimuth_deg: float,
  elevation_deg: float,
) -> tuple[float, float]:
  """
  Calculate the new lat/lon center after shifting the view by shift_x_px and shift_y_px.

  This is adapted from plan_tiles.py to handle isometric projection offsets.
  """
  meters_per_pixel = view_height_meters / viewport_height_px

  shift_right_meters = shift_x_px * meters_per_pixel
  shift_up_meters = shift_y_px * meters_per_pixel

  elev_rad = math.radians(elevation_deg)
  sin_elev = math.sin(elev_rad)

  if abs(sin_elev) < 1e-6:
    raise ValueError(f"Elevation {elevation_deg} is too close to 0/180.")

  delta_rot_x = shift_right_meters
  delta_rot_y = -shift_up_meters / sin_elev

  azimuth_rad = math.radians(azimuth_deg)
  cos_a = math.cos(azimuth_rad)
  sin_a = math.sin(azimuth_rad)

  delta_east_meters = delta_rot_x * cos_a + delta_rot_y * sin_a
  delta_north_meters = -delta_rot_x * sin_a + delta_rot_y * cos_a

  delta_lat = delta_north_meters / 111111.0
  delta_lon = delta_east_meters / (111111.0 * math.cos(math.radians(lat_center)))

  return lat_center + delta_lat, lon_center + delta_lon


def calculate_quadrant_anchors(
  tile_lat: float,
  tile_lon: float,
  width_px: int,
  height_px: int,
  view_height_meters: float,
  azimuth_deg: float,
  elevation_deg: float,
) -> list[tuple[float, float]]:
  """
  Calculate the anchor coordinates (bottom-right corner) for all 4 quadrants of a tile.

  The tile center is at (tile_lat, tile_lon). Each quadrant's anchor is at
  its bottom-right corner. For the top-left quadrant, this is the tile center.

  Returns list of (lat, lng) tuples in order: [TL, TR, BL, BR]
  """
  half_w = width_px // 2
  half_h = height_px // 2

  # Quadrant anchor offsets from tile center (in pixels)
  # Anchor is at bottom-right of each quadrant
  # TL quadrant: anchor at center (0, 0)
  # TR quadrant: anchor at (+half_w, 0) from center
  # BL quadrant: anchor at (0, -half_h) from center
  # BR quadrant: anchor at (+half_w, -half_h) from center
  anchor_offsets = [
    (0, 0),  # TL - anchor at tile center
    (half_w, 0),  # TR - anchor right of center
    (0, -half_h),  # BL - anchor below center
    (half_w, -half_h),  # BR - anchor right and below center
  ]

  anchors = []
  for shift_x, shift_y in anchor_offsets:
    lat, lng = calculate_offset(
      tile_lat,
      tile_lon,
      shift_x,
      shift_y,
      view_height_meters,
      height_px,
      azimuth_deg,
      elevation_deg,
    )
    anchors.append((lat, lng))

  return anchors


def load_generation_config(generation_dir: Path) -> dict[str, Any]:
  """Load and validate the generation configuration."""
  config_path = generation_dir / "generation_config.json"

  if not config_path.exists():
    raise FileNotFoundError(f"generation_config.json not found in {generation_dir}")

  with open(config_path, "r") as f:
    config = json.load(f)

  # Validate required fields
  required_fields = [
    "name",
    "seed",
    "n_tiles_x",
    "n_tiles_y",
    "camera_azimuth_degrees",
    "camera_elevation_degrees",
    "width_px",
    "height_px",
    "view_height_meters",
  ]

  for field in required_fields:
    if field not in config:
      raise ValueError(f"Missing required field '{field}' in generation_config.json")

  # Validate nested fields
  if "lat" not in config["seed"] or "lng" not in config["seed"]:
    raise ValueError("seed must contain 'lat' and 'lng' fields")

  # Validate tile counts (must be positive integers)
  for dim in ["n_tiles_x", "n_tiles_y"]:
    val = config[dim]
    if not isinstance(val, int) or val < 1:
      raise ValueError(f"{dim} must be a positive integer")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table with (quadrant_x, quadrant_y) as primary key
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT,
      PRIMARY KEY (quadrant_x, quadrant_y)
    )
  """)

  # Create indexes for efficient queries
  # Note: (quadrant_x, quadrant_y) is already indexed as the primary key
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_generated ON quadrants (is_generated)
  """)

  # Create metadata table for schema versioning and config storage
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
      key TEXT PRIMARY KEY,
      value TEXT
    )
  """)

  cursor.execute(
    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
    ("schema_version", str(SCHEMA_VERSION)),
  )

  conn.commit()
  return conn


def calculate_tile_grid(
  config: dict[str, Any],
) -> list[tuple[int, int, float, float]]:
  """
  Calculate tile positions for an n_tiles_x × n_tiles_y grid.

  The top-left tile (row=0, col=0) is centered at the seed lat/lng.
  Tiles extend right (increasing col) and down (increasing row).

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  n_tiles_x = config["n_tiles_x"]
  n_tiles_y = config["n_tiles_y"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  tiles = []

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Generate grid from (0, 0) to (n_tiles_y-1, n_tiles_x-1)
  # Seed tile is at top-left (0, 0)
  for row in range(n_tiles_y):
    for col in range(n_tiles_x):
      if row == 0 and col == 0:
        # Seed tile at top-left
        tiles.append((0, 0, seed_lat, seed_lng))
      else:
        # Calculate pixel shift from seed
        shift_x_px = col * step_x_px
        shift_y_px = -row * step_y_px  # Negative because row increases downward

        lat, lng = calculate_offset(
          seed_lat,
          seed_lng,
          shift_x_px,
          shift_y_px,
          view_height_meters,
          height_px,
          azimuth,
          elevation,
        )

        tiles.append((row, col, lat, lng))

  return tiles


def seed_database(generation_dir: Path) -> None:
  """
  Seed the quadrant database for a generation.

  Reads the generation_config.json, calculates all tile and quadrant positions,
  and populates the SQLite database.
  """
  print(f"📂 Loading generation config from {generation_dir}")
  config = load_generation_config(generation_dir)

  print(f"🏙️  Generation: {config['name']}")
  print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
  print(f"   View height: {config['view_height_meters']}m")
  print(f"   Tile size: {config['width_px']}x{config['height_px']}px")

  # Initialize database
  db_path = generation_dir / "quadrants.db"
  print(f"\n📊 Initializing database at {db_path}")
  conn = init_database(db_path)

  # Store config in metadata
  cursor = conn.cursor()
  cursor.execute(
    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
    ("generation_config", json.dumps(config)),
  )

  # Calculate tile grid
  print("\n🗺️  Calculating tile grid...")
  tiles = calculate_tile_grid(config)
  print(f"   Found {len(tiles)} tiles within bounds")

  # Generate quadrants for each tile
  print("\n🔲 Generating quadrants...")
  quadrants_added = 0
  quadrant_data = []

  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  for row, col, tile_lat, tile_lng in tiles:
    # Calculate anchor positions for all 4 quadrants
    anchors = calculate_quadrant_anchors(
      tile_lat,
      tile_lng,
      width_px,
      height_px,
      view_height_meters,
      azimuth,
      elevation,
    )

    for quadrant_idx, (anchor_lat, anchor_lng) in enumerate(anchors):
      # Calculate quadrant_x and quadrant_y relative to seed quadrant
      # Seed quadrant is TL (index 0) of tile (0, 0) at position (0, 0)
      # Each tile step (with tile_step=0.5) = 1 quadrant
      # quadrant_idx: 0=TL, 1=TR, 2=BL, 3=BR
      # TL adds (0,0), TR adds (1,0), BL adds (0,1), BR adds (1,1)
      dx = quadrant_idx % 2  # 0 for left column, 1 for right column
      dy = quadrant_idx // 2  # 0 for top row, 1 for bottom row

      # Quadrant position relative to seed (TL of tile 0,0 at origin)
      quadrant_x = col + dx
      quadrant_y = row + dy

      quadrant_data.append(
        (
          quadrant_x,
          quadrant_y,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants (using OR IGNORE to skip duplicates)
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index,
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    quadrant_data,
  )

  conn.commit()

  # Report statistics
  cursor.execute("SELECT COUNT(*) FROM quadrants")
  total_quadrants = cursor.fetchone()[0]

  cursor.execute("SELECT COUNT(DISTINCT tile_row || '_' || tile_col) FROM quadrants")
  total_tiles = cursor.fetchone()[0]

  print("\n✅ Database seeded successfully!")
  print(f"   Total tiles: {total_tiles}")
  print(f"   Total quadrants: {total_quadrants}")
  print(f"   Database: {db_path}")

  conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Seed the quadrant database for an e2e generation."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing generation_config.json",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  try:
    seed_database(generation_dir)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())
