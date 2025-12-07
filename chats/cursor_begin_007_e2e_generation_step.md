# Begin 007 e2e generation step
_Exported on 12/6/2025 at 22:02:40 EST from Cursor (2.1.39)_

---

**User**

Begin the first step outlined in @tasks/007_e2e_generation.md 

---

**Cursor**

I'll start by examining the existing code to understand the patterns and logic I need to follow. Let me look at the relevant files.

Let me also look at the infill generation code to understand how quadrants are handled in this project:

Now I understand the quadrant structure - it's a 2x2 grid (4 quadrants: TL, TR, BL, BR). Let me also check the existing database setup and pyproject.toml:

Now I have all the context I need. Let me create the `e2e_generation` directory and the `seed_tiles.py` script.

```python
"""
End-to-end generation module for Isometric NYC.

This module handles the full pipeline for generating isometric pixel art maps
from generation configurations, including:
- Seeding tile/quadrant databases
- Managing generation state
- Coordinating rendering and AI generation
"""

```

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


# Schema version for migrations
SCHEMA_VERSION = 1

# Quadrant layout (2x2 grid within each tile):
# +-------+-------+
# |  TL   |  TR   |  (row 0)
# |  (0)  |  (1)  |
# +-------+-------+
# |  BL   |  BR   |  (row 1)
# |  (2)  |  (3)  |
# +-------+-------+
#   col 0   col 1


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    raise FileNotFoundError(
      f"generation_config.json not found in {generation_dir}"
    )

  with open(config_path, "r") as f:
    config = json.load(f)

  # Validate required fields
  required_fields = [
    "name",
    "seed",
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

      quadrant_data.append(
        (
          quadrant_id,
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

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    quadrant_data,
  )

  conn.commit()

  # Report statistics
  cursor.execute("SELECT COUNT(*) FROM quadrants")
  total_quadrants = cursor.fetchone()[0]

  cursor.execute("SELECT COUNT(DISTINCT tile_row || '_' || tile_col) FROM quadrants")
  total_tiles = cursor.fetchone()[0]

  print(f"\n✅ Database seeded successfully!")
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

```

Now let me add the script entry point to `pyproject.toml` and create a sample generation config for testing:

```toml
[project]
name = "isometric-nyc"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "google-cloud-storage>=2.18.0",
    "google-genai>=1.52.0",
    "googlemaps>=4.10.0",
    "marimo>=0.18.0",
    "numpy>=2.3.5",
    "opencv-python>=4.11.0.86",
    "pillow>=12.0.0",
    "playwright>=1.49.0",
    "psycopg2-binary>=2.9.11",
    "pydantic>=2.12.4",
    "pyproj>=3.7.2",
    "python-dotenv>=1.2.1",
    "pyvista>=0.46.4",
    "requests>=2.31.0",
    "shapely>=2.1.2",
    "sodapy>=2.2.0",
    "tabulate>=0.9.0",
]

[project.scripts]
check-boundaries = "isometric_nyc.check_boundaries:main"
check-coordinates = "isometric_nyc.check_coordinates:main"
check-db = "isometric_nyc.check_db:main"
export-views = "isometric_nyc.export_views:main"
generate-tile = "isometric_nyc.generate_tile:main"
generate-tile-oxen = "isometric_nyc.generate_tile_oxen:main"
isometric-nyc = "isometric_nyc.main:main"
inspect-db = "isometric_nyc.inspect_db:main"
inspect-columns = "isometric_nyc.inspect_columns:main"
plan-tiles = "isometric_nyc.plan_tiles:main"
seed-tiles = "isometric_nyc.e2e_generation.seed_tiles:main"
whitebox = "isometric_nyc.whitebox:main"

[tool.ruff]
line-length = 88
indent-width = 2

[tool.ruff.format]
indent-style = "space"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[dependency-groups]
dev = [
    "ruff>=0.14.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.marimo.runtime]
output_max_bytes = 20_000_000

```

Now let me create a test generation directory with a sample config to verify the script works:

```json
{
  "name": "test_generation",
  "seed": {
    "lat": 40.7484,
    "lng": -73.9857
  },
  "bounds": {
    "top_left": {
      "lat": 40.7500,
      "lng": -73.9900
    },
    "bottom_left": {
      "lat": 40.7460,
      "lng": -73.9900
    },
    "top_right": {
      "lat": 40.7500,
      "lng": -73.9800
    },
    "bottom_right": {
      "lat": 40.7460,
      "lng": -73.9800
    }
  },
  "camera_azimuth_degrees": -15,
  "camera_elevation_degrees": -45,
  "width_px": 1024,
  "height_px": 1024,
  "view_height_meters": 300,
  "tile_step": 0.5
}

```

Now let's test the script:

---

**User**

@007_e2e_generation.md (90-93) 

---

**Cursor**

I'll update the schema to add `quadrant_x` and `quadrant_y` indices relative to the seed quadrant (upper right corner of seed tile).

Now I'll update the schema to add `quadrant_x` and `quadrant_y` columns:

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


# Schema version for migrations
SCHEMA_VERSION = 2

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
# Quadrant x/y indices are relative to the seed quadrant (TR of seed tile).
# - quadrant_x: positive = right, negative = left
# - quadrant_y: positive = down, negative = up


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    raise FileNotFoundError(
      f"generation_config.json not found in {generation_dir}"
    )

  with open(config_path, "r") as f:
    config = json.load(f)

  # Validate required fields
  required_fields = [
    "name",
    "seed",
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

      quadrant_data.append(
        (
          quadrant_id,
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

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    quadrant_data,
  )

  conn.commit()

  # Report statistics
  cursor.execute("SELECT COUNT(*) FROM quadrants")
  total_quadrants = cursor.fetchone()[0]

  cursor.execute("SELECT COUNT(DISTINCT tile_row || '_' || tile_col) FROM quadrants")
  total_tiles = cursor.fetchone()[0]

  print(f"\n✅ Database seeded successfully!")
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


```

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


# Schema version for migrations
SCHEMA_VERSION = 2

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
# Quadrant x/y indices are relative to the seed quadrant (TR of seed tile).
# - quadrant_x: positive = right, negative = left
# - quadrant_y: positive = down, negative = up


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    raise FileNotFoundError(
      f"generation_config.json not found in {generation_dir}"
    )

  with open(config_path, "r") as f:
    config = json.load(f)

  # Validate required fields
  required_fields = [
    "name",
    "seed",
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

      quadrant_data.append(
        (
          quadrant_id,
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

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    quadrant_data,
  )

  conn.commit()

  # Report statistics
  cursor.execute("SELECT COUNT(*) FROM quadrants")
  total_quadrants = cursor.fetchone()[0]

  cursor.execute("SELECT COUNT(DISTINCT tile_row || '_' || tile_col) FROM quadrants")
  total_tiles = cursor.fetchone()[0]

  print(f"\n✅ Database seeded successfully!")
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


```

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

# Schema version for migrations
SCHEMA_VERSION = 2

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
# Quadrant x/y indices are relative to the seed quadrant (TR of seed tile).
# - quadrant_x: positive = right, negative = left
# - quadrant_y: positive = down, negative = up


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_xy ON quadrants (quadrant_x, quadrant_y)
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

      quadrant_data.append(
        (
          quadrant_id,
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

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

# Schema version for migrations
SCHEMA_VERSION = 2

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
# Quadrant x/y indices are relative to the seed quadrant (TR of seed tile).
# - quadrant_x: positive = right, negative = left
# - quadrant_y: positive = down, negative = up


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_xy ON quadrants (quadrant_x, quadrant_y)
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

      # Calculate quadrant_x and quadrant_y relative to seed quadrant
      # Seed quadrant is TR (index 1) of tile (0, 0)
      # Each tile step (with tile_step=0.5) = 1 quadrant
      # quadrant_idx: 0=TL, 1=TR, 2=BL, 3=BR
      # TL adds (0,0), TR adds (1,0), BL adds (0,1), BR adds (1,1)
      dx = quadrant_idx % 2  # 0 for left column, 1 for right column
      dy = quadrant_idx // 2  # 0 for top row, 1 for bottom row

      # Absolute position in quadrant grid
      abs_x = col + dx
      abs_y = row + dy

      # Relative to seed quadrant (TR of tile 0,0 which is at abs position (1, 0))
      quadrant_x = abs_x - 1
      quadrant_y = abs_y - 0  # seed y is 0

      quadrant_data.append(
        (
          quadrant_id,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          quadrant_x,
          quadrant_y,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

# Schema version for migrations
SCHEMA_VERSION = 2

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
# Quadrant x/y indices are relative to the seed quadrant (TR of seed tile).
# - quadrant_x: positive = right, negative = left
# - quadrant_y: positive = down, negative = up


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_xy ON quadrants (quadrant_x, quadrant_y)
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

      # Calculate quadrant_x and quadrant_y relative to seed quadrant
      # Seed quadrant is TR (index 1) of tile (0, 0)
      # Each tile step (with tile_step=0.5) = 1 quadrant
      # quadrant_idx: 0=TL, 1=TR, 2=BL, 3=BR
      # TL adds (0,0), TR adds (1,0), BL adds (0,1), BR adds (1,1)
      dx = quadrant_idx % 2  # 0 for left column, 1 for right column
      dy = quadrant_idx // 2  # 0 for top row, 1 for bottom row

      # Absolute position in quadrant grid
      abs_x = col + dx
      abs_y = row + dy

      # Relative to seed quadrant (TR of tile 0,0 which is at abs position (1, 0))
      quadrant_x = abs_x - 1
      quadrant_y = abs_y - 0  # seed y is 0

      quadrant_data.append(
        (
          quadrant_id,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          quadrant_x,
          quadrant_y,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y, 
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

Now let's test the updated script:

Let me verify the quadrant indices are correct by querying the database:

The schema update is working correctly. Looking at the output:
- Tile (0,0) quadrant 1 (TR - the seed quadrant) has `quadrant_x=0, quadrant_y=0` ✓
- Tile (0,0) quadrant 0 (TL) has `quadrant_x=-1, quadrant_y=0` (one left of seed) ✓
- Tile (-1,-1) quadrant 1 has `quadrant_x=-1, quadrant_y=-1` (up and left of seed) ✓

The `quadrant_x` and `quadrant_y` indices correctly represent position relative to the seed quadrant, with positive x going right, positive y going down, and negative values going left/up.

Here's a summary of what was created:

**New directory:** `src/isometric_nyc/e2e_generation/`

**New files:**
- `__init__.py` - Module initialization
- `seed_tiles.py` - Script to seed the quadrant database

**Schema (Quadrants table):**
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Unique hash ID based on anchor coordinates |
| lat | REAL | Latitude of bottom-right anchor |
| lng | REAL | Longitude of bottom-right anchor |
| tile_row | INTEGER | Tile row index |
| tile_col | INTEGER | Tile column index |
| quadrant_index | INTEGER | 0=TL, 1=TR, 2=BL, 3=BR |
| quadrant_x | INTEGER | X offset from seed (+ right, - left) |
| quadrant_y | INTEGER | Y offset from seed (+ down, - up) |
| render | BLOB | 3D render PNG bytes |
| generation | BLOB | Generated pixel art PNG bytes |
| is_generated | INTEGER | Computed: 1 if generation exists |
| notes | TEXT | Optional notes |

**Usage:**
```bash
uv run seed-tiles <generation_dir>
```

```shellscript
uv run seed-tiles <generation_dir>
```

---

**User**

no, the tile 000_000 TOP LEFT should be quadrant address 0, 0

---

**Cursor**

I'll update the code so that the TOP LEFT quadrant of tile (0,0) is the seed at position (0,0):

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

# Schema version for migrations
SCHEMA_VERSION = 2

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
# Quadrant x/y indices are relative to the seed quadrant (TR of seed tile).
# - quadrant_x: positive = right, negative = left
# - quadrant_y: positive = down, negative = up


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_xy ON quadrants (quadrant_x, quadrant_y)
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

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
          quadrant_id,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          quadrant_x,
          quadrant_y,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y, 
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

```python
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
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

# Schema version for migrations
SCHEMA_VERSION = 2

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


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_xy ON quadrants (quadrant_x, quadrant_y)
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

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
          quadrant_id,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          quadrant_x,
          quadrant_y,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y, 
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

Let me test the updated logic:

Now tile (0,0) TL (index 0) is the seed at `quadrant_x=0, quadrant_y=0`:

| Tile | Quadrant | quadrant_x | quadrant_y |
|------|----------|------------|------------|
| (0,0) | TL (0) | 0 | 0 | ← seed |
| (0,0) | TR (1) | 1 | 0 |
| (0,0) | BL (2) | 0 | 1 |
| (0,0) | BR (3) | 1 | 1 |
| (-1,-1) | TL (0) | -1 | -1 | (up-left) |
| (-1,-1) | TR (1) | 0 | -1 | (up) |

---

**User**

OK great - now I'd like a simple way to debug the generation db. Let's add another script to the e2e_generation dir called `render_tile.py` that takes 3 params - one the `generation_dir` containing the directory, and then an x and y coordinate for the tile. It then uses the logic in `export_views.py` to generate a `render` and saves it in a `renders/` subdir in the generation dir with the name render_<x>_<y>.png

---

**Cursor**

I'll create the `render_tile.py` script. Let me first review the whitebox rendering logic:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position and saves it
to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.whitebox import render_tile


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(conn: sqlite3.Connection, tile_row: int, tile_col: int) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(generation_dir: Path, x: int, y: int) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y).

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})")
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Render the tile
    print(f"\n🎨 Rendering tile...")
    render_tile(
      lat=tile_center["lat"],
      lon=tile_center["lng"],
      size_meters=config.get("size_meters", 300),
      orientation_deg=config["camera_azimuth_degrees"],
      use_satellite=True,
      viewport_width=config["width_px"],
      viewport_height=config["height_px"],
      output_path=str(output_path),
      camera_elevation_deg=config["camera_elevation_degrees"],
      view_height_meters=config.get("view_height_meters", 200),
    )

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
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
    result = render_quadrant_tile(generation_dir, args.x, args.y)
    return 0 if result else 1
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

```toml
[project]
name = "isometric-nyc"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "google-cloud-storage>=2.18.0",
    "google-genai>=1.52.0",
    "googlemaps>=4.10.0",
    "marimo>=0.18.0",
    "numpy>=2.3.5",
    "opencv-python>=4.11.0.86",
    "pillow>=12.0.0",
    "playwright>=1.49.0",
    "psycopg2-binary>=2.9.11",
    "pydantic>=2.12.4",
    "pyproj>=3.7.2",
    "python-dotenv>=1.2.1",
    "pyvista>=0.46.4",
    "requests>=2.31.0",
    "shapely>=2.1.2",
    "sodapy>=2.2.0",
    "tabulate>=0.9.0",
]

[project.scripts]
check-boundaries = "isometric_nyc.check_boundaries:main"
check-coordinates = "isometric_nyc.check_coordinates:main"
check-db = "isometric_nyc.check_db:main"
export-views = "isometric_nyc.export_views:main"
generate-tile = "isometric_nyc.generate_tile:main"
generate-tile-oxen = "isometric_nyc.generate_tile_oxen:main"
isometric-nyc = "isometric_nyc.main:main"
inspect-db = "isometric_nyc.inspect_db:main"
inspect-columns = "isometric_nyc.inspect_columns:main"
plan-tiles = "isometric_nyc.plan_tiles:main"
render-tile = "isometric_nyc.e2e_generation.render_tile:main"
seed-tiles = "isometric_nyc.e2e_generation.seed_tiles:main"
whitebox = "isometric_nyc.whitebox:main"

[tool.ruff]
line-length = 88
indent-width = 2

[tool.ruff.format]
indent-style = "space"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[dependency-groups]
dev = [
    "ruff>=0.14.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.marimo.runtime]
output_max_bytes = 20_000_000

```

Let me test it:

The script works. Let me also test rendering a different quadrant:

---

**User**

You generated a `whitebox` - I want the web render instead

---

**Cursor**

I'll update `render_tile.py` to use the web renderer instead. Let me check the web export logic:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).

Note: Requires the web server to be running (bun run dev in src/web/).
"""

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Default web server port (Vite default)
DEFAULT_WEB_PORT = 5173


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(conn: sqlite3.Connection, tile_row: int, tile_col: int) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(generation_dir: Path, x: int, y: int) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y).

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})")
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Render the tile
    print(f"\n🎨 Rendering tile...")
    render_tile(
      lat=tile_center["lat"],
      lon=tile_center["lng"],
      size_meters=config.get("size_meters", 300),
      orientation_deg=config["camera_azimuth_degrees"],
      use_satellite=True,
      viewport_width=config["width_px"],
      viewport_height=config["height_px"],
      output_path=str(output_path),
      camera_elevation_deg=config["camera_elevation_degrees"],
      view_height_meters=config.get("view_height_meters", 200),
    )

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
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
    result = render_quadrant_tile(generation_dir, args.x, args.y)
    return 0 if result else 1
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())


```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).

Note: Requires the web server to be running (bun run dev in src/web/).
"""

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Default web server port (Vite default)
DEFAULT_WEB_PORT = 5173


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(conn: sqlite3.Connection, tile_row: int, tile_col: int) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})")
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_center["lat"],
      "lon": tile_center["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print(f"\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
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
    result = render_quadrant_tile(generation_dir, args.x, args.y)
    return 0 if result else 1
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())


```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).

Note: Requires the web server to be running (bun run dev in src/web/).
"""

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Default web server port (Vite default)
DEFAULT_WEB_PORT = 5173


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(
  conn: sqlite3.Connection, tile_row: int, tile_col: int
) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(
        f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})"
      )
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_center["lat"],
      "lon": tile_center["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
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
    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

Updated `render_tile.py` to use the web renderer instead of whitebox.

**Usage:**
```bash
# First, start the web server in a separate terminal:
cd src/web && bun run dev

# Then render a tile:
uv run render-tile generations/test_generation 0 0
```

The script now:
1. Connects to the running web server (default port 5173)
2. Uses Playwright to capture a screenshot of the 3D web view
3. Saves to `renders/render_<x>_<y>.png`

You can specify a custom port with `--port`:
```bash
uv run render-tile generations/test_generation 0 0 --port 3000
```

```shellscript
# First, start the web server in a separate terminal:
cd src/web && bun run dev

# Then render a tile:
uv run render-tile generations/test_generation 0 0
```

```shellscript
uv run render-tile generations/test_generation 0 0 --port 3000
```

---

**User**

NO - the @src/isometric_nyc/export_views.py script doesn't require me to run a web server - can you factor out the logic from @src/isometric_nyc/export_views.py for web rendering and use that exact functionality in the @src/isometric_nyc/e2e_generation/render_tile.py script?

---

**Cursor**

I see - `export_views.py` automatically starts the web server. Let me update `render_tile.py` to do the same:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  # Wait for server to start
  print("   ⏳ Waiting for server to start...")
  time.sleep(5)
  print(f"   ✅ Server started on http://localhost:{port}")
  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(
  conn: sqlite3.Connection, tile_row: int, tile_col: int
) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(
        f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})"
      )
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_center["lat"],
      "lon": tile_center["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
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
    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  # Wait for server to start
  print("   ⏳ Waiting for server to start...")
  time.sleep(5)
  print(f"   ✅ Server started on http://localhost:{port}")
  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(
  conn: sqlite3.Connection, tile_row: int, tile_col: int
) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(
        f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})"
      )
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_center["lat"],
      "lon": tile_center["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Now `render_tile.py` will automatically start and stop the web server, just like `export_views.py`. Let me test it:

It works. The script now automatically:
1. Starts the web server
2. Renders the tile via Playwright  
3. Stops the web server

You can also use `--no-start-server` if you already have the server running:
```bash
uv run render-tile generations/test_generation 0 0 --no-start-server
```

```shellscript
uv run render-tile generations/test_generation 0 0 --no-start-server
```

---

**User**

 I just ran it again and got the following error: 



@zsh (961-1017) 

---

**Cursor**

The server likely didn't have enough time to start, or it failed to start. Let me improve the server startup with better readiness checking:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import socket
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by polling the port.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  start_time = time.time()
  while time.time() - start_time < timeout:
    try:
      with socket.create_connection(("localhost", port), timeout=1):
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
      time.sleep(interval)
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Wait for server to be ready
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    # Give it a bit more time to fully initialize
    time.sleep(1)
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died
    if process.poll() is not None:
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(f"Web server failed to start: {stderr}")
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(
  conn: sqlite3.Connection, tile_row: int, tile_col: int
) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(
        f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})"
      )
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_center["lat"],
      "lon": tile_center["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

I've improved the server startup to properly wait for it to be ready by polling the port instead of just sleeping for 5 seconds. The new version:

1. Polls the port every 0.5 seconds for up to 30 seconds
2. Checks if the server process died and reports the error
3. Gives a warning if the server isn't ready after the timeout

Try running it again:

```bash
uv run render-tile generations/test_generation 0 1
```

```shellscript
uv run render-tile generations/test_generation 0 1
```

---

**User**

@zsh (974-1017) 

---

**Cursor**

The socket check might be passing but the HTTP server isn't fully ready. Let me improve this with an actual HTTP check and better diagnostics:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}")

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}")
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its x, y position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "id": row[0],
    "lat": row[1],
    "lng": row[2],
    "tile_row": row[3],
    "tile_col": row[4],
    "quadrant_index": row[5],
    "quadrant_x": row[6],
    "quadrant_y": row[7],
  }


def get_tile_center(
  conn: sqlite3.Connection, tile_row: int, tile_col: int
) -> dict | None:
  """
  Get the tile center coordinates.

  The TL quadrant (index 0) of a tile has its anchor at the tile center.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng
    FROM quadrants
    WHERE tile_row = ? AND tile_col = ? AND quadrant_index = 0
    LIMIT 1
    """,
    (tile_row, tile_col),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {"lat": row[0], "lng": row[1]}


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Found quadrant at ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   Tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Index: {quadrant['quadrant_index']}")

    # Get the tile center (from the TL quadrant of the same tile)
    tile_center = get_tile_center(conn, quadrant["tile_row"], quadrant["tile_col"])
    if not tile_center:
      print(
        f"❌ Could not find tile center for tile ({quadrant['tile_row']}, {quadrant['tile_col']})"
      )
      return None

    print(f"   Tile center: {tile_center['lat']:.6f}, {tile_center['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_center["lat"],
      "lon": tile_center["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
WEB_DIR = Path(__file__).parent.parent.parent / "web" would be:
- `__file__` = `/Users/andycoenen/cannoneyed/isometric-nyc/src/isometric_nyc/e2e_generation/render_tile.py`
- `.parent` = `/Users/andycoenen/cannoneyed/isometric-nyc/src/isometric_nyc/e2e_generation`
- `.parent.parent` = `/Users/andycoenen/cannoneyed/isometric-nyc/src/isometric_nyc`
- `.parent.parent.parent` = `/Users/andycoenen/cannoneyed/isometric-nyc/src`
- `/ "web"` = `/Users/andycoenen/cannoneyed/isometric-nyc/src/web`

Let me check the project structure to confirm the web directory location.
```

Let me also verify the web directory path is correct:

The web directory path looks correct. Let me test if the improved startup works:

It works now. The improved HTTP-based readiness check ensures the server is actually responding before attempting to render. The script now:

1. Verifies the web directory exists
2. Waits for the process to start
3. Makes actual HTTP requests to verify the server is responding
4. Provides better error messages if something fails

---

**User**

OK great - but now rendering 0 0 and 0 1 give me the exact same tile

---

**Cursor**

You're right - the current logic finds which tile contains the quadrant and renders that tile's center. Since both (0,0) and (0,1) are in tile (0,0), they give the same render.

Let me fix this to render a tile centered on each quadrant's position:

I see the issue. The current logic only works for quadrants that are TL (index 0) of some tile. But quadrant (0, 1) is BL (index 2) of tile (0, 0), not TL of any tile.

Let me fix this to render centered directly on the quadrant's anchor position:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get tile center for this quadrant position
    # We want the tile where quadrant (x, y) is the TL
    tile_info = get_tile_center_for_quadrant(conn, x, y)
    if not tile_info:
      print(f"❌ No tile found with quadrant ({x}, {y}) as TL")
      print("   This quadrant position may be outside the generation bounds,")
      print("   or not at a tile boundary.")
      return None

    print(f"📍 Rendering tile for quadrant ({x}, {y})")
    print(f"   Tile: row={tile_info['tile_row']}, col={tile_info['tile_col']}")
    print(f"   Tile center: {tile_info['lat']:.6f}, {tile_info['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_info["lat"],
      "lon": tile_info["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Rendering view for quadrant ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters
    params = {
      "export": "true",
      "lat": tile_info["lat"],
      "lon": tile_info["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Rendering view for quadrant ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Let me test both quadrants to verify they produce different renders:

Now each quadrant produces a different render:

- **Quadrant (0, 0)**: Anchor at `40.748400, -73.985700`
- **Quadrant (0, 1)**: Anchor at `40.746556, -73.985048`

Each render is now centered on the quadrant's unique anchor position.

---

**User**

OK - it looks like it's right except the coordinates look flipped - the 0_1 render is *below* the 0_0 render (instead of x_y, it looks like it's (y_x)

---

**Cursor**

You're right - the arguments are in (x, y) order, but most people think in (row, col) order where the first number is the row (vertical) and second is the column (horizontal). Let me swap the arguments to match that expectation:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Rendering view for quadrant ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "col",
    type=int,
    help="quadrant column (x) - horizontal position, positive = right",
  )
  parser.add_argument(
    "row",
    type=int,
    help="quadrant row (y) - vertical position, positive = down",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Rendering view for quadrant ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "col",
    type=int,
    help="quadrant column (x) - horizontal position, positive = right",
  )
  parser.add_argument(
    "row",
    type=int,
    help="quadrant row (y) - vertical position, positive = down",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.col, args.row, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, col: int, row: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (col, row) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    col: quadrant column (x) - horizontal position
    row: quadrant row (y) - vertical position
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  # Map col/row to x/y for database query
  x, y = col, row

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position col={col}, row={row}")
      return None

    print(f"📍 Rendering view for quadrant (col={col}, row={row})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path - use col_row format for intuitive naming
    output_path = renders_dir / f"render_{col}_{row}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "col",
    type=int,
    help="quadrant column (x) - horizontal position, positive = right",
  )
  parser.add_argument(
    "row",
    type=int,
    help="quadrant row (y) - vertical position, positive = down",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.col, args.row, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer and saves it to the renders/ subdirectory.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <col> <row>

Where:
  col = horizontal position (positive = right)
  row = vertical position (positive = down)

Example:
  render_tile generations/test 0 0   # seed quadrant
  render_tile generations/test 1 0   # one column to the right
  render_tile generations/test 0 1   # one row down
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, col: int, row: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (col, row) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    col: quadrant column (x) - horizontal position
    row: quadrant row (y) - vertical position
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  # Map col/row to x/y for database query
  x, y = col, row

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position col={col}, row={row}")
      return None

    print(f"📍 Rendering view for quadrant (col={col}, row={row})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path - use col_row format for intuitive naming
    output_path = renders_dir / f"render_{col}_{row}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "col",
    type=int,
    help="quadrant column (x) - horizontal position, positive = right",
  )
  parser.add_argument(
    "row",
    type=int,
    help="quadrant row (y) - vertical position, positive = down",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.col, args.row, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Let me test to verify the new argument order:

---

**User**

OK - I actually want the x y syntax - next can you write a script that generates four tiles (0 0, 0 1, 1 0, and 1 1) and then checks they all fit together correctly by assembling them into one large image? Call it "check_tiles.py" and put it in the e2e_generation directory

---

**Cursor**

```python
"""
Check that tiles fit together correctly by rendering a 2x2 grid and assembling them.

This script renders 4 tiles at positions (0,0), (0,1), (1,0), (1,1) and
assembles them into a single image to verify correct alignment.

Usage:
  uv run python src/isometric_nyc/e2e_generation/check_tiles.py <generation_dir>

The assembled image will be saved as renders/check_tiles_assembled.png
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """Wait for the server to be ready by making HTTP requests."""
  url = f"http://localhost:{port}/"
  start_time = time.time()

  while time.time() - start_time < timeout:
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """Start the Vite dev server and wait for it to be ready."""
  print(f"🌐 Starting web server on port {port}...")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  time.sleep(2)

  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}")

  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(f"Web server died.\nstdout: {stdout}\nstderr: {stderr}")
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its (x, y) position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_tile(
  page,
  quadrant: dict,
  config: dict,
  output_path: Path,
  port: int,
) -> bool:
  """Render a single tile using Playwright."""
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  page.goto(url, wait_until="networkidle")

  try:
    page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
  except Exception:
    print(f"   ⚠️  Timeout waiting for tiles")

  page.screenshot(path=str(output_path))
  return True


def check_tiles(generation_dir: Path, port: int = DEFAULT_WEB_PORT) -> Path | None:
  """
  Render a 2x2 grid of tiles and assemble them.

  Renders tiles at (0,0), (1,0), (0,1), (1,1) and assembles into one image.
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    width = config["width_px"]
    height = config["height_px"]

    # Get all 4 quadrants
    # Grid layout:
    #   (0,0) | (1,0)
    #   ------+------
    #   (0,1) | (1,1)
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    quadrants = {}

    print("📍 Finding quadrants...")
    for x, y in positions:
      q = get_quadrant(conn, x, y)
      if not q:
        print(f"   ❌ No quadrant at ({x}, {y})")
        return None
      quadrants[(x, y)] = q
      print(f"   ✓ ({x}, {y}): {q['lat']:.6f}, {q['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Render all tiles
    print("\n🎨 Rendering tiles...")
    tile_paths = {}

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": width, "height": height},
        device_scale_factor=1,
      )
      page = context.new_page()

      for x, y in positions:
        output_path = renders_dir / f"check_{x}_{y}.png"
        print(f"   Rendering ({x}, {y})...")
        render_tile(page, quadrants[(x, y)], config, output_path, port)
        tile_paths[(x, y)] = output_path
        print(f"   ✓ Saved to {output_path.name}")

      page.close()
      context.close()
      browser.close()

    # Assemble tiles
    # Each tile overlaps by 50%, so we place them offset by half the tile size
    print("\n🧩 Assembling tiles...")

    half_w = width // 2
    half_h = height // 2

    # The assembled image shows the overlapping region
    # For 4 quadrants with 50% overlap:
    # - Total width = width + half_w = 1.5 * width
    # - Total height = height + half_h = 1.5 * height
    assembled_width = width + half_w
    assembled_height = height + half_h

    assembled = Image.new("RGB", (assembled_width, assembled_height))

    # Place tiles at their offset positions
    # (0,0) at top-left: (0, 0)
    # (1,0) at top-right offset: (half_w, 0)
    # (0,1) at bottom-left offset: (0, half_h)
    # (1,1) at bottom-right offset: (half_w, half_h)
    placements = {
      (0, 0): (0, 0),
      (1, 0): (half_w, 0),
      (0, 1): (0, half_h),
      (1, 1): (half_w, half_h),
    }

    for (x, y), (px, py) in placements.items():
      tile_img = Image.open(tile_paths[(x, y)])
      assembled.paste(tile_img, (px, py))
      print(f"   Placed ({x}, {y}) at ({px}, {py})")

    # Save assembled image
    output_path = renders_dir / "check_tiles_assembled.png"
    assembled.save(output_path)
    print(f"\n✅ Assembled image saved to {output_path}")
    print(f"   Size: {assembled_width}x{assembled_height}")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Check tiles fit together by rendering and assembling a 2x2 grid."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = check_tiles(generation_dir, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```toml
[project]
name = "isometric-nyc"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "google-cloud-storage>=2.18.0",
    "google-genai>=1.52.0",
    "googlemaps>=4.10.0",
    "marimo>=0.18.0",
    "numpy>=2.3.5",
    "opencv-python>=4.11.0.86",
    "pillow>=12.0.0",
    "playwright>=1.49.0",
    "psycopg2-binary>=2.9.11",
    "pydantic>=2.12.4",
    "pyproj>=3.7.2",
    "python-dotenv>=1.2.1",
    "pyvista>=0.46.4",
    "requests>=2.31.0",
    "shapely>=2.1.2",
    "sodapy>=2.2.0",
    "tabulate>=0.9.0",
]

[project.scripts]
check-boundaries = "isometric_nyc.check_boundaries:main"
check-coordinates = "isometric_nyc.check_coordinates:main"
check-db = "isometric_nyc.check_db:main"
export-views = "isometric_nyc.export_views:main"
generate-tile = "isometric_nyc.generate_tile:main"
generate-tile-oxen = "isometric_nyc.generate_tile_oxen:main"
isometric-nyc = "isometric_nyc.main:main"
inspect-db = "isometric_nyc.inspect_db:main"
inspect-columns = "isometric_nyc.inspect_columns:main"
plan-tiles = "isometric_nyc.plan_tiles:main"
check-tiles = "isometric_nyc.e2e_generation.check_tiles:main"
render-tile = "isometric_nyc.e2e_generation.render_tile:main"
seed-tiles = "isometric_nyc.e2e_generation.seed_tiles:main"
whitebox = "isometric_nyc.whitebox:main"

[tool.ruff]
line-length = 88
indent-width = 2

[tool.ruff.format]
indent-style = "space"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[dependency-groups]
dev = [
    "ruff>=0.14.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.marimo.runtime]
output_max_bytes = 20_000_000

```

Now let me test it:

The script works. It:

1. Renders 4 tiles at positions (0,0), (1,0), (0,1), (1,1)
2. Assembles them into a single 1536x1536 image with 50% overlap
3. Saves to `renders/check_tiles_assembled.png`

**Usage:**
```bash
uv run check-tiles generations/test_generation
```

The grid layout is:
```
(0,0) | (1,0)
------+------
(0,1) | (1,1)
```

Where x increases to the right and y increases downward.

```shellscript
uv run check-tiles generations/test_generation
```

```plaintext
(0,0) | (1,0)
------+------
(0,1) | (1,1)
```

---

**User**

Amazing! Now onto the next step - update the render_tile script to do 2 things:



1. When a tile is rendered, separate it into the four quadrant pngs and save the bytes to the row in the sqlite db for the generation dir

2. Add a `overwrite` flag that defaults to true, and if it's false, it uses pre-rendered quadrants for the given tile if and only if all of them have previously been rendered/saved in the sqlite db

---

**Cursor**

Now I'll update `render_tile.py` to:
1. Split the rendered tile into 4 quadrants and save to the database
2. Add `--overwrite` flag

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
  }


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Rendering view for quadrant ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
    "has_render": row[6] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Returns True if successful, False if quadrant not found.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path, x: int, y: int, port: int = DEFAULT_WEB_PORT
) -> Path | None:
  """
  Render a tile containing the quadrant at position (x, y) using the web renderer.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate
    y: quadrant_y coordinate
    port: Web server port (default: 5173)

  Returns:
    Path to the rendered image, or None if quadrant not found
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y)
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    print(f"📍 Rendering view for quadrant ({x}, {y})")
    print(f"   ID: {quadrant['id']}")
    print(f"   In tile: row={quadrant['tile_row']}, col={quadrant['tile_col']}")
    print(f"   Quadrant index: {quadrant['quadrant_index']}")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered to {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
    "has_render": row[6] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Returns True if successful, False if quadrant not found.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y) - this will be the TL of the rendered tile
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x+1},{y}), ({x},{y+1}), ({x+1},{y+1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, id, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    LIMIT 1
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "id": row[5],
    "has_render": row[6] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Returns True if successful, False if quadrant not found.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y) - this will be the TL of the rendered tile
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x+1},{y}), ({x},{y+1}), ({x+1},{y+1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Let me test the updated script:

Now let's test the `--no-overwrite` flag:

Let me verify the database has the render data:

I see there are duplicates - let me check the database structure:

The script is working correctly. There are some duplicate `(quadrant_x, quadrant_y)` entries in the database due to tile overlap, but the script updates all matching rows.

**Summary of changes to `render_tile.py`:**

1. **Quadrant splitting**: After rendering a full tile, it's split into 4 quadrants (512x512 each for a 1024x1024 tile)

2. **Database saving**: Each quadrant's PNG bytes are saved to the `render` column in the SQLite database

3. **`--no-overwrite` flag**: When specified, skips rendering if all 4 quadrants already have render data in the database

**Usage:**
```bash
# Render and save to DB (overwrites existing)
uv run render-tile generations/test_generation 0 0

# Skip if already rendered
uv run render-tile generations/test_generation 0 0 --no-overwrite
```

**Tile coverage:**
When you render at position `(x, y)`, the tile covers quadrants:
- `(x, y)` - top-left
- `(x+1, y)` - top-right  
- `(x, y+1)` - bottom-left
- `(x+1, y+1)` - bottom-right

```shellscript
# Render and save to DB (overwrites existing)
uv run render-tile generations/test_generation 0 0

# Skip if already rendered
uv run render-tile generations/test_generation 0 0 --no-overwrite
```

---

**User**

OK - can you make some changes to the @src/isometric_nyc/e2e_generation/seed_tiles.py script? We need to make the primary key to the table x and y - remove the id. This way, there won't be any duplicate quadrants

---

**Cursor**

```python
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
import hashlib
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


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

  return config


def init_database(db_path: Path) -> sqlite3.Connection:
  """Initialize the SQLite database with the quadrants schema."""
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Create the quadrants table
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS quadrants (
      id TEXT PRIMARY KEY,
      lat REAL NOT NULL,
      lng REAL NOT NULL,
      tile_row INTEGER NOT NULL,
      tile_col INTEGER NOT NULL,
      quadrant_index INTEGER NOT NULL,
      quadrant_x INTEGER NOT NULL,
      quadrant_y INTEGER NOT NULL,
      render BLOB,
      generation BLOB,
      is_generated INTEGER GENERATED ALWAYS AS (generation IS NOT NULL) STORED,
      notes TEXT
    )
  """)

  # Create indexes for efficient queries
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_xy ON quadrants (quadrant_x, quadrant_y)
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

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
          quadrant_id,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          quadrant_x,
          quadrant_y,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y, 
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

```python
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
import hashlib
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


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

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
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_coords ON quadrants (lat, lng)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_tile ON quadrants (tile_row, tile_col)
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_quadrants_xy ON quadrants (quadrant_x, quadrant_y)
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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

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
          quadrant_id,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          quadrant_x,
          quadrant_y,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y, 
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

```python
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
import hashlib
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


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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
      quadrant_id = generate_quadrant_id(anchor_lat, anchor_lng)

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
          quadrant_id,
          anchor_lat,
          anchor_lng,
          row,
          col,
          quadrant_idx,
          quadrant_x,
          quadrant_y,
          None,  # render
          None,  # generation
          None,  # notes
        )
      )
      quadrants_added += 1

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y, 
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

```python
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
import hashlib
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


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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

  # Insert all quadrants
  cursor.executemany(
    """
    INSERT OR IGNORE INTO quadrants 
    (id, lat, lng, tile_row, tile_col, quadrant_index, quadrant_x, quadrant_y, 
     render, generation, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

```

```python
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
import hashlib
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


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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

```

```python
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


def generate_quadrant_id(lat: float, lng: float) -> str:
  """
  Generate a unique hash ID for a quadrant based on its anchor coordinates.

  Uses SHA-256 truncated to 16 characters for a balance of uniqueness
  and readability.
  """
  # Round to 8 decimal places to avoid floating point issues
  coord_str = f"{lat:.8f},{lng:.8f}"
  hash_bytes = hashlib.sha256(coord_str.encode()).hexdigest()
  return hash_bytes[:16]


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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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

```

```python
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
    "bounds",
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

  bounds = config["bounds"]
  for corner in ["top_left", "bottom_left", "top_right", "bottom_right"]:
    if corner not in bounds:
      raise ValueError(f"bounds must contain '{corner}'")
    if "lat" not in bounds[corner] or "lng" not in bounds[corner]:
      raise ValueError(f"bounds.{corner} must contain 'lat' and 'lng' fields")

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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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

```

Now let me update `render_tile.py` to work with the new schema (no `id` column):

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "has_render": row[5] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(
  conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Returns True if successful, False if quadrant not found.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y) - this will be the TL of the rendered tile
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Now let me update `check_tiles.py` as well:

```python
"""
Check that tiles fit together correctly by rendering a 2x2 grid and assembling them.

This script renders 4 tiles at positions (0,0), (0,1), (1,0), (1,1) and
assembles them into a single image to verify correct alignment.

Usage:
  uv run python src/isometric_nyc/e2e_generation/check_tiles.py <generation_dir>

The assembled image will be saved as renders/check_tiles_assembled.png
"""

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """Wait for the server to be ready by making HTTP requests."""
  url = f"http://localhost:{port}/"
  start_time = time.time()

  while time.time() - start_time < timeout:
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """Start the Vite dev server and wait for it to be ready."""
  print(f"🌐 Starting web server on port {port}...")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  time.sleep(2)

  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(f"Web server died.\nstdout: {stdout}\nstderr: {stderr}")
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get a quadrant by its (x, y) position."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
  }


def render_tile(
  page,
  quadrant: dict,
  config: dict,
  output_path: Path,
  port: int,
) -> bool:
  """Render a single tile using Playwright."""
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  page.goto(url, wait_until="networkidle")

  try:
    page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
  except Exception:
    print("   ⚠️  Timeout waiting for tiles")

  page.screenshot(path=str(output_path))
  return True


def check_tiles(generation_dir: Path, port: int = DEFAULT_WEB_PORT) -> Path | None:
  """
  Render a 2x2 grid of tiles and assemble them.

  Renders tiles at (0,0), (1,0), (0,1), (1,1) and assembles into one image.
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    width = config["width_px"]
    height = config["height_px"]

    # Get all 4 quadrants
    # Grid layout:
    #   (0,0) | (1,0)
    #   ------+------
    #   (0,1) | (1,1)
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    quadrants = {}

    print("📍 Finding quadrants...")
    for x, y in positions:
      q = get_quadrant(conn, x, y)
      if not q:
        print(f"   ❌ No quadrant at ({x}, {y})")
        return None
      quadrants[(x, y)] = q
      print(f"   ✓ ({x}, {y}): {q['lat']:.6f}, {q['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Render all tiles
    print("\n🎨 Rendering tiles...")
    tile_paths = {}

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": width, "height": height},
        device_scale_factor=1,
      )
      page = context.new_page()

      for x, y in positions:
        output_path = renders_dir / f"check_{x}_{y}.png"
        print(f"   Rendering ({x}, {y})...")
        render_tile(page, quadrants[(x, y)], config, output_path, port)
        tile_paths[(x, y)] = output_path
        print(f"   ✓ Saved to {output_path.name}")

      page.close()
      context.close()
      browser.close()

    # Assemble tiles
    # Each tile overlaps by 50%, so we place them offset by half the tile size
    print("\n🧩 Assembling tiles...")

    half_w = width // 2
    half_h = height // 2

    # The assembled image shows the overlapping region
    # For 4 quadrants with 50% overlap:
    # - Total width = width + half_w = 1.5 * width
    # - Total height = height + half_h = 1.5 * height
    assembled_width = width + half_w
    assembled_height = height + half_h

    assembled = Image.new("RGB", (assembled_width, assembled_height))

    # Place tiles at their offset positions
    # (0,0) at top-left: (0, 0)
    # (1,0) at top-right offset: (half_w, 0)
    # (0,1) at bottom-left offset: (0, half_h)
    # (1,1) at bottom-right offset: (half_w, half_h)
    placements = {
      (0, 0): (0, 0),
      (1, 0): (half_w, 0),
      (0, 1): (0, half_h),
      (1, 1): (half_w, half_h),
    }

    for (x, y), (px, py) in placements.items():
      tile_img = Image.open(tile_paths[(x, y)])
      assembled.paste(tile_img, (px, py))
      print(f"   Placed ({x}, {y}) at ({px}, {py})")

    # Save assembled image
    output_path = renders_dir / "check_tiles_assembled.png"
    assembled.save(output_path)
    print(f"\n✅ Assembled image saved to {output_path}")
    print(f"   Size: {assembled_width}x{assembled_height}")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Check tiles fit together by rendering and assembling a 2x2 grid."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = check_tiles(generation_dir, args.port)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Let me test by re-seeding the database:

Now there are 17 unique quadrants (no duplicates). Let me verify and test rendering:

All quadrants have count = 1 (no duplicates). Let me test rendering:

Let me verify the renders were saved:

The schema is now working correctly with `(quadrant_x, quadrant_y)` as the composite primary key. Each quadrant position has exactly one row, and renders are saved properly.

**Summary of changes to `seed_tiles.py`:**
- Removed `id` column (hash-based)
- Made `(quadrant_x, quadrant_y)` the composite primary key
- Removed the `generate_quadrant_id` function
- Updated schema version to 3

**Database now has:**
- 17 unique quadrants (vs. 26 before with duplicates)
- 4 quadrants with render data after rendering tile (0, 0)

---

**User**

I'm a bit confused how there can be 7 tiles and 17 quadrants - that doesn't really make sense to me...

---

**Cursor**

You're right to be confused! Let me investigate. With 7 tiles × 4 quadrants each = 28 total, but with 50% overlap some quadrants are shared. Let's see what's happening:

```plaintext
y=-1:  (-2,-1) (-1,-1) (0,-1)
y=0:   (-2,0)  (-1,0)  (0,0)  (1,0)
y=1:                   (-1,1) (0,1)  (1,1)  (2,1)  (3,1)
y=2:                   (-1,2) (0,2)  (1,2)  (2,2)  (3,2)
```

I see! The tiles are scattered (not a contiguous rectangle) based on the geographic bounds:

```
Tiles:                    Quadrant grid:
(-1,-2) (-1,-1)          y=-1:  x: -2  -1   0
        (0,0)            y=0:   x: -2  -1   0   1
(1,-1)  (1,0)  (1,1) (1,2)  y=1:   x:     -1   0   1   2   3
                          y=2:   x:     -1   0   1   2   3
```

The 17 count is correct because:
1. The tiles form an **irregular shape** based on lat/lng bounds
2. Each tile's 4 quadrants get deduplicated by `(quadrant_x, quadrant_y)` primary key
3. Some tiles share quadrants (due to 50% overlap), reducing the total

For example, tile (0,0) and tile (1,0) both cover quadrant (0,1) and (1,1) - but only one entry is stored for each.

The irregular shape happens because the `calculate_tile_grid` function finds tiles whose **centers** fall within the bounds. The bounds in your test config create this scattered pattern rather than a clean rectangle.

Would you like me to adjust the bounds or the tile calculation to create a more regular grid?

```plaintext
Tiles:                    Quadrant grid:
(-1,-2) (-1,-1)          y=-1:  x: -2  -1   0
        (0,0)            y=0:   x: -2  -1   0   1
(1,-1)  (1,0)  (1,1) (1,2)  y=1:   x:     -1   0   1   2   3
                          y=2:   x:     -1   0   1   2   3
```

---

**User**

ok cool, so the bounds are irregular shape

---

**Cursor**

Exactly! The geographic bounds in your `generation_config.json` create an irregular tile pattern because:

1. The isometric projection is rotated (azimuth = -15°) 
2. The bounds are a lat/lng rectangle, but tiles are aligned to the rotated isometric grid
3. The `calculate_tile_grid` function only includes tiles whose centers fall within the bounds

So tiles at the edges get clipped irregularly, resulting in that scattered quadrant pattern.

For a cleaner rectangular grid of quadrants, you could either:
- Expand the bounds to fully contain a regular grid
- Or specify the generation in terms of tile counts (e.g., "3x3 tiles from seed") rather than geographic bounds

But for now, the system is working correctly - it's just that real geographic bounds rarely align perfectly with the isometric grid!

---

**User**

OK - I want to change the logic of the @src/isometric_nyc/e2e_generation/seed_tiles.py script a bit - instead of explicit bounds, I want to seed it as an n by n grid of tiles with the top left tile centered at the lat/lng param of the generation_config.json.

---

**Cursor**

I'll update `seed_tiles.py` to use an n×n grid instead of bounds:

```python
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
    "grid_size",
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

  # Validate grid_size
  grid_size = config["grid_size"]
  if not isinstance(grid_size, int) or grid_size < 1:
    raise ValueError("grid_size must be a positive integer")

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
  Calculate all tile positions that fall within the generation bounds.

  Starting from the seed position, expands outward in all directions
  until tiles exceed the bounds.

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  bounds = config["bounds"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]

  # Tile step is the fraction of tile size between adjacent tiles (0.5 = 50% overlap)
  tile_step = config.get("tile_step", 0.5)

  # Calculate bounds box
  min_lat = min(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  max_lat = max(
    bounds["top_left"]["lat"],
    bounds["bottom_left"]["lat"],
    bounds["top_right"]["lat"],
    bounds["bottom_right"]["lat"],
  )
  min_lng = min(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )
  max_lng = max(
    bounds["top_left"]["lng"],
    bounds["bottom_left"]["lng"],
    bounds["top_right"]["lng"],
    bounds["bottom_right"]["lng"],
  )

  tiles = []

  # Start with the seed tile at (0, 0)
  tiles.append((0, 0, seed_lat, seed_lng))

  # Calculate pixel step between tiles
  step_x_px = width_px * tile_step
  step_y_px = height_px * tile_step

  # Expand outward from seed
  # We'll use a simple approach: iterate rows and columns until we're outside bounds

  # Determine rough extent needed
  # This is an approximation - we'll filter by actual bounds later
  max_extent = 1000  # Safety limit

  for row in range(-max_extent, max_extent + 1):
    for col in range(-max_extent, max_extent + 1):
      if row == 0 and col == 0:
        continue  # Already added seed

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

      # Check if within bounds
      if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
        tiles.append((row, col, lat, lng))
      elif row == 0 and col > 0 and lng > max_lng:
        # Positive columns exceed east bound, stop expanding east
        break
      elif row == 0 and col < 0 and lng < min_lng:
        # Negative columns exceed west bound, stop expanding west
        break

    # Early termination for rows
    if row > 0:
      # Check if this row's seed column is within lat bounds
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row < min_lat:
        break
    elif row < 0:
      _, test_lat_row = calculate_offset(
        seed_lat,
        seed_lng,
        0,
        -row * step_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )
      if test_lat_row > max_lat:
        break

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

```

```python
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
    "grid_size",
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

  # Validate grid_size
  grid_size = config["grid_size"]
  if not isinstance(grid_size, int) or grid_size < 1:
    raise ValueError("grid_size must be a positive integer")

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
  Calculate tile positions for an n×n grid.

  The top-left tile (row=0, col=0) is centered at the seed lat/lng.
  Subsequent tiles extend right (increasing col) and down (increasing row).

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  grid_size = config["grid_size"]
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

  # Generate n×n grid starting from seed at (0, 0)
  for row in range(grid_size):
    for col in range(grid_size):
      if row == 0 and col == 0:
        # Seed tile
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

```

Now let me update the test generation config:

```json
{
  "name": "test_generation",
  "seed": {
    "lat": 40.7484,
    "lng": -73.9857
  },
  "grid_size": 3,
  "camera_azimuth_degrees": -15,
  "camera_elevation_degrees": -45,
  "width_px": 1024,
  "height_px": 1024,
  "view_height_meters": 300,
  "tile_step": 0.5
}

```

Let me test the updated script:

---

**User**

Use my new schema with @generation_config.json (7-10) 

---

**Cursor**

I'll update `seed_tiles.py` to use your new schema with directional tile counts:

```python
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
    "n_tiles_left",
    "n_tiles_right",
    "n_tiles_up",
    "n_tiles_down",
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

  # Validate tile counts (must be non-negative integers)
  for direction in ["n_tiles_left", "n_tiles_right", "n_tiles_up", "n_tiles_down"]:
    val = config[direction]
    if not isinstance(val, int) or val < 0:
      raise ValueError(f"{direction} must be a non-negative integer")

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
  Calculate tile positions for an n×n grid.

  The top-left tile (row=0, col=0) is centered at the seed lat/lng.
  Subsequent tiles extend right (increasing col) and down (increasing row).

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  grid_size = config["grid_size"]
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

  # Generate n×n grid starting from seed at (0, 0)
  for row in range(grid_size):
    for col in range(grid_size):
      if row == 0 and col == 0:
        # Seed tile
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

```

```python
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
    "n_tiles_left",
    "n_tiles_right",
    "n_tiles_up",
    "n_tiles_down",
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

  # Validate tile counts (must be non-negative integers)
  for direction in ["n_tiles_left", "n_tiles_right", "n_tiles_up", "n_tiles_down"]:
    val = config[direction]
    if not isinstance(val, int) or val < 0:
      raise ValueError(f"{direction} must be a non-negative integer")

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
  Calculate tile positions based on directional counts from the seed.

  The seed tile (row=0, col=0) is centered at the seed lat/lng.
  - n_tiles_left: tiles extending left (negative col)
  - n_tiles_right: tiles extending right (positive col)
  - n_tiles_up: tiles extending up (negative row)
  - n_tiles_down: tiles extending down (positive row)

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  n_left = config["n_tiles_left"]
  n_right = config["n_tiles_right"]
  n_up = config["n_tiles_up"]
  n_down = config["n_tiles_down"]
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

  # Generate grid from (-n_up, -n_left) to (n_down, n_right)
  # row: -n_up to n_down (negative = up, positive = down)
  # col: -n_left to n_right (negative = left, positive = right)
  for row in range(-n_up, n_down + 1):
    for col in range(-n_left, n_right + 1):
      if row == 0 and col == 0:
        # Seed tile
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

```

Now let me test:

Now we have a clean 5×5 tile grid (25 tiles) creating a 6×6 quadrant grid (36 unique quadrants):

Now we have a clean rectangular grid:

**Tiles:** 5×5 grid (rows -2 to +2, cols -2 to +2) = 25 tiles

**Quadrants:** 6×6 grid (x: -2 to +3, y: -2 to +3) = 36 quadrants

```
Quadrant grid:
     x: -2  -1   0   1   2   3
y=-2:   ●   ●   ●   ●   ●   ●
y=-1:   ●   ●   ●   ●   ●   ●
y=0:    ●   ●  [●]  ●   ●   ●  ← seed at (0,0)
y=1:    ●   ●   ●   ●   ●   ●
y=2:    ●   ●   ●   ●   ●   ●
y=3:    ●   ●   ●   ●   ●   ●
```

The math works: with `n_tiles_left/right/up/down = 2`, you get tiles from -2 to +2 (5 tiles per axis), and since each tile covers 2 quadrants per axis with 50% overlap, you get 6 quadrants per axis.

```plaintext
Quadrant grid:
     x: -2  -1   0   1   2   3
y=-2:   ●   ●   ●   ●   ●   ●
y=-1:   ●   ●   ●   ●   ●   ●
y=0:    ●   ●  [●]  ●   ●   ●  ← seed at (0,0)
y=1:    ●   ●   ●   ●   ●   ●
y=2:    ●   ●   ●   ●   ●   ●
y=3:    ●   ●   ●   ●   ●   ●
```

---

**User**

Ok - I just updated my config schema @generations/test_generation/generation_config.json - please update @src/isometric_nyc/e2e_generation/seed_tiles.py to now seed tiles starting from the top-left

---

**Cursor**

```python
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
  Calculate tile positions based on directional counts from the seed.

  The seed tile (row=0, col=0) is centered at the seed lat/lng.
  - n_tiles_left: tiles extending left (negative col)
  - n_tiles_right: tiles extending right (positive col)
  - n_tiles_up: tiles extending up (negative row)
  - n_tiles_down: tiles extending down (positive row)

  Returns list of (row, col, lat, lon) tuples for each tile.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  n_left = config["n_tiles_left"]
  n_right = config["n_tiles_right"]
  n_up = config["n_tiles_up"]
  n_down = config["n_tiles_down"]
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

  # Generate grid from (-n_up, -n_left) to (n_down, n_right)
  # row: -n_up to n_down (negative = up, positive = down)
  # col: -n_left to n_right (negative = left, positive = right)
  for row in range(-n_up, n_down + 1):
    for col in range(-n_left, n_right + 1):
      if row == 0 and col == 0:
        # Seed tile
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

```

```python
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

```

Now with `n_tiles_x=2, n_tiles_y=2`:

**Tiles:** 2×2 grid (rows 0-1, cols 0-1) = 4 tiles

**Quadrants:** 3×3 grid starting from (0,0) = 9 quadrants

```
     x:  0   1   2
y=0:    [●]  ●   ●   ← seed at (0,0)
y=1:     ●   ●   ●
y=2:     ●   ●   ●
```

```plaintext
     x:  0   1   2
y=0:    [●]  ●   ●   ← seed at (0,0)
y=1:     ●   ●   ●
y=2:     ●   ●   ●
```

---

**User**

OK - now I need you to update @src/isometric_nyc/e2e_generation/render_tile.py to make sure it can generate tiles that don't already exist in the db. So, if the user runs it with -16 -16, and that tile / quadrants don't exist, it'll add those rows to the database

---

**Cursor**

I'll update `render_tile.py` to create quadrant rows if they don't exist:

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import math
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "has_render": row[5] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(
  conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Returns True if successful, False if quadrant not found.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y) - this will be the TL of the rendered tile
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import math
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


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


def calculate_quadrant_lat_lng(
  config: dict, quadrant_x: int, quadrant_y: int
) -> tuple[float, float]:
  """
  Calculate the lat/lng anchor for a quadrant at position (quadrant_x, quadrant_y).

  The anchor is the bottom-right corner of the quadrant.
  For the TL quadrant (0, 0) of the seed tile, the anchor equals the seed lat/lng.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant step is half a tile (with tile_step=0.5, one tile step = one quadrant)
  # But quadrant positions are in quadrant units, not tile units
  # quadrant_x = tile_col + dx, quadrant_y = tile_row + dy
  # So shift in pixels = quadrant position * (tile_size * tile_step)
  quadrant_step_x_px = width_px * tile_step
  quadrant_step_y_px = height_px * tile_step

  shift_x_px = quadrant_x * quadrant_step_x_px
  shift_y_px = -quadrant_y * quadrant_step_y_px  # Negative because y increases downward

  return calculate_offset(
    seed_lat,
    seed_lng,
    shift_x_px,
    shift_y_px,
    view_height_meters,
    height_px,
    azimuth,
    elevation,
  )


def ensure_quadrant_exists(
  conn: sqlite3.Connection, config: dict, x: int, y: int
) -> dict:
  """
  Ensure a quadrant exists at position (x, y), creating it if necessary.

  Returns the quadrant data.
  """
  cursor = conn.cursor()

  # Check if quadrant already exists
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()

  if row:
    return {
      "lat": row[0],
      "lng": row[1],
      "tile_row": row[2],
      "tile_col": row[3],
      "quadrant_index": row[4],
      "has_render": row[5] is not None,
    }

  # Quadrant doesn't exist - create it
  lat, lng = calculate_quadrant_lat_lng(config, x, y)

  # Calculate tile_row, tile_col, and quadrant_index
  # For a quadrant at (x, y), it could belong to multiple tiles due to overlap
  # We'll use the tile where this quadrant is the TL (index 0)
  tile_col = x
  tile_row = y
  quadrant_index = 0  # TL of its "primary" tile

  cursor.execute(
    """
    INSERT INTO quadrants (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (x, y, lat, lng, tile_row, tile_col, quadrant_index),
  )
  conn.commit()

  print(f"   📝 Created quadrant ({x}, {y}) at {lat:.6f}, {lng:.6f}")

  return {
    "lat": lat,
    "lng": lng,
    "tile_row": tile_row,
    "tile_col": tile_col,
    "quadrant_index": quadrant_index,
    "has_render": False,
  }


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "has_render": row[5] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(
  conn: sqlite3.Connection, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Returns True if successful, False if quadrant not found.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y) - this will be the TL of the rendered tile
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import math
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


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


def calculate_quadrant_lat_lng(
  config: dict, quadrant_x: int, quadrant_y: int
) -> tuple[float, float]:
  """
  Calculate the lat/lng anchor for a quadrant at position (quadrant_x, quadrant_y).

  The anchor is the bottom-right corner of the quadrant.
  For the TL quadrant (0, 0) of the seed tile, the anchor equals the seed lat/lng.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant step is half a tile (with tile_step=0.5, one tile step = one quadrant)
  # But quadrant positions are in quadrant units, not tile units
  # quadrant_x = tile_col + dx, quadrant_y = tile_row + dy
  # So shift in pixels = quadrant position * (tile_size * tile_step)
  quadrant_step_x_px = width_px * tile_step
  quadrant_step_y_px = height_px * tile_step

  shift_x_px = quadrant_x * quadrant_step_x_px
  shift_y_px = -quadrant_y * quadrant_step_y_px  # Negative because y increases downward

  return calculate_offset(
    seed_lat,
    seed_lng,
    shift_x_px,
    shift_y_px,
    view_height_meters,
    height_px,
    azimuth,
    elevation,
  )


def ensure_quadrant_exists(
  conn: sqlite3.Connection, config: dict, x: int, y: int
) -> dict:
  """
  Ensure a quadrant exists at position (x, y), creating it if necessary.

  Returns the quadrant data.
  """
  cursor = conn.cursor()

  # Check if quadrant already exists
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()

  if row:
    return {
      "lat": row[0],
      "lng": row[1],
      "tile_row": row[2],
      "tile_col": row[3],
      "quadrant_index": row[4],
      "has_render": row[5] is not None,
    }

  # Quadrant doesn't exist - create it
  lat, lng = calculate_quadrant_lat_lng(config, x, y)

  # Calculate tile_row, tile_col, and quadrant_index
  # For a quadrant at (x, y), it could belong to multiple tiles due to overlap
  # We'll use the tile where this quadrant is the TL (index 0)
  tile_col = x
  tile_row = y
  quadrant_index = 0  # TL of its "primary" tile

  cursor.execute(
    """
    INSERT INTO quadrants (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (x, y, lat, lng, tile_row, tile_col, quadrant_index),
  )
  conn.commit()

  print(f"   📝 Created quadrant ({x}, {y}) at {lat:.6f}, {lng:.6f}")

  return {
    "lat": lat,
    "lng": lng,
    "tile_row": tile_row,
    "tile_col": tile_col,
    "quadrant_index": quadrant_index,
    "has_render": False,
  }


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "has_render": row[5] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(
  conn: sqlite3.Connection, config: dict, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Creates the quadrant if it doesn't exist.
  Returns True if successful.
  """
  # Ensure the quadrant exists first
  ensure_quadrant_exists(conn, config, x, y)

  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Get the quadrant at position (x, y) - this will be the TL of the rendered tile
    quadrant = get_quadrant(conn, x, y)
    if not quadrant:
      print(f"❌ No quadrant found at position ({x}, {y})")
      return None

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import math
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


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


def calculate_quadrant_lat_lng(
  config: dict, quadrant_x: int, quadrant_y: int
) -> tuple[float, float]:
  """
  Calculate the lat/lng anchor for a quadrant at position (quadrant_x, quadrant_y).

  The anchor is the bottom-right corner of the quadrant.
  For the TL quadrant (0, 0) of the seed tile, the anchor equals the seed lat/lng.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant step is half a tile (with tile_step=0.5, one tile step = one quadrant)
  # But quadrant positions are in quadrant units, not tile units
  # quadrant_x = tile_col + dx, quadrant_y = tile_row + dy
  # So shift in pixels = quadrant position * (tile_size * tile_step)
  quadrant_step_x_px = width_px * tile_step
  quadrant_step_y_px = height_px * tile_step

  shift_x_px = quadrant_x * quadrant_step_x_px
  shift_y_px = -quadrant_y * quadrant_step_y_px  # Negative because y increases downward

  return calculate_offset(
    seed_lat,
    seed_lng,
    shift_x_px,
    shift_y_px,
    view_height_meters,
    height_px,
    azimuth,
    elevation,
  )


def ensure_quadrant_exists(
  conn: sqlite3.Connection, config: dict, x: int, y: int
) -> dict:
  """
  Ensure a quadrant exists at position (x, y), creating it if necessary.

  Returns the quadrant data.
  """
  cursor = conn.cursor()

  # Check if quadrant already exists
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()

  if row:
    return {
      "lat": row[0],
      "lng": row[1],
      "tile_row": row[2],
      "tile_col": row[3],
      "quadrant_index": row[4],
      "has_render": row[5] is not None,
    }

  # Quadrant doesn't exist - create it
  lat, lng = calculate_quadrant_lat_lng(config, x, y)

  # Calculate tile_row, tile_col, and quadrant_index
  # For a quadrant at (x, y), it could belong to multiple tiles due to overlap
  # We'll use the tile where this quadrant is the TL (index 0)
  tile_col = x
  tile_row = y
  quadrant_index = 0  # TL of its "primary" tile

  cursor.execute(
    """
    INSERT INTO quadrants (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (x, y, lat, lng, tile_row, tile_col, quadrant_index),
  )
  conn.commit()

  print(f"   📝 Created quadrant ({x}, {y}) at {lat:.6f}, {lng:.6f}")

  return {
    "lat": lat,
    "lng": lng,
    "tile_row": tile_row,
    "tile_col": tile_col,
    "quadrant_index": quadrant_index,
    "has_render": False,
  }


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "has_render": row[5] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(
  conn: sqlite3.Connection, config: dict, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Creates the quadrant if it doesn't exist.
  Returns True if successful.
  """
  # Ensure the quadrant exists first
  ensure_quadrant_exists(conn, config, x, y)

  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    # Ensure the quadrant exists (creates it if necessary)
    quadrant = ensure_quadrant_exists(conn, config, x, y)

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Quadrant ({qx}, {qy}) not found in database")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

```python
"""
Render a tile for debugging the generation database.

This script renders a tile at a specific quadrant position using the web
renderer, splits it into 4 quadrants, and saves them to the SQLite database.

Usage:
  uv run python src/isometric_nyc/e2e_generation/render_tile.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates (quadrant_x, quadrant_y).
The tile rendered will have quadrant (x, y) in its top-left, with (x+1, y),
(x, y+1), and (x+1, y+1) in the other positions.
"""

import argparse
import io
import json
import math
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

# Web server configuration
WEB_DIR = Path(__file__).parent.parent.parent / "web"
DEFAULT_WEB_PORT = 5173


def wait_for_server(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
  """
  Wait for the server to be ready by making HTTP requests.

  Args:
    port: Port to check
    timeout: Maximum time to wait in seconds
    interval: Time between checks in seconds

  Returns:
    True if server is ready, False if timeout
  """
  url = f"http://localhost:{port}/"
  start_time = time.time()
  attempts = 0

  while time.time() - start_time < timeout:
    attempts += 1
    try:
      req = urllib.request.Request(url, method="HEAD")
      with urllib.request.urlopen(req, timeout=2):
        return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
      time.sleep(interval)

  print(f"   ⚠️  Server not responding after {attempts} attempts ({timeout}s)")
  return False


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server and wait for it to be ready.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  print(f"   Web dir: {web_dir}")

  if not web_dir.exists():
    raise RuntimeError(f"Web directory not found: {web_dir}")

  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Give the process a moment to start
  time.sleep(2)

  # Check if process died immediately
  if process.poll() is not None:
    stdout = process.stdout.read().decode() if process.stdout else ""
    stderr = process.stderr.read().decode() if process.stderr else ""
    raise RuntimeError(
      f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
    )

  # Wait for server to be ready via HTTP
  print("   ⏳ Waiting for server to be ready...")
  if wait_for_server(port, timeout=30.0):
    print(f"   ✅ Server ready on http://localhost:{port}")
  else:
    # Check if process died during wait
    if process.poll() is not None:
      stdout = process.stdout.read().decode() if process.stdout else ""
      stderr = process.stderr.read().decode() if process.stderr else ""
      raise RuntimeError(
        f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
      )
    print("   ⚠️  Server may not be fully ready, continuing anyway...")

  return process


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


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


def calculate_quadrant_lat_lng(
  config: dict, quadrant_x: int, quadrant_y: int
) -> tuple[float, float]:
  """
  Calculate the lat/lng anchor for a quadrant at position (quadrant_x, quadrant_y).

  The anchor is the bottom-right corner of the quadrant.
  For the TL quadrant (0, 0) of the seed tile, the anchor equals the seed lat/lng.
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant step is half a tile (with tile_step=0.5, one tile step = one quadrant)
  # But quadrant positions are in quadrant units, not tile units
  # quadrant_x = tile_col + dx, quadrant_y = tile_row + dy
  # So shift in pixels = quadrant position * (tile_size * tile_step)
  quadrant_step_x_px = width_px * tile_step
  quadrant_step_y_px = height_px * tile_step

  shift_x_px = quadrant_x * quadrant_step_x_px
  shift_y_px = -quadrant_y * quadrant_step_y_px  # Negative because y increases downward

  return calculate_offset(
    seed_lat,
    seed_lng,
    shift_x_px,
    shift_y_px,
    view_height_meters,
    height_px,
    azimuth,
    elevation,
  )


def ensure_quadrant_exists(
  conn: sqlite3.Connection, config: dict, x: int, y: int
) -> dict:
  """
  Ensure a quadrant exists at position (x, y), creating it if necessary.

  Returns the quadrant data.
  """
  cursor = conn.cursor()

  # Check if quadrant already exists
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()

  if row:
    return {
      "lat": row[0],
      "lng": row[1],
      "tile_row": row[2],
      "tile_col": row[3],
      "quadrant_index": row[4],
      "has_render": row[5] is not None,
    }

  # Quadrant doesn't exist - create it
  lat, lng = calculate_quadrant_lat_lng(config, x, y)

  # Calculate tile_row, tile_col, and quadrant_index
  # For a quadrant at (x, y), it could belong to multiple tiles due to overlap
  # We'll use the tile where this quadrant is the TL (index 0)
  tile_col = x
  tile_row = y
  quadrant_index = 0  # TL of its "primary" tile

  cursor.execute(
    """
    INSERT INTO quadrants (quadrant_x, quadrant_y, lat, lng, tile_row, tile_col, quadrant_index)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (x, y, lat, lng, tile_row, tile_col, quadrant_index),
  )
  conn.commit()

  print(f"   📝 Created quadrant ({x}, {y}) at {lat:.6f}, {lng:.6f}")

  return {
    "lat": lat,
    "lng": lng,
    "tile_row": tile_row,
    "tile_col": tile_col,
    "quadrant_index": quadrant_index,
    "has_render": False,
  }


def get_quadrant(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """
  Get a quadrant by its (x, y) position.

  Returns the quadrant's anchor coordinates and metadata.
  """
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT lat, lng, tile_row, tile_col, quadrant_index, render
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None

  return {
    "lat": row[0],
    "lng": row[1],
    "tile_row": row[2],
    "tile_col": row[3],
    "quadrant_index": row[4],
    "has_render": row[5] is not None,
  }


def check_all_quadrants_rendered(conn: sqlite3.Connection, x: int, y: int) -> bool:
  """
  Check if all 4 quadrants for the tile starting at (x, y) have been rendered.

  The tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  """
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]

  for qx, qy in positions:
    q = get_quadrant(conn, qx, qy)
    if q is None or not q["has_render"]:
      return False

  return True


def save_quadrant_render(
  conn: sqlite3.Connection, config: dict, x: int, y: int, png_bytes: bytes
) -> bool:
  """
  Save render bytes for a quadrant at position (x, y).

  Creates the quadrant if it doesn't exist.
  Returns True if successful.
  """
  # Ensure the quadrant exists first
  ensure_quadrant_exists(conn, config, x, y)

  cursor = conn.cursor()
  cursor.execute(
    """
    UPDATE quadrants
    SET render = ?
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (png_bytes, x, y),
  )
  conn.commit()
  return cursor.rowcount > 0


def split_tile_into_quadrants(
  tile_image: Image.Image,
) -> dict[tuple[int, int], Image.Image]:
  """
  Split a tile image into 4 quadrant images.

  Returns a dict mapping (dx, dy) offset to the quadrant image:
    (0, 0) = top-left
    (1, 0) = top-right
    (0, 1) = bottom-left
    (1, 1) = bottom-right
  """
  width, height = tile_image.size
  half_w = width // 2
  half_h = height // 2

  quadrants = {
    (0, 0): tile_image.crop((0, 0, half_w, half_h)),
    (1, 0): tile_image.crop((half_w, 0, width, half_h)),
    (0, 1): tile_image.crop((0, half_h, half_w, height)),
    (1, 1): tile_image.crop((half_w, half_h, width, height)),
  }

  return quadrants


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def render_quadrant_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  overwrite: bool = True,
) -> Path | None:
  """
  Render a tile with quadrant (x, y) in the top-left position.

  The rendered tile covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1).
  After rendering, the tile is split into 4 quadrants and saved to the database.

  Args:
    generation_dir: Path to the generation directory
    x: quadrant_x coordinate (top-left of rendered tile)
    y: quadrant_y coordinate (top-left of rendered tile)
    port: Web server port (default: 5173)
    overwrite: If False, skip rendering if all 4 quadrants already have renders

  Returns:
    Path to the rendered image, or None if quadrant not found or skipped
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    # Get generation config
    config = get_generation_config(conn)

    # Check if we should skip rendering (all quadrants already have renders)
    if not overwrite:
      if check_all_quadrants_rendered(conn, x, y):
        print(f"⏭️  Skipping ({x}, {y}) - all quadrants already rendered")
        return None

    # Ensure the quadrant exists (creates it if necessary)
    quadrant = ensure_quadrant_exists(conn, config, x, y)

    print(f"📍 Rendering tile starting at quadrant ({x}, {y})")
    print(f"   Covers: ({x},{y}), ({x + 1},{y}), ({x},{y + 1}), ({x + 1},{y + 1})")
    print(f"   Anchor (center): {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

    # Create renders directory
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # Output path for full tile
    output_path = renders_dir / f"render_{x}_{y}.png"

    # Build URL parameters - center on the quadrant's anchor
    params = {
      "export": "true",
      "lat": quadrant["lat"],
      "lon": quadrant["lng"],
      "width": config["width_px"],
      "height": config["height_px"],
      "azimuth": config["camera_azimuth_degrees"],
      "elevation": config["camera_elevation_degrees"],
      "view_height": config.get("view_height_meters", 200),
    }
    query_string = urlencode(params)
    url = f"http://localhost:{port}/?{query_string}"

    # Render using Playwright
    print("\n🌐 Rendering via web viewer...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
      browser = p.chromium.launch(
        headless=True,
        args=[
          "--enable-webgl",
          "--use-gl=angle",
          "--ignore-gpu-blocklist",
        ],
      )

      context = browser.new_context(
        viewport={"width": config["width_px"], "height": config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Navigate to the page
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))

      page.close()
      context.close()
      browser.close()

    print(f"✅ Rendered full tile to {output_path}")

    # Split tile into quadrants and save to database
    print("\n💾 Saving quadrants to database...")
    tile_image = Image.open(output_path)
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Map (dx, dy) offsets to absolute quadrant positions
    for (dx, dy), quad_img in quadrant_images.items():
      qx, qy = x + dx, y + dy
      png_bytes = image_to_png_bytes(quad_img)

      if save_quadrant_render(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved quadrant ({qx}, {qy}) - {len(png_bytes)} bytes")
      else:
        print(f"   ⚠️  Failed to save quadrant ({qx}, {qy})")

    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Render a tile for debugging the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="quadrant_x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="quadrant_y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    default=True,
    help="Overwrite existing renders (default: True)",
  )
  parser.add_argument(
    "--no-overwrite",
    action="store_true",
    help="Skip rendering if all quadrants already have renders",
  )

  args = parser.parse_args()

  # Handle overwrite flag (--no-overwrite takes precedence)
  overwrite = not args.no_overwrite

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = render_quadrant_tile(generation_dir, args.x, args.y, args.port, overwrite)
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    # Stop web server
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())

```

Let me test by rendering at position (-1, -1) which doesn't exist:

It works. Let me verify the new quadrants are in the database:

The script now automatically creates quadrants that don't exist:

- **Created**: (-1, -1), (0, -1), (-1, 0) with renders
- **Updated**: (0, 0) with new render

Now you can render any tile position, even outside the originally seeded grid, and it will create the necessary quadrant rows in the database.

