"""
Initialize a new layer directory for parallel map generation.

This script creates a layer directory with:
- layer_config.json: Configuration for the generation
- progress.db: SQLite database with the generation plan

The generation plan uses a 3-step tiling algorithm:
1. Generate 2x2 tiles with 1-quadrant gaps between them
2. Fill the 1x2 and 2x1 gaps between 2x2 tiles
3. Fill the remaining 1x1 corner quadrants

Usage:
  uv run python src/isometric_nyc/generation/init_layer.py \
    --name snow_layer \
    --generation-dir generations/nyc \
    --source-layer generations \
    --output layers/snow

  # With explicit bounds:
  uv run python src/isometric_nyc/generation/init_layer.py \
    --name snow_layer \
    --generation-dir generations/nyc \
    --source-layer generations \
    --bounds "-10,-10,50,50" \
    --output layers/snow
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Bounds:
  """Bounding box for the generation area."""

  min_x: int
  max_x: int
  min_y: int
  max_y: int

  @classmethod
  def from_string(cls, s: str) -> Bounds:
    """Parse bounds from string format: 'min_x,min_y,max_x,max_y'."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 4:
      raise ValueError(f"Invalid bounds format: {s}. Expected: min_x,min_y,max_x,max_y")
    return cls(
      min_x=int(parts[0]),
      min_y=int(parts[1]),
      max_x=int(parts[2]),
      max_y=int(parts[3]),
    )

  def to_dict(self) -> dict:
    return {
      "min_x": self.min_x,
      "max_x": self.max_x,
      "min_y": self.min_y,
      "max_y": self.max_y,
    }


@dataclass
class GenerationItem:
  """A single item in the generation plan."""

  step: int
  block_type: str  # '2x2', '1x2', '2x1', '1x1'
  top_left_x: int
  top_left_y: int
  width: int
  height: int


def get_bounds_from_generation_dir(generation_dir: Path, source_layer: str) -> Bounds:
  """
  Get the bounds of existing quadrants from the generation directory.

  Args:
      generation_dir: Path to the generation directory
      source_layer: Either 'generations' or 'renders'

  Returns:
      Bounds of existing quadrants
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()

    # Determine which column to check based on source_layer
    if source_layer == "generations":
      column = "generation"
    elif source_layer == "renders":
      column = "render"
    else:
      raise ValueError(
        f"Invalid source_layer: {source_layer}. Must be 'generations' or 'renders'"
      )

    # Get bounds of quadrants that have the source data
    cursor.execute(f"""
            SELECT MIN(quadrant_x), MAX(quadrant_x), MIN(quadrant_y), MAX(quadrant_y)
            FROM quadrants
            WHERE {column} IS NOT NULL
        """)
    row = cursor.fetchone()

    if row[0] is None:
      raise ValueError(
        f"No quadrants found with {source_layer} data in {generation_dir}"
      )

    return Bounds(
      min_x=row[0],
      max_x=row[1],
      min_y=row[2],
      max_y=row[3],
    )
  finally:
    conn.close()


def get_existing_quadrants(
  generation_dir: Path, source_layer: str
) -> set[tuple[int, int]]:
  """
  Get the set of quadrant coordinates that have source data.

  Args:
      generation_dir: Path to the generation directory
      source_layer: Either 'generations' or 'renders'

  Returns:
      Set of (x, y) tuples for quadrants with source data
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()

    if source_layer == "generations":
      column = "generation"
    elif source_layer == "renders":
      column = "render"
    else:
      raise ValueError(f"Invalid source_layer: {source_layer}")

    cursor.execute(f"""
            SELECT quadrant_x, quadrant_y
            FROM quadrants
            WHERE {column} IS NOT NULL
        """)

    return {(row[0], row[1]) for row in cursor.fetchall()}
  finally:
    conn.close()


def generate_tiling_plan(
  bounds: Bounds,
  existing_quadrants: set[tuple[int, int]] | None = None,
) -> list[GenerationItem]:
  """
  Generate the 3-step tiling plan for the given bounds.

  Step 1: 2x2 tiles with 1-quadrant gaps (positions: 0, 3, 6, 9, ...)
  Step 2: 1x2 and 2x1 strips filling the gaps
  Step 3: 1x1 corner quadrants

  Args:
      bounds: The bounds of the area to generate
      existing_quadrants: Optional set of (x, y) tuples for quadrants that exist.
                         If provided, only items where ALL quadrants exist will be included.

  Returns:
      List of GenerationItems in order (all step 1, then step 2, then step 3)
  """
  items: list[GenerationItem] = []

  def all_quadrants_exist(x: int, y: int, width: int, height: int) -> bool:
    """Check if all quadrants in a block exist in the source."""
    if existing_quadrants is None:
      return True
    for dy in range(height):
      for dx in range(width):
        if (x + dx, y + dy) not in existing_quadrants:
          return False
    return True

  # Step 1: 2x2 tiles with gaps
  # Start positions are at min_x, min_x+3, min_x+6, etc.
  # Each 2x2 covers (x, y), (x+1, y), (x, y+1), (x+1, y+1)
  step1_x_positions = list(range(bounds.min_x, bounds.max_x, 3))
  step1_y_positions = list(range(bounds.min_y, bounds.max_y, 3))

  for y in step1_y_positions:
    for x in step1_x_positions:
      # Check if the 2x2 block fits within bounds
      if x + 1 <= bounds.max_x and y + 1 <= bounds.max_y:
        if all_quadrants_exist(x, y, 2, 2):
          items.append(
            GenerationItem(
              step=1,
              block_type="2x2",
              top_left_x=x,
              top_left_y=y,
              width=2,
              height=2,
            )
          )

  # Step 2: Fill gaps between 2x2 tiles
  # 2a: Vertical 1x2 strips (filling horizontal gaps)
  # These are at x = min_x+2, min_x+5, min_x+8, ...
  for x in range(bounds.min_x + 2, bounds.max_x + 1, 3):
    for y in step1_y_positions:
      # Check if the 1x2 block fits within bounds
      if x <= bounds.max_x and y + 1 <= bounds.max_y:
        if all_quadrants_exist(x, y, 1, 2):
          items.append(
            GenerationItem(
              step=2,
              block_type="1x2",
              top_left_x=x,
              top_left_y=y,
              width=1,
              height=2,
            )
          )

  # 2b: Horizontal 2x1 strips (filling vertical gaps)
  # These are at y = min_y+2, min_y+5, min_y+8, ...
  for y in range(bounds.min_y + 2, bounds.max_y + 1, 3):
    for x in step1_x_positions:
      # Check if the 2x1 block fits within bounds
      if x + 1 <= bounds.max_x and y <= bounds.max_y:
        if all_quadrants_exist(x, y, 2, 1):
          items.append(
            GenerationItem(
              step=2,
              block_type="2x1",
              top_left_x=x,
              top_left_y=y,
              width=2,
              height=1,
            )
          )

  # Step 3: 1x1 corner quadrants
  # These are at positions where x = min_x+2, min_x+5, ... AND y = min_y+2, min_y+5, ...
  for y in range(bounds.min_y + 2, bounds.max_y + 1, 3):
    for x in range(bounds.min_x + 2, bounds.max_x + 1, 3):
      if x <= bounds.max_x and y <= bounds.max_y:
        if all_quadrants_exist(x, y, 1, 1):
          items.append(
            GenerationItem(
              step=3,
              block_type="1x1",
              top_left_x=x,
              top_left_y=y,
              width=1,
              height=1,
            )
          )

  return items


def create_progress_db(db_path: Path, items: list[GenerationItem]) -> None:
  """
  Create the progress database with the generation plan.

  Args:
      db_path: Path to the database file
      items: List of GenerationItems to insert
  """
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
            CREATE TABLE generation_plan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                top_left_x INTEGER NOT NULL,
                top_left_y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                model_name TEXT,
                started_at REAL,
                completed_at REAL,
                error_message TEXT,
                UNIQUE(step, top_left_x, top_left_y)
            )
        """)

    cursor.execute("""
            CREATE INDEX idx_plan_status ON generation_plan(status, step)
        """)

    cursor.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

    # Insert all items
    for item in items:
      cursor.execute(
        """
                INSERT INTO generation_plan (step, block_type, top_left_x, top_left_y, width, height)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
        (
          item.step,
          item.block_type,
          item.top_left_x,
          item.top_left_y,
          item.width,
          item.height,
        ),
      )

    # Insert metadata
    cursor.execute(
      """
            INSERT INTO metadata (key, value) VALUES ('created_at', ?)
        """,
      (datetime.now().isoformat(),),
    )

    conn.commit()
  finally:
    conn.close()


def create_layer_config(
  output_dir: Path,
  name: str,
  generation_dir: Path,
  source_layer: str,
  bounds: Bounds,
  model_endpoints: list[dict] | None = None,
  generation_params: dict | None = None,
) -> dict:
  """
  Create the layer configuration dictionary and save it.

  Args:
      output_dir: Path to the layer directory
      name: Name of the layer
      generation_dir: Path to the source generation directory
      source_layer: Either 'generations' or 'renders'
      bounds: Bounds of the generation area
      model_endpoints: Optional list of model endpoint configurations
      generation_params: Optional generation parameters

  Returns:
      The configuration dictionary
  """
  config = {
    "name": name,
    "generation_dir": str(generation_dir),
    "source_layer": source_layer,
    "bounds": bounds.to_dict(),
    "model_endpoints": model_endpoints or [],
    "generation_params": generation_params or {},
    "created_at": datetime.now().isoformat(),
  }

  config_path = output_dir / "layer_config.json"
  with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

  return config


def init_layer(
  name: str,
  generation_dir: Path,
  source_layer: str,
  output_dir: Path,
  bounds: Bounds | None = None,
  model_endpoints_file: Path | None = None,
) -> None:
  """
  Initialize a new layer directory.

  Args:
      name: Name of the layer
      generation_dir: Path to the source generation directory
      source_layer: Either 'generations' or 'renders'
      output_dir: Path to create the layer directory
      bounds: Optional explicit bounds (if None, derived from generation_dir)
      model_endpoints_file: Optional path to JSON file with model endpoints
  """
  print(f"\n{'=' * 60}")
  print(f"🗂️  Initializing Layer: {name}")
  print(f"{'=' * 60}")

  # Resolve paths
  generation_dir = generation_dir.resolve()
  output_dir = output_dir.resolve()

  print(f"   Source: {generation_dir}")
  print(f"   Source layer: {source_layer}")
  print(f"   Output: {output_dir}")

  # Get or validate bounds
  if bounds is None:
    print("\n📊 Deriving bounds from source generation...")
    bounds = get_bounds_from_generation_dir(generation_dir, source_layer)
    print(
      f"   Bounds: ({bounds.min_x}, {bounds.min_y}) to ({bounds.max_x}, {bounds.max_y})"
    )
  else:
    print(
      f"\n📊 Using provided bounds: ({bounds.min_x}, {bounds.min_y}) to ({bounds.max_x}, {bounds.max_y})"
    )

  width = bounds.max_x - bounds.min_x + 1
  height = bounds.max_y - bounds.min_y + 1
  print(f"   Size: {width} x {height} = {width * height} quadrants")

  # Get existing quadrants to filter the plan
  print("\n📦 Loading existing quadrants from source...")
  existing_quadrants = get_existing_quadrants(generation_dir, source_layer)
  print(f"   Found {len(existing_quadrants)} quadrants with source data")

  # Load model endpoints if provided
  model_endpoints = None
  if model_endpoints_file:
    with open(model_endpoints_file) as f:
      model_endpoints = json.load(f)
    print(f"\n📡 Loaded {len(model_endpoints)} model endpoints")

  # Create output directory
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "generations").mkdir(exist_ok=True)

  # Generate the tiling plan
  print("\n🔧 Generating tiling plan...")
  items = generate_tiling_plan(bounds, existing_quadrants)

  # Count items by step
  step_counts = {1: 0, 2: 0, 3: 0}
  for item in items:
    step_counts[item.step] += 1

  print(f"   Step 1 (2x2 tiles): {step_counts[1]} items")
  print(f"   Step 2 (1x2/2x1 strips): {step_counts[2]} items")
  print(f"   Step 3 (1x1 corners): {step_counts[3]} items")
  print(f"   Total: {len(items)} items")

  # Calculate total quadrants
  total_quadrants = sum(item.width * item.height for item in items)
  print(f"   Total quadrants to generate: {total_quadrants}")

  # Create the config file
  print("\n📄 Creating layer_config.json...")
  create_layer_config(
    output_dir=output_dir,
    name=name,
    generation_dir=generation_dir,
    source_layer=source_layer,
    bounds=bounds,
    model_endpoints=model_endpoints,
  )

  # Create the progress database
  print("📄 Creating progress.db...")
  db_path = output_dir / "progress.db"
  create_progress_db(db_path, items)

  print(f"\n{'=' * 60}")
  print("✅ Layer initialized successfully!")
  print(f"   Config: {output_dir / 'layer_config.json'}")
  print(f"   Database: {output_dir / 'progress.db'}")
  print("\nNext steps:")
  print("  1. Add model endpoints to layer_config.json")
  print("  2. Run debug visualization:")
  print(
    "     uv run python src/isometric_nyc/generation/debug_generate_full_map_layer.py \\"
  )
  print(f"       --layer-dir {output_dir}")
  print("  3. Start generation:")
  print("     uv run python src/isometric_nyc/generation/generate_full_map_layer.py \\")
  print(f"       --layer-dir {output_dir}")
  print(f"{'=' * 60}")


def main():
  parser = argparse.ArgumentParser(
    description="Initialize a new layer directory for parallel map generation.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Name of the layer (e.g., 'snow_layer')",
  )
  parser.add_argument(
    "--generation-dir",
    type=Path,
    required=True,
    help="Path to the source generation directory (e.g., 'generations/nyc')",
  )
  parser.add_argument(
    "--source-layer",
    type=str,
    choices=["generations", "renders"],
    default="generations",
    help="Which layer to use as source: 'generations' or 'renders' (default: generations)",
  )
  parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Path to create the layer directory (e.g., 'layers/snow')",
  )
  parser.add_argument(
    "--bounds",
    type=str,
    default=None,
    help="Explicit bounds as 'min_x,min_y,max_x,max_y' (default: derived from source)",
  )
  parser.add_argument(
    "--model-endpoints",
    type=Path,
    default=None,
    help="Path to JSON file with model endpoint configurations",
  )

  args = parser.parse_args()

  # Parse bounds if provided
  bounds = None
  if args.bounds:
    bounds = Bounds.from_string(args.bounds)

  try:
    init_layer(
      name=args.name,
      generation_dir=args.generation_dir,
      source_layer=args.source_layer,
      output_dir=args.output,
      bounds=bounds,
      model_endpoints_file=args.model_endpoints,
    )
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
