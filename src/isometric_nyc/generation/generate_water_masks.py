"""
Automated Water Mask Generation - Generate water masks for WATER_EDGE quadrants.

This script automatically generates water masks for quadrants classified as WATER_EDGE
(tiles with partial water content). It uses a random sampling approach with generation
rules similar to tile generation.

Algorithm:
1. Select a random WATER_EDGE quadrant without a water mask (or use --x/--y override)
2. Check if the quadrant can be generated based on 2x2 block context rules
3. If not, find the closest generatable quadrant and generate that instead
4. Repeat until all WATER_EDGE quadrants have masks or no more can be generated

Usage:
  # Generate a single water mask (random selection):
  uv run python src/isometric_nyc/generation/generate_water_masks.py [generation_dir]

  # Generate at specific coordinates:
  uv run python src/isometric_nyc/generation/generate_water_masks.py [generation_dir] --x 5 --y 10

  # Generate multiple masks:
  uv run python src/isometric_nyc/generation/generate_water_masks.py [generation_dir] --count 10

  # Dry run (show what would be generated):
  uv run python src/isometric_nyc/generation/generate_water_masks.py [generation_dir] --dry-run

Arguments:
  generation_dir: Path to the generation directory (default: generations/nyc)

Options:
  --x X: Override random selection with specific x coordinate
  --y Y: Override random selection with specific y coordinate
  --count N: Number of quadrants to generate (default: 1)
  --model-id ID: Model ID to use for generation (must have is_water_mask=true)
  --dry-run: Show what would be generated without actually generating
  --verbose: Show detailed progress
"""

import argparse
import math
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Point:
  """A 2D point representing a quadrant coordinate."""

  x: int
  y: int

  def __str__(self) -> str:
    return f"({self.x}, {self.y})"

  def distance_to(self, other: "Point") -> float:
    """Calculate Euclidean distance to another point."""
    return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def get_2x2_block_positions(p: Point) -> list[list[Point]]:
  """
  Get all 2x2 blocks that contain the point p.

  A point can be in up to 4 different 2x2 blocks:
  - As top-left
  - As top-right
  - As bottom-left
  - As bottom-right
  """
  blocks = []
  # p as top-left
  blocks.append([p, Point(p.x + 1, p.y), Point(p.x, p.y + 1), Point(p.x + 1, p.y + 1)])
  # p as top-right
  blocks.append([Point(p.x - 1, p.y), p, Point(p.x - 1, p.y + 1), Point(p.x, p.y + 1)])
  # p as bottom-left
  blocks.append([Point(p.x, p.y - 1), Point(p.x + 1, p.y - 1), p, Point(p.x + 1, p.y)])
  # p as bottom-right
  blocks.append([Point(p.x - 1, p.y - 1), Point(p.x, p.y - 1), Point(p.x - 1, p.y), p])
  return blocks


def can_generate_water_mask(p: Point, has_water_mask: set[Point]) -> bool:
  """
  Check if a water mask can be generated at point p.

  A water mask can be generated if 3 of 4 quadrants in at least one 2x2 block
  containing p already have water masks (providing context).

  This is analogous to the 1x1 tile generation rule but for water masks.
  """
  blocks = get_2x2_block_positions(p)
  for block in blocks:
    # Count how many OTHER quadrants (not p) have water masks
    other_with_masks = sum(1 for q in block if q != p and q in has_water_mask)
    if other_with_masks >= 3:
      return True
  return False


def get_water_edge_quadrants_without_masks(
  conn: sqlite3.Connection,
) -> list[tuple[int, int]]:
  """
  Get all WATER_EDGE quadrants that don't have water mask data yet.

  Returns:
    List of (x, y) coordinates for quadrants needing water masks.
  """
  cursor = conn.cursor()
  cursor.execute("""
    SELECT quadrant_x, quadrant_y
    FROM quadrants
    WHERE water_type = 'WATER_EDGE'
      AND (water_mask IS NULL OR length(water_mask) = 0)
    ORDER BY quadrant_x, quadrant_y
  """)
  return [(row[0], row[1]) for row in cursor.fetchall()]


def get_quadrants_with_water_masks(conn: sqlite3.Connection) -> set[Point]:
  """
  Get all quadrants that already have water mask data.

  Returns:
    Set of Points for quadrants with existing water masks.
  """
  cursor = conn.cursor()
  cursor.execute("""
    SELECT quadrant_x, quadrant_y
    FROM quadrants
    WHERE water_mask IS NOT NULL AND length(water_mask) > 0
  """)
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def find_generatable_quadrant(
  target: Point,
  needs_masks: list[Point],
  has_masks: set[Point],
) -> Point | None:
  """
  Find a quadrant that can be generated, starting from target.

  First checks if target itself can be generated.
  If not, searches for the closest quadrant that can be generated.

  Args:
    target: The preferred quadrant to generate
    needs_masks: List of quadrants that need water masks
    has_masks: Set of quadrants that already have water masks

  Returns:
    Point of a generatable quadrant, or None if no quadrant can be generated.
  """
  needs_masks_set = set(needs_masks)

  # First check if target can be generated
  if target in needs_masks_set and can_generate_water_mask(target, has_masks):
    return target

  # Find closest generatable quadrant
  candidates = []
  for p in needs_masks:
    if can_generate_water_mask(p, has_masks):
      distance = target.distance_to(p)
      candidates.append((distance, p))

  if not candidates:
    return None

  # Sort by distance and return closest
  candidates.sort(key=lambda x: x[0])
  return candidates[0][1]


def format_time(seconds: float) -> str:
  """Format seconds into a human-readable time string."""
  if seconds < 60:
    return f"{seconds:.1f}s"
  elif seconds < 3600:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"
  else:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def run_water_mask_generation(
  conn: sqlite3.Connection,
  config: dict,
  quadrant: Point,
  model_id: str,
  verbose: bool = False,
) -> dict:
  """
  Run water mask generation for a single quadrant.

  Args:
    conn: Database connection
    config: Generation config
    quadrant: The quadrant to generate
    model_id: Model ID to use (must have is_water_mask=true)
    verbose: Show detailed output

  Returns:
    Dict with success status and message/error.
  """
  from isometric_nyc.generation.generate_omni import run_generation_for_quadrants
  from isometric_nyc.generation.model_config import load_app_config
  from isometric_nyc.generation.shared import start_web_server

  # Load model config
  app_config = load_app_config()
  model_config = app_config.get_model(model_id)

  if not model_config:
    return {"success": False, "error": f"Model '{model_id}' not found"}

  if not model_config.is_water_mask:
    return {
      "success": False,
      "error": f"Model '{model_id}' does not have is_water_mask=true",
    }

  # Start web server if needed
  port = 5173
  start_web_server(port)

  # Calculate context quadrants (quadrants with existing water masks)
  has_masks = get_quadrants_with_water_masks(conn)
  context_quadrants = []

  # Find context from 2x2 blocks
  blocks = get_2x2_block_positions(quadrant)
  for block in blocks:
    for q in block:
      if q != quadrant and q in has_masks:
        coord = (q.x, q.y)
        if coord not in context_quadrants:
          context_quadrants.append(coord)

  if verbose:
    print(f"   Context quadrants: {context_quadrants}")

  # Run generation
  selected_quadrants = [(quadrant.x, quadrant.y)]

  def status_callback(status: str, message: str) -> None:
    if verbose:
      print(f"   [{status}] {message}")

  result = run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=port,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants if context_quadrants else None,
  )

  return result


def generate_water_masks(
  generation_dir: Path,
  start_x: int | None = None,
  start_y: int | None = None,
  count: int = 1,
  model_id: str | None = None,
  dry_run: bool = False,
  verbose: bool = False,
) -> dict:
  """
  Main function to generate water masks.

  Args:
    generation_dir: Path to generation directory
    start_x: Optional x coordinate to start from (overrides random)
    start_y: Optional y coordinate to start from (overrides random)
    count: Number of quadrants to generate
    model_id: Model ID to use (must have is_water_mask=true)
    dry_run: If True, don't actually generate
    verbose: Show detailed progress

  Returns:
    Dict with statistics about the operation.
  """
  from isometric_nyc.generation.shared import get_generation_config

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Find a water mask model if not specified
  if not model_id:
    from isometric_nyc.generation.model_config import load_app_config

    app_config = load_app_config()
    for model in app_config.models:
      if model.is_water_mask:
        model_id = model.model_id
        break

    if not model_id:
      raise ValueError(
        "No model with is_water_mask=true found. "
        "Add one to app_config.json or specify --model-id"
      )

  print(f"   Using model: {model_id}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    stats = {
      "total_needing_masks": 0,
      "generatable": 0,
      "generated": 0,
      "skipped": 0,
      "errors": 0,
    }

    # Get quadrants needing masks
    needs_masks_raw = get_water_edge_quadrants_without_masks(conn)
    needs_masks = [Point(x, y) for x, y in needs_masks_raw]
    has_masks = get_quadrants_with_water_masks(conn)

    stats["total_needing_masks"] = len(needs_masks)

    if not needs_masks:
      print("   ✅ No WATER_EDGE quadrants need masks!")
      return stats

    print(f"   📊 Found {len(needs_masks):,} WATER_EDGE quadrants needing masks")
    print(f"   📊 {len(has_masks):,} quadrants already have masks (context)")

    # Count how many are generatable
    generatable = [p for p in needs_masks if can_generate_water_mask(p, has_masks)]
    stats["generatable"] = len(generatable)
    print(f"   📊 {len(generatable):,} quadrants are currently generatable")

    if not generatable:
      print(
        "\n   ⚠️  No quadrants can be generated yet. Need more context."
      )
      print(
        "   💡 Try generating some quadrants manually first, or check if water masks exist."
      )
      return stats

    # Process quadrants
    generated_count = 0
    start_time = time.time()

    for i in range(count):
      if generated_count >= len(needs_masks):
        print(f"\n   ✅ All {len(needs_masks)} quadrants have been processed!")
        break

      # Re-fetch current state
      needs_masks_raw = get_water_edge_quadrants_without_masks(conn)
      needs_masks = [Point(x, y) for x, y in needs_masks_raw]
      has_masks = get_quadrants_with_water_masks(conn)

      if not needs_masks:
        print(f"\n   ✅ All quadrants have water masks!")
        break

      # Determine target quadrant
      if i == 0 and start_x is not None and start_y is not None:
        target = Point(start_x, start_y)
        print(f"\n   🎯 Using specified start point: {target}")
      else:
        target = random.choice(needs_masks)
        print(f"\n   🎲 Randomly selected: {target}")

      # Find generatable quadrant
      to_generate = find_generatable_quadrant(target, needs_masks, has_masks)

      if not to_generate:
        print(f"   ⚠️  No generatable quadrant found near {target}")
        stats["skipped"] += 1
        continue

      if to_generate != target:
        print(f"   ➡️  Closest generatable: {to_generate}")

      if dry_run:
        print(f"   🔍 [DRY RUN] Would generate water mask for {to_generate}")
        stats["generated"] += 1
        generated_count += 1
        continue

      # Generate
      print(f"   🎨 Generating water mask for {to_generate}...")
      result = run_water_mask_generation(
        conn, config, to_generate, model_id, verbose
      )

      if result.get("success"):
        print(f"   ✅ Generated water mask for {to_generate}")
        stats["generated"] += 1
        generated_count += 1
      else:
        print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        stats["errors"] += 1

      # Show progress
      elapsed = time.time() - start_time
      if generated_count > 0:
        avg_time = elapsed / generated_count
        remaining = count - i - 1
        eta = avg_time * remaining
        print(
          f"   ⏱️  Progress: {generated_count}/{count} | "
          f"Avg: {format_time(avg_time)} | ETA: {format_time(eta)}"
        )

    return stats

  finally:
    conn.close()


def main():
  load_dotenv()

  parser = argparse.ArgumentParser(
    description="Generate water masks for WATER_EDGE quadrants."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/nyc"),
    help="Path to the generation directory (default: generations/nyc)",
  )
  parser.add_argument(
    "--x",
    type=int,
    default=None,
    help="X coordinate to start from (overrides random selection)",
  )
  parser.add_argument(
    "--y",
    type=int,
    default=None,
    help="Y coordinate to start from (overrides random selection)",
  )
  parser.add_argument(
    "--count",
    "-n",
    type=int,
    default=1,
    help="Number of quadrants to generate (default: 1)",
  )
  parser.add_argument(
    "--model-id",
    type=str,
    default=None,
    help="Model ID to use for generation (must have is_water_mask=true)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be generated without actually generating",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Show detailed progress",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  # Validate x/y args
  if (args.x is not None) != (args.y is not None):
    print("❌ Error: Both --x and --y must be specified together")
    return 1

  print(f"\n{'=' * 60}")
  print("💧 Automated Water Mask Generation")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")
  print(f"   Count: {args.count}")
  if args.x is not None:
    print(f"   Start point: ({args.x}, {args.y})")
  else:
    print("   Start point: random")
  if args.model_id:
    print(f"   Model: {args.model_id}")
  if args.dry_run:
    print("   Mode: DRY RUN")
  print()

  try:
    stats = generate_water_masks(
      generation_dir,
      start_x=args.x,
      start_y=args.y,
      count=args.count,
      model_id=args.model_id,
      dry_run=args.dry_run,
      verbose=args.verbose,
    )

    print(f"\n{'=' * 60}")
    print("📈 Results")
    print(f"{'=' * 60}")
    print(f"   Total needing masks: {stats['total_needing_masks']:,}")
    print(f"   Currently generatable: {stats['generatable']:,}")
    print(f"   Generated: {stats['generated']:,}")
    print(f"   Skipped: {stats['skipped']:,}")
    print(f"   Errors: {stats['errors']:,}")

    if args.dry_run:
      print("\n⚠️  DRY RUN - No changes were made")

    return 0

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  sys.exit(main())

