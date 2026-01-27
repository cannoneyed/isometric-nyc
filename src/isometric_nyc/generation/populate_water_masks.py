"""
Populate Water Masks - Generate water mask images for all quadrants.

This script analyzes each tile in the quadrants database and:
1. For tiles that are ALL water (100% water color #4A6372): creates a white mask, sets water_type=ALL_WATER
2. For tiles that have NO water: creates a black mask, sets water_type=ALL_LAND
3. For tiles with partial water (WATER_EDGE): leaves water_mask empty for manual processing

Usage:
  uv run python src/isometric_nyc/generation/populate_water_masks.py [generation_dir]

Arguments:
  generation_dir: Path to the generation directory (default: generations/nyc)

Options:
  --dry-run: Show what would be done without making changes
  --verbose: Show each tile as it's processed
  --workers N: Number of parallel workers (default: CPU count)
"""

import argparse
import multiprocessing
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from io import BytesIO
from pathlib import Path

from PIL import Image

# The water color to detect
WATER_COLOR_HEX = "4A6372"
WATER_COLOR_RGB = tuple(int(WATER_COLOR_HEX[i : i + 2], 16) for i in (0, 2, 4))


class WaterType(str, Enum):
  """Water type classification for tiles."""

  ALL_WATER = "ALL_WATER"
  ALL_LAND = "ALL_LAND"
  WATER_EDGE = "WATER_EDGE"


def ensure_water_mask_columns(conn: sqlite3.Connection) -> None:
  """Ensure the water_mask and water_type columns exist in the quadrants table."""
  cursor = conn.cursor()
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]

  if "water_mask" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN water_mask BLOB")
    conn.commit()
    print("📝 Added 'water_mask' column to quadrants table")

  if "water_type" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN water_type TEXT")
    conn.commit()
    print("📝 Added 'water_type' column to quadrants table")


def get_all_quadrants_with_generations(
  conn: sqlite3.Connection,
) -> list[tuple[int, int, bytes]]:
  """
  Get all quadrants that have generation data.

  Returns:
    List of (quadrant_x, quadrant_y, generation_bytes) tuples.
  """
  cursor = conn.cursor()
  cursor.execute("""
    SELECT quadrant_x, quadrant_y, generation
    FROM quadrants
    WHERE generation IS NOT NULL
    ORDER BY quadrant_x, quadrant_y
  """)
  return [(row[0], row[1], row[2]) for row in cursor.fetchall()]


def create_solid_mask(width: int, height: int, color: tuple[int, int, int]) -> bytes:
  """Create a solid color PNG mask."""
  img = Image.new("RGB", (width, height), color)
  buffer = BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def analyze_tile_water_content(
  args: tuple[int, int, bytes, int],
) -> tuple[int, int, WaterType, float, int, int]:
  """
  Analyze a single tile for water content.

  Args:
    args: Tuple of (x, y, png_bytes, tolerance)

  Returns:
    Tuple of (x, y, water_type, water_percentage, width, height)
  """
  x, y, png_bytes, tolerance = args

  try:
    img = Image.open(BytesIO(png_bytes))
    img_rgb = img.convert("RGB")
    width, height = img_rgb.size

    pixels = list(img_rgb.getdata())
    total_pixels = len(pixels)

    if total_pixels == 0:
      return x, y, WaterType.ALL_LAND, 0.0, width, height

    # Count pixels that match the water color (within tolerance)
    water_r, water_g, water_b = WATER_COLOR_RGB
    matching_pixels = 0

    for r, g, b in pixels:
      if (
        abs(r - water_r) <= tolerance
        and abs(g - water_g) <= tolerance
        and abs(b - water_b) <= tolerance
      ):
        matching_pixels += 1

    percentage = (matching_pixels / total_pixels) * 100

    # Classify based on percentage
    if percentage >= 99.5:  # Essentially all water
      water_type = WaterType.ALL_WATER
    elif percentage <= 0.5:  # Essentially no water
      water_type = WaterType.ALL_LAND
    else:
      water_type = WaterType.WATER_EDGE

    return x, y, water_type, percentage, width, height

  except Exception as e:
    print(f"Error analyzing tile ({x}, {y}): {e}")
    return x, y, WaterType.ALL_LAND, 0.0, 256, 256


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


def print_progress(
  current: int,
  total: int,
  stats: dict,
  start_time: float,
  workers: int,
  bar_width: int = 30,
) -> None:
  """Print a progress bar with stats."""
  if total == 0:
    return

  progress = current / total
  filled = int(bar_width * progress)
  bar = "█" * filled + "░" * (bar_width - filled)

  elapsed = time.time() - start_time
  if current > 0:
    eta = (elapsed / current) * (total - current)
    eta_str = format_time(eta)
  else:
    eta_str = "..."

  status = (
    f"\r🔍 [{bar}] {current:,}/{total:,} ({progress * 100:.1f}%) "
    f"| 💧 {stats['all_water']:,} | 🏝️ {stats['all_land']:,} | 🏖️ {stats['water_edge']:,} "
    f"| ⚡ {workers} | ETA: {eta_str}  "
  )

  sys.stdout.write(status)
  sys.stdout.flush()


def populate_water_masks(
  generation_dir: Path,
  dry_run: bool = False,
  verbose: bool = False,
  num_workers: int | None = None,
) -> dict:
  """
  Analyze all tiles and populate water masks for ALL_WATER and ALL_LAND tiles.

  Args:
    generation_dir: Path to the generation directory
    dry_run: If True, don't make any changes to the database
    verbose: If True, print each tile as processed
    num_workers: Number of parallel workers (default: CPU count)

  Returns:
    Dict with statistics about the operation
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  if num_workers is None:
    num_workers = os.cpu_count() or 4
  num_workers = max(1, num_workers)

  conn = sqlite3.connect(db_path)

  try:
    # Ensure columns exist
    if not dry_run:
      ensure_water_mask_columns(conn)

    # Get all quadrants with generations
    print("   Loading quadrants from database...")
    all_quadrants = get_all_quadrants_with_generations(conn)
    total = len(all_quadrants)

    print(f"\n📊 Found {total:,} quadrants with generations")

    stats = {
      "total_processed": 0,
      "all_water": 0,
      "all_land": 0,
      "water_edge": 0,
      "errors": 0,
      "quadrants_by_type": {
        "ALL_WATER": [],
        "ALL_LAND": [],
        "WATER_EDGE": [],
      },
    }

    if total == 0:
      print("   No tiles to process")
      return stats

    print(f"🔍 Analyzing {total:,} tiles for water content...")
    print(f"   Water color: #{WATER_COLOR_HEX} (RGB{WATER_COLOR_RGB})")
    print(f"   Workers: {num_workers}")
    if verbose:
      print()

    start_time = time.time()
    tolerance = 5

    # Prepare work items
    work_items = [
      (x, y, generation_bytes, tolerance) for x, y, generation_bytes in all_quadrants
    ]

    # Results to batch update
    results: list[tuple[int, int, WaterType, float, int, int]] = []

    # Process in parallel
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
      future_to_coord = {
        executor.submit(analyze_tile_water_content, item): (item[0], item[1])
        for item in work_items
      }

      for future in as_completed(future_to_coord):
        x, y, water_type, percentage, width, height = future.result()
        completed += 1
        stats["total_processed"] += 1

        results.append((x, y, water_type, percentage, width, height))

        if water_type == WaterType.ALL_WATER:
          stats["all_water"] += 1
          stats["quadrants_by_type"]["ALL_WATER"].append((x, y, percentage))
        elif water_type == WaterType.ALL_LAND:
          stats["all_land"] += 1
          stats["quadrants_by_type"]["ALL_LAND"].append((x, y, percentage))
        else:
          stats["water_edge"] += 1
          stats["quadrants_by_type"]["WATER_EDGE"].append((x, y, percentage))

        if verbose:
          sys.stdout.write("\r" + " " * 120 + "\r")
          icon = (
            "💧"
            if water_type == WaterType.ALL_WATER
            else ("🏝️" if water_type == WaterType.ALL_LAND else "🏖️")
          )
          print(
            f"   {icon} ({x:4d}, {y:4d}) - {percentage:.1f}% water → {water_type.value}"
          )

        # Update progress bar frequently (every 10 items or always for small batches)
        if total < 100 or completed % 10 == 0 or completed == total:
          print_progress(completed, total, stats, start_time, num_workers)

    # Clear progress bar
    elapsed = time.time() - start_time
    sys.stdout.write("\r" + " " * 120 + "\r")
    tiles_per_sec = total / elapsed if elapsed > 0 else 0
    print(
      f"✅ Analyzed {total:,} tiles in {format_time(elapsed)} ({tiles_per_sec:.0f} tiles/sec)"
    )

    # Batch update database
    if not dry_run:
      print("\n💾 Updating database with water masks...")
      cursor = conn.cursor()

      updated_count = 0
      total_results = len(results)
      update_start_time = time.time()
      bar_width = 30

      for idx, (x, y, water_type, percentage, width, height) in enumerate(results):
        if water_type == WaterType.ALL_WATER:
          # Create white mask (255, 255, 255)
          mask_bytes = create_solid_mask(width, height, (255, 255, 255))
          cursor.execute(
            """
            UPDATE quadrants
            SET water_mask = ?, water_type = ?
            WHERE quadrant_x = ? AND quadrant_y = ?
            """,
            (mask_bytes, water_type.value, x, y),
          )
          updated_count += 1
        elif water_type == WaterType.ALL_LAND:
          # Create black mask (0, 0, 0)
          mask_bytes = create_solid_mask(width, height, (0, 0, 0))
          cursor.execute(
            """
            UPDATE quadrants
            SET water_mask = ?, water_type = ?
            WHERE quadrant_x = ? AND quadrant_y = ?
            """,
            (mask_bytes, water_type.value, x, y),
          )
          updated_count += 1
        else:
          # WATER_EDGE - just set the type, leave mask empty for manual processing
          cursor.execute(
            """
            UPDATE quadrants
            SET water_type = ?
            WHERE quadrant_x = ? AND quadrant_y = ?
            """,
            (water_type.value, x, y),
          )

        # Update progress bar every 100 items or at the end
        current = idx + 1
        if total_results < 100 or current % 100 == 0 or current == total_results:
          progress = current / total_results
          filled = int(bar_width * progress)
          bar = "█" * filled + "░" * (bar_width - filled)
          elapsed = time.time() - update_start_time
          if current > 0 and elapsed > 0:
            eta = (elapsed / current) * (total_results - current)
            eta_str = format_time(eta)
            rate = current / elapsed
          else:
            eta_str = "..."
            rate = 0
          sys.stdout.write(
            f"\r💾 [{bar}] {current:,}/{total_results:,} ({progress * 100:.1f}%) "
            f"| ✓ {updated_count:,} masks | {rate:.0f}/sec | ETA: {eta_str}  "
          )
          sys.stdout.flush()

      conn.commit()
      update_elapsed = time.time() - update_start_time
      sys.stdout.write("\r" + " " * 100 + "\r")
      print(
        f"✅ Updated {updated_count:,} quadrant(s) with water masks in {format_time(update_elapsed)}"
      )

    return stats

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Populate water masks for tiles in the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/nyc"),
    help="Path to the generation directory (default: generations/nyc)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be done without making changes",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Show each tile as it's processed",
  )
  parser.add_argument(
    "--workers",
    "-w",
    type=int,
    default=None,
    help=f"Number of parallel workers (default: CPU count = {os.cpu_count()})",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  num_workers = args.workers or os.cpu_count() or 4

  print(f"\n{'=' * 60}")
  print("💧 Water Mask Population")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")
  print(f"   Water color: #{WATER_COLOR_HEX} (RGB{WATER_COLOR_RGB})")
  print(f"   Workers: {num_workers}")
  mode_parts = []
  if args.dry_run:
    mode_parts.append("DRY RUN")
  if args.verbose:
    mode_parts.append("VERBOSE")
  if mode_parts:
    print(f"   Mode: {', '.join(mode_parts)}")
  print()

  try:
    stats = populate_water_masks(
      generation_dir,
      dry_run=args.dry_run,
      verbose=args.verbose,
      num_workers=num_workers,
    )

    print(f"\n{'=' * 60}")
    print("📈 Results")
    print(f"{'=' * 60}")
    print(f"   Total processed: {stats['total_processed']:,}")
    print(f"   💧 ALL_WATER:    {stats['all_water']:,} (white mask)")
    print(f"   🏝️ ALL_LAND:     {stats['all_land']:,} (black mask)")
    print(f"   🏖️ WATER_EDGE:   {stats['water_edge']:,} (needs manual mask)")

    if args.dry_run:
      print("\n⚠️  DRY RUN - No changes were made to the database")
    else:
      print(
        f"\n✅ Created water masks for {stats['all_water'] + stats['all_land']:,} tile(s)"
      )
      if stats["water_edge"] > 0:
        print(
          f"⚠️  {stats['water_edge']:,} tile(s) need manual water mask creation (WATER_EDGE)"
        )

    return 0

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  multiprocessing.freeze_support()
  exit(main())
