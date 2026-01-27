"""
Detect Water Tiles - Scan generation tiles and mark those containing water color.

This script iterates over all tiles in the quadrants database and marks them
with the is_water flag if they contain the water color #4A6372.

Usage:
  uv run python src/isometric_nyc/generation/detect_water_tiles.py [generation_dir]

Arguments:
  generation_dir: Path to the generation directory (default: generations/nyc)

Options:
  --threshold N: Minimum percentage of pixels that must be water color (default: 1.0)
  --dry-run: Show what would be marked without making changes
  --verbose: Show each water tile as it's detected
  --workers N: Number of parallel workers (default: CPU count)
"""

import argparse
import multiprocessing
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

# The water color to detect
WATER_COLOR_HEX = "4A6372"
WATER_COLOR_RGB = tuple(int(WATER_COLOR_HEX[i : i + 2], 16) for i in (0, 2, 4))

# Water status values in the database
WATER_STATUS_NOT_WATER = 0  # Auto-detected as not water (can be overwritten)
WATER_STATUS_WATER = 1  # Water tile (auto or manual)
WATER_STATUS_EXPLICIT_NOT_WATER = -1  # Explicitly marked as NOT water (protected)


def ensure_is_water_column(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table."""
  cursor = conn.cursor()
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


def get_all_quadrants_with_generations(
  conn: sqlite3.Connection,
) -> list[tuple[int, int, bytes, int]]:
  """
  Get all quadrants that have generation data.

  Returns:
    List of (quadrant_x, quadrant_y, generation_bytes, is_water_status) tuples.
    is_water_status is -1 for explicitly not water, 0 for unclassified, 1 for water.
  """
  cursor = conn.cursor()
  cursor.execute("""
    SELECT quadrant_x, quadrant_y, generation, COALESCE(is_water, 0)
    FROM quadrants
    WHERE generation IS NOT NULL
    ORDER BY quadrant_x, quadrant_y
  """)
  return [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]


def analyze_tile(
  args: tuple[int, int, bytes, float, int],
) -> tuple[int, int, bool, float]:
  """
  Analyze a single tile for water content. Worker function for parallel processing.

  Args:
    args: Tuple of (x, y, png_bytes, threshold_percent, tolerance)

  Returns:
    Tuple of (x, y, is_water, percentage)
  """
  x, y, png_bytes, threshold_percent, tolerance = args

  try:
    img = Image.open(BytesIO(png_bytes))
    img = img.convert("RGB")

    pixels = list(img.getdata())
    total_pixels = len(pixels)

    if total_pixels == 0:
      return x, y, False, 0.0

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
    is_water = percentage >= threshold_percent

    return x, y, is_water, percentage

  except Exception:
    return x, y, False, 0.0


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
  water_count: int,
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

  # Build the status line
  status = (
    f"\r🔍 [{bar}] {current:,}/{total:,} ({progress * 100:.1f}%) "
    f"| 💧 {water_count:,} water | ⚡ {workers} workers | ETA: {eta_str}  "
  )

  # Print with carriage return to overwrite
  sys.stdout.write(status)
  sys.stdout.flush()


def detect_water_tiles(
  generation_dir: Path,
  threshold_percent: float = 1.0,
  dry_run: bool = False,
  verbose: bool = False,
  num_workers: int | None = None,
) -> dict:
  """
  Scan all tiles and detect which ones contain water using parallel processing.

  Args:
    generation_dir: Path to the generation directory
    threshold_percent: Minimum percentage of pixels that must be water color
    dry_run: If True, don't make any changes to the database
    verbose: If True, print each water tile as detected
    num_workers: Number of parallel workers (default: CPU count)

  Returns:
    Dict with statistics about the scan
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Determine number of workers
  if num_workers is None:
    num_workers = os.cpu_count() or 4
  num_workers = max(1, num_workers)

  conn = sqlite3.connect(db_path)

  try:
    # Ensure the is_water column exists
    if not dry_run:
      ensure_is_water_column(conn)

    # Get all quadrants with generations
    print("   Loading quadrants from database...")
    all_quadrants = get_all_quadrants_with_generations(conn)
    total_in_db = len(all_quadrants)

    # Filter out explicitly protected tiles (is_water = -1)
    protected_quadrants = [
      (x, y)
      for x, y, _, is_water in all_quadrants
      if is_water == WATER_STATUS_EXPLICIT_NOT_WATER
    ]
    quadrants_to_scan = [
      (x, y, gen_bytes, is_water)
      for x, y, gen_bytes, is_water in all_quadrants
      if is_water != WATER_STATUS_EXPLICIT_NOT_WATER
    ]

    total = len(quadrants_to_scan)
    skipped = len(protected_quadrants)

    print(f"\n📊 Found {total_in_db:,} quadrants with generations")
    if skipped > 0:
      print(f"   🛡️  Skipping {skipped:,} explicitly protected tiles")

    stats = {
      "total_scanned": 0,
      "water_tiles": 0,
      "non_water_tiles": 0,
      "skipped_protected": skipped,
      "errors": 0,
      "water_quadrants": [],
      "protected_quadrants": protected_quadrants,
    }

    if total == 0:
      print("   No tiles to scan (all protected or none exist)")
      return stats

    print(f"🔍 Scanning {total:,} tiles for water color #{WATER_COLOR_HEX}...")
    print(f"   Threshold: {threshold_percent}% of pixels")
    print(f"   Workers: {num_workers}")
    if verbose:
      print()  # Extra line before verbose output

    start_time = time.time()
    tolerance = 5

    # Prepare work items (excluding protected tiles)
    work_items = [
      (x, y, generation_bytes, threshold_percent, tolerance)
      for x, y, generation_bytes, _ in quadrants_to_scan
    ]

    # Lists to batch database updates
    water_coords: list[tuple[int, int]] = []
    non_water_coords: list[tuple[int, int]] = []

    # Process in parallel
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
      # Submit all tasks
      future_to_coord = {
        executor.submit(analyze_tile, item): (item[0], item[1]) for item in work_items
      }

      # Process results as they complete
      for future in as_completed(future_to_coord):
        x, y, is_water, percentage = future.result()
        completed += 1
        stats["total_scanned"] += 1

        if is_water:
          stats["water_tiles"] += 1
          stats["water_quadrants"].append((x, y, percentage))
          water_coords.append((x, y))

          if verbose:
            # Clear the progress bar line first
            sys.stdout.write("\r" + " " * 120 + "\r")
            if dry_run:
              print(f"   💧 ({x:4d}, {y:4d}) - {percentage:.1f}% water (would mark)")
            else:
              print(f"   💧 ({x:4d}, {y:4d}) - {percentage:.1f}% water ✓")
        else:
          stats["non_water_tiles"] += 1
          non_water_coords.append((x, y))

        # Update progress bar periodically
        if total < 100 or completed % 50 == 0 or completed == total:
          print_progress(
            completed, total, stats["water_tiles"], start_time, num_workers
          )

    # Clear the progress bar and print final stats
    elapsed = time.time() - start_time
    sys.stdout.write("\r" + " " * 120 + "\r")
    tiles_per_sec = total / elapsed if elapsed > 0 else 0
    print(
      f"✅ Scanned {total:,} tiles in {format_time(elapsed)} "
      f"({tiles_per_sec:.0f} tiles/sec)"
    )

    # Batch update database
    if not dry_run:
      print("   Updating database...")
      cursor = conn.cursor()

      # Mark water tiles
      if water_coords:
        cursor.executemany(
          "UPDATE quadrants SET is_water = 1 WHERE quadrant_x = ? AND quadrant_y = ?",
          water_coords,
        )

      # Unmark non-water tiles
      if non_water_coords:
        cursor.executemany(
          "UPDATE quadrants SET is_water = 0 WHERE quadrant_x = ? AND quadrant_y = ?",
          non_water_coords,
        )

      conn.commit()
      print("   ✓ Database updated")

    return stats

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Detect and mark water tiles in the generation database."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/nyc"),
    help="Path to the generation directory (default: generations/nyc)",
  )
  parser.add_argument(
    "--threshold",
    type=float,
    default=1.0,
    help="Minimum percentage of pixels that must be water color (default: 1.0)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be marked without making changes",
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Show each water tile as it's detected",
  )
  parser.add_argument(
    "--workers",
    "-w",
    type=int,
    default=None,
    help=f"Number of parallel workers (default: CPU count = {os.cpu_count()})",
  )

  args = parser.parse_args()

  # Resolve path
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  num_workers = args.workers or os.cpu_count() or 4

  print(f"\n{'=' * 60}")
  print("💧 Water Tile Detection (Parallel)")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")
  print(f"   Water color: #{WATER_COLOR_HEX} (RGB{WATER_COLOR_RGB})")
  print(f"   Threshold: {args.threshold}%")
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
    stats = detect_water_tiles(
      generation_dir,
      threshold_percent=args.threshold,
      dry_run=args.dry_run,
      verbose=args.verbose,
      num_workers=num_workers,
    )

    print(f"\n{'=' * 60}")
    print("📈 Results")
    print(f"{'=' * 60}")
    print(f"   Total scanned: {stats['total_scanned']:,}")
    print(f"   Water tiles:   {stats['water_tiles']:,}")
    print(f"   Non-water:     {stats['non_water_tiles']:,}")
    if stats.get("skipped_protected", 0) > 0:
      print(f"   🛡️  Protected:   {stats['skipped_protected']:,} (skipped)")

    # Only show the full list if not already shown in verbose mode
    if stats["water_quadrants"] and not args.verbose:
      print(f"\n💧 Water tiles found ({len(stats['water_quadrants']):,}):")
      # Sort by coordinates for consistent output
      sorted_water = sorted(stats["water_quadrants"], key=lambda t: (t[0], t[1]))
      for x, y, pct in sorted_water:
        print(f"      ({x}, {y}) - {pct:.1f}%")

    if args.dry_run:
      print("\n⚠️  DRY RUN - No changes were made to the database")
    else:
      print(f"\n✅ Marked {stats['water_tiles']:,} tile(s) as water")

    return 0

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  # Required for multiprocessing on Windows and macOS
  multiprocessing.freeze_support()
  exit(main())
