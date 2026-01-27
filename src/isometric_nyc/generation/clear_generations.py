"""
Clear generations from specific quadrant coordinates.

Usage:
  # Clear a single quadrant
  uv run python src/isometric_nyc/generation/clear_generations.py <generation_dir> <x> <y>

  # Clear a tile (all 4 quadrants)
  uv run python src/isometric_nyc/generation/clear_generations.py <generation_dir> <x> <y> --tile

  # Clear all generations
  uv run python src/isometric_nyc/generation/clear_generations.py <generation_dir> --all

  # Dry run (show what would be cleared)
  uv run python src/isometric_nyc/generation/clear_generations.py <generation_dir> <x> <y> --dry-run
"""

import argparse
import sqlite3
from pathlib import Path


def get_quadrant_info(conn: sqlite3.Connection, x: int, y: int) -> dict | None:
  """Get info about a quadrant."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT quadrant_x, quadrant_y, 
           render IS NOT NULL as has_render,
           generation IS NOT NULL as has_gen
    FROM quadrants
    WHERE quadrant_x = ? AND quadrant_y = ?
    """,
    (x, y),
  )
  row = cursor.fetchone()
  if not row:
    return None
  return {
    "x": row[0],
    "y": row[1],
    "has_render": bool(row[2]),
    "has_generation": bool(row[3]),
  }


def clear_generation(
  conn: sqlite3.Connection, x: int, y: int, dry_run: bool = False
) -> bool:
  """Clear generation for a single quadrant."""
  info = get_quadrant_info(conn, x, y)
  if not info:
    print(f"   ⚠️  Quadrant ({x}, {y}) not found in database")
    return False

  if not info["has_generation"]:
    print(f"   ⏭️  Quadrant ({x}, {y}) has no generation to clear")
    return False

  if dry_run:
    print(f"   🔍 Would clear generation for quadrant ({x}, {y})")
    return True

  cursor = conn.cursor()
  cursor.execute(
    "UPDATE quadrants SET generation = NULL WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  conn.commit()
  print(f"   ✓ Cleared generation for quadrant ({x}, {y})")
  return True


def clear_tile_generations(
  conn: sqlite3.Connection, x: int, y: int, dry_run: bool = False
) -> int:
  """Clear generations for all 4 quadrants in a tile."""
  positions = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
  cleared = 0
  for qx, qy in positions:
    if clear_generation(conn, qx, qy, dry_run):
      cleared += 1
  return cleared


def clear_all_generations(conn: sqlite3.Connection, dry_run: bool = False) -> int:
  """Clear all generations in the database."""
  cursor = conn.cursor()

  # Count how many have generations
  cursor.execute("SELECT COUNT(*) FROM quadrants WHERE generation IS NOT NULL")
  count = cursor.fetchone()[0]

  if count == 0:
    print("   ⏭️  No generations to clear")
    return 0

  if dry_run:
    print(f"   🔍 Would clear {count} generation(s)")
    return count

  cursor.execute("UPDATE quadrants SET generation = NULL")
  conn.commit()
  print(f"   ✓ Cleared {count} generation(s)")
  return count


def show_status(conn: sqlite3.Connection) -> None:
  """Show the current status of all quadrants."""
  cursor = conn.cursor()
  cursor.execute(
    """
    SELECT quadrant_x, quadrant_y,
           render IS NOT NULL as has_render,
           generation IS NOT NULL as has_gen
    FROM quadrants
    ORDER BY quadrant_y, quadrant_x
    """
  )
  rows = cursor.fetchall()

  print("\n📊 Quadrant Status:")
  print("   " + "-" * 40)
  print(f"   {'Coord':<10} {'Render':<10} {'Generation':<10}")
  print("   " + "-" * 40)

  for row in rows:
    x, y, has_render, has_gen = row
    render_str = "✓" if has_render else "✗"
    gen_str = "✓" if has_gen else "✗"
    print(f"   ({x}, {y}){'':<4} {render_str:<10} {gen_str:<10}")

  print("   " + "-" * 40)

  # Summary
  cursor.execute("SELECT COUNT(*) FROM quadrants")
  total = cursor.fetchone()[0]
  cursor.execute("SELECT COUNT(*) FROM quadrants WHERE render IS NOT NULL")
  renders = cursor.fetchone()[0]
  cursor.execute("SELECT COUNT(*) FROM quadrants WHERE generation IS NOT NULL")
  gens = cursor.fetchone()[0]

  print(f"\n   Total quadrants: {total}")
  print(f"   With renders: {renders}")
  print(f"   With generations: {gens}")


def main():
  parser = argparse.ArgumentParser(
    description="Clear generations from specific quadrant coordinates."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    nargs="?",
    help="Quadrant x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    nargs="?",
    help="Quadrant y coordinate",
  )
  parser.add_argument(
    "--tile",
    action="store_true",
    help="Clear all 4 quadrants for the tile starting at (x, y)",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Clear ALL generations in the database",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be cleared without actually clearing",
  )
  parser.add_argument(
    "--status",
    action="store_true",
    help="Show status of all quadrants",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  conn = sqlite3.connect(db_path)

  try:
    # Show status if requested
    if args.status:
      show_status(conn)
      return 0

    # Clear all if requested
    if args.all:
      print("\n🗑️  Clearing ALL generations...")
      if args.dry_run:
        print("   (DRY RUN - no changes will be made)")
      clear_all_generations(conn, args.dry_run)
      return 0

    # Need coordinates for single/tile clear
    if args.x is None or args.y is None:
      print("❌ Error: x and y coordinates required (or use --all)")
      return 1

    if args.tile:
      print(f"\n🗑️  Clearing tile at ({args.x}, {args.y})...")
      print(
        f"   Quadrants: ({args.x},{args.y}), ({args.x + 1},{args.y}), "
        f"({args.x},{args.y + 1}), ({args.x + 1},{args.y + 1})"
      )
      if args.dry_run:
        print("   (DRY RUN - no changes will be made)")
      cleared = clear_tile_generations(conn, args.x, args.y, args.dry_run)
      print(f"\n   Total cleared: {cleared}")
    else:
      print(f"\n🗑️  Clearing quadrant ({args.x}, {args.y})...")
      if args.dry_run:
        print("   (DRY RUN - no changes will be made)")
      clear_generation(conn, args.x, args.y, args.dry_run)

    return 0

  finally:
    conn.close()


if __name__ == "__main__":
  exit(main())
