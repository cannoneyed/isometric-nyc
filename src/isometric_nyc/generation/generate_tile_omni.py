"""
Generate pixel art for a tile using the Oxen.ai omni model.

This script generates pixel art for quadrants at specific coordinates.
It uses the generation library from generate_omni.py, same as the web app.

Usage:
  uv run python src/isometric_nyc/generation/generate_tile_omni.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/generation/generate_tile_omni.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.
"""

import argparse
import sqlite3
from pathlib import Path

from isometric_nyc.generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# =============================================================================
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


def get_tile_quadrants(anchor_x: int, anchor_y: int) -> list[tuple[int, int]]:
  """
  Get all 4 quadrant coordinates for a tile anchored at (anchor_x, anchor_y).

  Returns:
    List of (x, y) tuples for TL, TR, BL, BR quadrants
  """
  return [
    (anchor_x, anchor_y),  # TL
    (anchor_x + 1, anchor_y),  # TR
    (anchor_x, anchor_y + 1),  # BL
    (anchor_x + 1, anchor_y + 1),  # BR
  ]


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile_omni(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the omni model.

  Args:
    generation_dir: Path to the generation directory
    x: Quadrant x coordinate (or target x if target_position is specified)
    y: Quadrant y coordinate (or target y if target_position is specified)
    port: Web server port for rendering
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), generates all 4 quadrants in the tile starting at (x, y).
      If specified, (x, y) is the target quadrant to generate, positioned within
      the 2x2 tile context.

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🎯 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🎯 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation using the shared library
    result = run_generation_for_quadrants(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      port=port,
      status_callback=status_callback,
    )

    if result["success"]:
      print(f"\n{'=' * 60}")
      print(f"✅ Generation complete: {result['message']}")
      print(f"{'=' * 60}")
      return True
    else:
      print(f"\n{'=' * 60}")
      print(f"❌ Generation failed: {result['error']}")
      print(f"{'=' * 60}")
      return False

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API (omni model)."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Quadrant x coordinate (tile anchor x, or target x if --target-position is used)",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Quadrant y coordinate (tile anchor y, or target y if --target-position is used)",
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
    "--target-position",
    "-t",
    choices=["tl", "tr", "bl", "br"],
    help=(
      "Position of target quadrant (x,y) within the 2x2 tile context. "
      "tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right. "
      "If specified, (x,y) is the target to generate and surrounding quadrants "
      "provide context. If not specified, (x,y) is the tile anchor and all "
      "4 quadrants are generated."
    ),
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

    success = generate_tile_omni(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
    )
    return 0 if success else 1

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
