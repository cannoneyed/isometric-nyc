"""
Fill quadrants with solid water color.

Fills a rectangular region of quadrants with a solid water color (#4A6372).

Usage:
  uv run python src/isometric_nyc/generation/fill_water.py <generation_dir> --tl X,Y --br X,Y

Examples:
  # Fill quadrants from (0,0) to (5,5) with water
  uv run python src/isometric_nyc/generation/fill_water.py generations/nyc --tl 0,0 --br 5,5

  # Dry run to see what would be filled
  uv run python src/isometric_nyc/generation/fill_water.py generations/nyc --tl 0,0 --br 5,5 --dry-run

  # Overwrite existing quadrants
  uv run python src/isometric_nyc/generation/fill_water.py generations/nyc --tl 0,0 --br 5,5 --overwrite
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from PIL import Image

from isometric_nyc.generation.shared import (
    get_generation_config,
    get_quadrant_generation,
    image_to_png_bytes,
    save_quadrant_generation,
)

# Constants
QUADRANT_SIZE = 512
WATER_COLOR = (0x4A, 0x63, 0x72)  # #4A6372


def parse_coordinate(coord_str: str) -> tuple[int, int]:
    """Parse a coordinate string like '10,15' into (x, y) tuple."""
    try:
        parts = coord_str.split(",")
        if len(parts) != 2:
            raise ValueError()
        return int(parts[0].strip()), int(parts[1].strip())
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid coordinate format: '{coord_str}'. Expected 'X,Y' (e.g., '10,15')"
        )


def create_water_image(width: int, height: int) -> Image.Image:
    """Create a solid water color image."""
    return Image.new("RGB", (width, height), WATER_COLOR)


def quadrant_has_data(conn: sqlite3.Connection, x: int, y: int) -> bool:
    """Check if a quadrant already has generation data."""
    gen = get_quadrant_generation(conn, x, y)
    return gen is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill quadrants with solid water color (#4A6372)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory (contains quadrants.db)",
    )
    parser.add_argument(
        "--tl",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Top-left coordinate of the region to fill (e.g., '0,0')",
    )
    parser.add_argument(
        "--br",
        type=parse_coordinate,
        required=True,
        metavar="X,Y",
        help="Bottom-right coordinate of the region to fill (e.g., '5,5')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quadrants (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filled without actually filling",
    )

    args = parser.parse_args()

    # Resolve paths
    generation_dir = args.generation_dir.resolve()
    tl = args.tl
    br = args.br

    # Validate inputs
    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Validate coordinate range
    if tl[0] > br[0] or tl[1] > br[1]:
        print(
            f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
        )
        return 1

    # Calculate total quadrants
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1
    total = width * height

    print("=" * 50)
    print("🌊 Fill Water Quadrants")
    print("=" * 50)
    print(f"📁 Generation dir: {generation_dir}")
    print(f"📍 Region: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    print(f"📐 Size: {width}×{height} = {total} quadrants")
    print(f"🎨 Water color: #{WATER_COLOR[0]:02X}{WATER_COLOR[1]:02X}{WATER_COLOR[2]:02X}")
    print()

    if args.dry_run:
        print("🔍 Dry run - no changes will be made")
        print(f"   Would fill {total} quadrants with water color")
        return 0

    # Fill quadrants
    conn = sqlite3.connect(db_path)
    try:
        # Get generation config (needed for ensure_quadrant_exists)
        config = get_generation_config(conn)

        # Determine quadrant size from config
        quadrant_width = config.get("width_px", QUADRANT_SIZE) // 2
        quadrant_height = config.get("height_px", QUADRANT_SIZE) // 2

        # Create water image
        water_img = create_water_image(quadrant_width, quadrant_height)
        water_png = image_to_png_bytes(water_img)
        print(f"💧 Created water tile: {quadrant_width}×{quadrant_height}, {len(water_png)} bytes")
        print()

        filled = 0
        skipped = 0

        for qy in range(tl[1], br[1] + 1):
            for qx in range(tl[0], br[0] + 1):
                # Check if quadrant already has data
                if not args.overwrite and quadrant_has_data(conn, qx, qy):
                    skipped += 1
                    continue

                # Save water tile (this handles creating quadrant with proper lat/lng)
                save_quadrant_generation(conn, config, qx, qy, water_png)
                filled += 1

            # Progress indicator
            row_num = qy - tl[1] + 1
            print(f"   Row {row_num}/{height}: filled {filled}, skipped {skipped}")

    finally:
        conn.close()

    print()
    print("=" * 50)
    print("✅ Fill complete!")
    print(f"   Filled: {filled} quadrants")
    print(f"   Skipped: {skipped} quadrants (already had data)")
    print(f"   Total: {total} quadrants")

    return 0


if __name__ == "__main__":
    sys.exit(main())

