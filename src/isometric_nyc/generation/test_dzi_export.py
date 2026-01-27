"""
Test script to validate DZI generation from tile database using libvips.

This tests whether we can:
1. Extract tiles from the SQLite database
2. Assemble them into a composite image using pyvips
3. Generate a DZI pyramid using dzsave()

Run with:
  DYLD_LIBRARY_PATH=/opt/homebrew/lib uv run python src/isometric_nyc/generation/test_dzi_export.py
"""

import io
import os
import sqlite3
import tempfile
import time
from pathlib import Path

# Set library path for pyvips before import
os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")

import pyvips
from PIL import Image

# Constants
TILE_SIZE = 512


def get_tile_subset(
    db_path: Path,
    center_x: int,
    center_y: int,
    radius: int = 5,
) -> dict[tuple[int, int], bytes]:
    """
    Get a small subset of tiles around a center point.

    Args:
        db_path: Path to quadrants.db
        center_x: Center quadrant x coordinate
        center_y: Center quadrant y coordinate
        radius: Number of tiles in each direction from center

    Returns:
        Dict mapping (x, y) to raw image bytes
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        min_x = center_x - radius
        max_x = center_x + radius
        min_y = center_y - radius
        max_y = center_y + radius

        cursor.execute(
            """
            SELECT quadrant_x, quadrant_y, generation
            FROM quadrants
            WHERE quadrant_x >= ? AND quadrant_x <= ?
              AND quadrant_y >= ? AND quadrant_y <= ?
              AND generation IS NOT NULL
            """,
            (min_x, max_x, min_y, max_y),
        )

        tiles = {}
        for row in cursor.fetchall():
            x, y, data = row
            if data:
                tiles[(x, y)] = data

        return tiles
    finally:
        conn.close()


def get_database_center(db_path: Path) -> tuple[int, int]:
    """Get the center point of tiles in the database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MIN(quadrant_x), MAX(quadrant_x), MIN(quadrant_y), MAX(quadrant_y)
            FROM quadrants
            WHERE generation IS NOT NULL
            """
        )
        min_x, max_x, min_y, max_y = cursor.fetchone()
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        return center_x, center_y
    finally:
        conn.close()


def assemble_tiles_pyvips(
    tiles: dict[tuple[int, int], bytes],
    tile_size: int = TILE_SIZE,
) -> pyvips.Image:
    """
    Assemble tiles into a single image using pyvips.

    This uses pyvips's arrayjoin to efficiently combine tiles
    without loading all images into memory simultaneously.
    """
    if not tiles:
        raise ValueError("No tiles provided")

    # Get bounds
    xs = [coord[0] for coord in tiles.keys()]
    ys = [coord[1] for coord in tiles.keys()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    print(f"  Assembling {len(tiles)} tiles into {width}x{height} grid...")
    print(f"  Bounds: ({min_x},{min_y}) to ({max_x},{max_y})")
    print(f"  Output size: {width * tile_size} x {height * tile_size} pixels")

    # Create a black tile for missing positions
    black_tile = pyvips.Image.black(tile_size, tile_size, bands=3)

    # Build rows of tiles
    rows = []
    for y in range(min_y, max_y + 1):
        row_tiles = []
        for x in range(min_x, max_x + 1):
            if (x, y) in tiles:
                # Load tile from bytes using pyvips
                tile_data = tiles[(x, y)]
                tile_img = pyvips.Image.new_from_buffer(tile_data, "")
                # Ensure RGB (drop alpha if present)
                if tile_img.bands == 4:
                    tile_img = tile_img.extract_band(0, n=3)
                elif tile_img.bands == 1:
                    tile_img = tile_img.bandjoin([tile_img, tile_img])
                row_tiles.append(tile_img)
            else:
                row_tiles.append(black_tile)

        # Join row horizontally
        row_img = pyvips.Image.arrayjoin(row_tiles, across=len(row_tiles))
        rows.append(row_img)

    # Join all rows vertically
    result = pyvips.Image.arrayjoin(rows, across=1)

    return result


def generate_dzi(
    image: pyvips.Image,
    output_path: str,
    tile_size: int = 512,
    overlap: int = 0,
    suffix: str = ".png",
) -> dict:
    """
    Generate DZI pyramid from a pyvips image.

    Args:
        image: pyvips.Image to convert
        output_path: Base path for output (without extension)
        tile_size: Size of tiles in the pyramid
        overlap: Pixel overlap between tiles
        suffix: File extension for tiles

    Returns:
        Dict with generation stats
    """
    start_time = time.time()

    # Generate DZI
    image.dzsave(
        output_path,
        tile_size=tile_size,
        overlap=overlap,
        suffix=suffix,
        depth="onetile",  # Generate pyramid until fits in one tile
        background=[0, 0, 0],  # Black background for edge tiles
    )

    elapsed = time.time() - start_time

    # Count generated files
    dzi_file = Path(f"{output_path}.dzi")
    files_dir = Path(f"{output_path}_files")

    tile_count = 0
    level_count = 0
    if files_dir.exists():
        for level_dir in files_dir.iterdir():
            if level_dir.is_dir():
                level_count += 1
                tile_count += len(list(level_dir.glob(f"*{suffix}")))

    return {
        "elapsed_seconds": elapsed,
        "dzi_file": str(dzi_file),
        "files_dir": str(files_dir),
        "tile_count": tile_count,
        "level_count": level_count,
    }


def main():
    """Run DZI generation test with a small tile subset."""
    print("=" * 60)
    print("DZI Export Test - Validation with Small Tile Subset")
    print("=" * 60)
    print()

    # Find the database
    db_path = Path("generations/nyc/quadrants.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    print(f"📊 Database: {db_path}")
    print()

    # Get center of the tile grid
    center_x, center_y = get_database_center(db_path)
    print(f"📍 Grid center: ({center_x}, {center_y})")

    # Test with different subset sizes
    test_sizes = [3, 5, 10]

    for radius in test_sizes:
        print()
        print(f"{'='*60}")
        print(f"Test: {2*radius+1}x{2*radius+1} tile subset (radius={radius})")
        print(f"{'='*60}")

        # Extract tiles
        print("\n1. Extracting tiles from database...")
        extract_start = time.time()
        tiles = get_tile_subset(db_path, center_x, center_y, radius=radius)
        extract_time = time.time() - extract_start
        print(f"   Found {len(tiles)} tiles in {extract_time:.2f}s")

        if not tiles:
            print("   ❌ No tiles found, skipping...")
            continue

        # Assemble tiles
        print("\n2. Assembling tiles with pyvips...")
        assemble_start = time.time()
        try:
            composite = assemble_tiles_pyvips(tiles)
            assemble_time = time.time() - assemble_start
            print(f"   ✅ Assembled in {assemble_time:.2f}s")
            print(f"   Image size: {composite.width} x {composite.height} pixels")
        except Exception as e:
            print(f"   ❌ Assembly failed: {e}")
            continue

        # Generate DZI
        print("\n3. Generating DZI pyramid...")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "test_tiles"

            try:
                stats = generate_dzi(composite, str(output_base), tile_size=512)
                print(f"   ✅ Generated in {stats['elapsed_seconds']:.2f}s")
                print(f"   DZI file: {stats['dzi_file']}")
                print(f"   Tiles directory: {stats['files_dir']}")
                print(f"   Total tiles: {stats['tile_count']}")
                print(f"   Zoom levels: {stats['level_count']}")

                # Read and display DZI file contents
                dzi_file = Path(stats["dzi_file"])
                if dzi_file.exists():
                    print(f"\n   DZI descriptor content:")
                    print("   " + "-" * 40)
                    with open(dzi_file) as f:
                        for line in f:
                            print(f"   {line.rstrip()}")

            except Exception as e:
                print(f"   ❌ DZI generation failed: {e}")
                import traceback

                traceback.print_exc()
                continue

        print()
        total_time = extract_time + assemble_time + stats["elapsed_seconds"]
        print(f"📊 Total time for {2*radius+1}x{2*radius+1} grid: {total_time:.2f}s")

    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
