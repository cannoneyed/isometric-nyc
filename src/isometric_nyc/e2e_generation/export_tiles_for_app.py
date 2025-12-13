"""
Export quadrants from the generation database to the web app's tile directory.

Exports a rectangular region of quadrants to the web app's public/tiles/0/ directory,
normalizing coordinates so the top-left of the region becomes (0, 0).

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py <generation_dir> [--tl X,Y --br X,Y]

Examples:
  # Export ALL quadrants in the database (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py generations/v01

  # Export quadrants from (0,0) to (19,19) - a 20x20 grid
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py generations/v01 --tl 0,0 --br 19,19

  # Export a smaller 5x5 region starting at (10,10)
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py generations/v01 --tl 10,10 --br 14,14

  # Use render images instead of generations
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py generations/v01 --tl 0,0 --br 9,9 --render

  # Specify custom output directory
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py generations/v01 --tl 0,0 --br 9,9 --output-dir ./my-tiles
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


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


def get_quadrant_data(
    db_path: Path, x: int, y: int, use_render: bool = False
) -> bytes | None:
    """
    Get the image bytes for a quadrant at position (x, y).

    Args:
        db_path: Path to the quadrants.db file.
        x: X coordinate of the quadrant.
        y: Y coordinate of the quadrant.
        use_render: If True, get render bytes; otherwise get generation bytes.

    Returns:
        PNG bytes or None if not found.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        column = "render" if use_render else "generation"
        cursor.execute(
            f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
            (x, y),
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
    """
    Get the bounding box of all quadrants in the database.

    Returns:
        (min_x, min_y, max_x, max_y) or None if no quadrants exist.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
            FROM quadrants
            """
        )
        row = cursor.fetchone()
        if row and row[0] is not None:
            return row[0], row[1], row[2], row[3]
        return None
    finally:
        conn.close()


def count_generated_quadrants(
    db_path: Path, tl: tuple[int, int], br: tuple[int, int], use_render: bool = False
) -> tuple[int, int]:
    """
    Count total and generated quadrants in the specified range.

    Returns:
        (total_in_range, with_data_count)
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        column = "render" if use_render else "generation"
        
        # Count quadrants with data in range
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM quadrants
            WHERE quadrant_x >= ? AND quadrant_x <= ?
              AND quadrant_y >= ? AND quadrant_y <= ?
              AND {column} IS NOT NULL
            """,
            (tl[0], br[0], tl[1], br[1]),
        )
        with_data = cursor.fetchone()[0]
        
        total = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
        return total, with_data
    finally:
        conn.close()


def export_tiles(
    db_path: Path,
    tl: tuple[int, int],
    br: tuple[int, int],
    output_dir: Path,
    use_render: bool = False,
    skip_existing: bool = True,
) -> tuple[int, int, int]:
    """
    Export quadrants from the database to the output directory.

    Coordinates are normalized so that tl becomes (0, 0) in the output.

    Args:
        db_path: Path to the quadrants.db file.
        tl: Top-left coordinate (x, y) of the region to export.
        br: Bottom-right coordinate (x, y) of the region to export.
        output_dir: Directory to save tiles (e.g., public/tiles/0/).
        use_render: If True, export render images; otherwise export generations.
        skip_existing: If True, skip tiles that already exist.

    Returns:
        Tuple of (exported_count, skipped_count, missing_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_type = "render" if use_render else "generation"
    exported = 0
    skipped = 0
    missing = 0
    
    # Calculate grid dimensions
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1
    total = width * height
    
    print(f"📦 Exporting {width}×{height} = {total} tiles ({data_type})")
    print(f"   Source range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    print(f"   Output range: (0,0) to ({width-1},{height-1})")
    print(f"   Output dir: {output_dir}")
    print()
    
    # Iterate through all quadrants in the range
    for src_y in range(tl[1], br[1] + 1):
        row_exported = 0
        row_missing = 0
        
        for src_x in range(tl[0], br[0] + 1):
            # Normalize coordinates (tl becomes 0,0)
            dst_x = src_x - tl[0]
            dst_y = src_y - tl[1]
            
            output_path = output_dir / f"{dst_x}_{dst_y}.png"
            
            # Check if output already exists
            if skip_existing and output_path.exists():
                skipped += 1
                continue
            
            # Get quadrant data from database
            data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)
            
            if data is None:
                missing += 1
                row_missing += 1
                continue
            
            # Save to output file
            output_path.write_bytes(data)
            exported += 1
            row_exported += 1
        
        # Print row progress
        progress = (src_y - tl[1] + 1) / height * 100
        status_parts = [f"Row {src_y - tl[1]:3d}: {row_exported:3d} exported"]
        if row_missing > 0:
            status_parts.append(f"{row_missing} missing")
        print(f"   [{progress:5.1f}%] {', '.join(status_parts)}")
    
    return exported, skipped, missing


def write_manifest(
    output_dir: Path,
    width: int,
    height: int,
    tile_size: int = 512,
) -> None:
    """
    Write a manifest.json file with grid configuration.

    Args:
        output_dir: Directory containing tiles (e.g., public/tiles/0/).
        width: Grid width in tiles.
        height: Grid height in tiles.
        tile_size: Size of each tile in pixels.
    """
    # Write manifest to parent directory (tiles/ not tiles/0/)
    manifest_path = output_dir.parent / "manifest.json"
    
    manifest = {
        "gridWidth": width,
        "gridHeight": height,
        "tileSize": tile_size,
        "totalTiles": width * height,
        "generated": datetime.now(timezone.utc).isoformat(),
        "urlPattern": "{z}/{x}_{y}.png",
    }
    
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"📝 Wrote manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export quadrants from the generation database to the web app's tile directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export ALL tiles in the database (auto-detect bounds)
  %(prog)s generations/v01

  # Export a 20x20 grid
  %(prog)s generations/v01 --tl 0,0 --br 19,19

  # Export with custom output directory
  %(prog)s generations/v01 --tl 0,0 --br 9,9 --output-dir ./custom/tiles/0
        """,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--tl",
        type=parse_coordinate,
        required=False,
        default=None,
        metavar="X,Y",
        help="Top-left coordinate of the region to export (e.g., '0,0'). If omitted, auto-detects from database.",
    )
    parser.add_argument(
        "--br",
        type=parse_coordinate,
        required=False,
        default=None,
        metavar="X,Y",
        help="Bottom-right coordinate of the region to export (e.g., '19,19'). If omitted, auto-detects from database.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Export render images instead of generation images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for tiles (default: src/app/public/tiles/0/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tiles (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without actually exporting",
    )

    args = parser.parse_args()

    # Resolve paths
    generation_dir = args.generation_dir.resolve()
    
    # Default output directory is src/app/public/tiles/0/ relative to project root
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        # Find project root (look for src/app directory)
        project_root = generation_dir.parent
        while project_root != project_root.parent:
            if (project_root / "src" / "app").exists():
                break
            project_root = project_root.parent
        output_dir = project_root / "src" / "app" / "public" / "tiles" / "0"

    # Validate inputs
    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Get database bounds
    bounds = get_quadrant_bounds(db_path)
    if not bounds:
        print("❌ Error: No quadrants found in database")
        return 1
    
    print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

    # Use provided coordinates or auto-detect from database bounds
    if args.tl is None or args.br is None:
        if args.tl is not None or args.br is not None:
            print("❌ Error: Both --tl and --br must be provided together, or neither for auto-detect")
            return 1
        tl = (bounds[0], bounds[1])
        br = (bounds[2], bounds[3])
        print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    else:
        tl = args.tl
        br = args.br

    # Validate coordinate range
    if tl[0] > br[0] or tl[1] > br[1]:
        print(f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})")
        return 1
    
    # Count available data
    total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
    data_type = "render" if args.render else "generation"
    print(f"   Available {data_type} data: {available}/{total} quadrants")
    print()

    if args.dry_run:
        print("🔍 Dry run - no files will be written")
        width = br[0] - tl[0] + 1
        height = br[1] - tl[1] + 1
        print(f"   Would export: {width}×{height} = {total} tiles")
        print(f"   To: {output_dir}")
        return 0

    # Export tiles
    exported, skipped, missing = export_tiles(
        db_path,
        tl,
        br,
        output_dir,
        use_render=args.render,
        skip_existing=not args.overwrite,
    )

    # Calculate final dimensions
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1

    # Write manifest with grid configuration
    write_manifest(output_dir, width, height)

    # Print summary
    print()
    print("=" * 50)
    print(f"✅ Export complete!")
    print(f"   Exported: {exported} tiles")
    if skipped > 0:
        print(f"   Skipped (existing): {skipped} tiles")
    if missing > 0:
        print(f"   Missing data: {missing} tiles")
    print(f"   Output: {output_dir}")
    print(f"   Grid size: {width}×{height}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

