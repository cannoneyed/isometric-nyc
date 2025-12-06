"""
Copy infill dataset files following a balanced distribution strategy.

Recipe 2 Distribution:
- First half of tiles: L/R halves (g_r_g_r, r_g_r_g) + 1 quadrant
- Second half of tiles: T/B halves (g_g_r_r, r_r_g_g) + 1 quadrant
- Quadrants cycle across all tiles: TL, TR, BL, BR, TL, TR, ...

Usage:
  uv run python src/isometric_nyc/synthetic_data/copy_infill_dataset.py \
    --source synthetic_data/tiles/v04 \
    --dest synthetic_data/datasets/v04/infills
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Variant definitions
LEFT_RIGHT_HALVES = ["infill_g_r_g_r.png", "infill_r_g_r_g.png"]
TOP_BOTTOM_HALVES = ["infill_g_g_r_r.png", "infill_r_r_g_g.png"]
QUADRANTS = [
    "infill_r_g_g_g.png",  # TL
    "infill_g_r_g_g.png",  # TR
    "infill_g_g_r_g.png",  # BL
    "infill_g_g_g_r.png",  # BR
]

# Mapping from variant filename to region description
VARIANT_TO_REGION: Dict[str, str] = {
    "infill_g_r_g_r.png": "right half",
    "infill_r_g_r_g.png": "left half",
    "infill_g_g_r_r.png": "bottom half",
    "infill_r_r_g_g.png": "top half",
    "infill_r_g_g_g.png": "top left quadrant",
    "infill_g_r_g_g.png": "top right quadrant",
    "infill_g_g_r_g.png": "bottom left quadrant",
    "infill_g_g_g_r.png": "bottom right quadrant",
}


def get_variants_for_tile(tile_index: int, total_tiles: int) -> List[str]:
    """
    Determine which infill variants to use for a given tile.

    Recipe 2:
    - First half of tiles: L/R halves
    - Second half of tiles: T/B halves
    - All tiles: 1 quadrant (cycling through TL, TR, BL, BR)

    Args:
        tile_index: 0-based index of the tile in sorted order.
        total_tiles: Total number of tiles.

    Returns:
        List of infill filenames to copy for this tile.
    """
    variants = []

    # Half variants based on tile position
    midpoint = total_tiles // 2
    if tile_index < midpoint:
        variants.extend(LEFT_RIGHT_HALVES)
    else:
        variants.extend(TOP_BOTTOM_HALVES)

    # Quadrant variant (cycles through all 4)
    quadrant_idx = tile_index % 4
    variants.append(QUADRANTS[quadrant_idx])

    return variants


def copy_infill_files(
    source_dir: Path,
    dest_dir: Path,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Copy infill files from tile subdirs to destination with prefixed names.

    Args:
        source_dir: Parent directory containing tile subdirs (e.g., v04/).
        dest_dir: Destination directory for copied files.
        dry_run: If True, print what would be done without copying.

    Returns:
        Tuple of (files_copied, files_skipped).
    """
    # Find all tile subdirectories
    tile_dirs = sorted(
        [d for d in source_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    )

    if not tile_dirs:
        print(f"❌ No tile directories found in {source_dir}")
        return 0, 0

    print(f"📂 Found {len(tile_dirs)} tile directories")

    # Create destination directory
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    files_skipped = 0

    for tile_index, tile_dir in enumerate(tile_dirs):
        tile_name = tile_dir.name
        variants = get_variants_for_tile(tile_index, len(tile_dirs))

        print(f"\n📁 {tile_name} (index {tile_index}):")
        print(f"   Variants: {', '.join(v.replace('infill_', '').replace('.png', '') for v in variants)}")

        for variant in variants:
            src_file = tile_dir / variant
            if not src_file.exists():
                print(f"   ⚠️  Missing: {variant}")
                files_skipped += 1
                continue

            # Create prefixed filename
            dest_filename = f"{tile_name}_{variant}"
            dest_file = dest_dir / dest_filename

            if dry_run:
                print(f"   Would copy: {variant} → {dest_filename}")
            else:
                shutil.copy2(src_file, dest_file)
                print(f"   ✅ Copied: {variant} → {dest_filename}")

            files_copied += 1

    return files_copied, files_skipped


def create_infills_csv(
    dest_dir: Path,
    infills_folder_name: str = "infills_v04",
    generations_folder_name: str = "generations",
    dry_run: bool = False,
) -> int:
    """
    Create a CSV file mapping infill images to generations with prompts.

    Args:
        dest_dir: Directory containing the copied infill files.
        infills_folder_name: Name to use for infills folder in paths.
        generations_folder_name: Name to use for generations folder in paths.
        dry_run: If True, print what would be done without creating file.

    Returns:
        Number of rows written to CSV.
    """
    # Find all infill files in the destination directory
    infill_files = sorted(dest_dir.glob("*_infill_*.png"))

    if not infill_files:
        print(f"❌ No infill files found in {dest_dir}")
        return 0

    csv_path = dest_dir.parent / "infills.csv"
    rows = []

    for infill_file in infill_files:
        filename = infill_file.name
        # Extract tile prefix (e.g., "000" from "000_infill_g_r_g_r.png")
        tile_prefix = filename.split("_")[0]
        # Extract variant name (e.g., "infill_g_r_g_r.png")
        variant = "_".join(filename.split("_")[1:])

        # Get region description
        region = VARIANT_TO_REGION.get(variant, "region")

        # Build row
        template = f"{infills_folder_name}/{filename}"
        generation = f"{generations_folder_name}/{tile_prefix}.png"
        prompt = (
            f"Convert the {region} of the image to isometric nyc pixel art "
            f"in precisely the style of the other part of the image."
        )

        rows.append({
            "template": template,
            "generation": generation,
            "prompt": prompt,
        })

    if dry_run:
        print(f"\n📄 Would create CSV: {csv_path}")
        print(f"   Rows: {len(rows)}")
        for row in rows[:3]:
            print(f"   Example: {row['template']} → {row['generation']}")
        if len(rows) > 3:
            print(f"   ... and {len(rows) - 3} more rows")
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["template", "generation", "prompt"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n📄 Created CSV: {csv_path}")
        print(f"   Rows written: {len(rows)}")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Copy infill dataset files with balanced distribution"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("synthetic_data/tiles/v04"),
        help="Source directory containing tile subdirs",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("synthetic_data/datasets/v04/infills"),
        help="Destination directory for copied files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying files",
    )

    args = parser.parse_args()

    if not args.source.exists():
        print(f"❌ Source directory not found: {args.source}")
        sys.exit(1)

    print("=" * 60)
    print("📋 INFILL DATASET COPY - Recipe 2 Distribution")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Destination: {args.dest}")
    print()
    print("Distribution strategy:")
    print("  • First half of tiles → L/R halves (g_r_g_r, r_g_r_g)")
    print("  • Second half of tiles → T/B halves (g_g_r_r, r_r_g_g)")
    print("  • All tiles → 1 quadrant (cycling TL, TR, BL, BR)")

    if args.dry_run:
        print("\n🔍 DRY RUN - No files will be copied\n")

    copied, skipped = copy_infill_files(args.source, args.dest, args.dry_run)

    # Create CSV mapping file
    csv_rows = create_infills_csv(args.dest, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print(f"✨ COMPLETE")
    print(f"   Files copied: {copied}")
    print(f"   Files skipped: {skipped}")
    print(f"   CSV rows: {csv_rows}")
    print("=" * 60)


if __name__ == "__main__":
    main()

