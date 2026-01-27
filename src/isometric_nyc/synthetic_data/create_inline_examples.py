"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(
  tile_dir: Path, no_output: bool = False, infill: bool = False
) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  no_output (infill=False):
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]

  no_output (infill=True):
    [Whitebox] [Render]
    [Left-Half-Generation] [Empty]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if infill or not no_output:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if infill or not no_output:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not no_output and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and not no_output and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--no-output",
    action="store_true",
    help="Don't show the full generation output (affects quadrants based on infill flag)",
  )
  parser.add_argument(
    "--infill",
    action="store_true",
    help="Show left-half generation in bottom-left and full generation in bottom-right",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d, no_output=args.no_output, infill=args.infill)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()
