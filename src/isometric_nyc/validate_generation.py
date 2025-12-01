import argparse
from pathlib import Path

from isometric_nyc.validate_plan import stitch_images


def main():
  parser = argparse.ArgumentParser(
    description="Validate generation by stitching generation.png images."
  )
  parser.add_argument(
    "tile_dir", type=Path, help="Directory containing tile subdirectories"
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    return

  print(f"🔍 Validating generation in {args.tile_dir}...")

  # Stitch generations
  stitch_images(args.tile_dir, "generation.png", "full_generation.png")


if __name__ == "__main__":
  main()
