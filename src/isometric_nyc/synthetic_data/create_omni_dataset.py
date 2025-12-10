"""
Create omni generation dataset combining all generation strategies.

This script generates training examples combining:
1. Full generation (20%) - entire tile is render pixels
2. Quadrant generation (20%) - one quarter render, rest generation
3. Half generation (20%) - one half render, one half generation
4. Middle generation (15%) - middle strip (vertical/horizontal) is render
5. Rectangle strips (10%) - full horizontal/vertical strip of render (25-60%)
6. Rectangle infills (15%) - rectangle of render (25-60% area) anywhere

All render pixels are outlined with a 1px solid red border.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_omni_dataset.py
  uv run python src/isometric_nyc/synthetic_data/create_omni_dataset.py --dry-run
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

# Image dimensions
TARGET_SIZE = (1024, 1024)

# Red outline settings
OUTLINE_COLOR = (255, 0, 0)
OUTLINE_WIDTH = 2

# Generation type constants
TYPE_FULL = "full"
TYPE_QUADRANT_TL = "quadrant_tl"
TYPE_QUADRANT_TR = "quadrant_tr"
TYPE_QUADRANT_BL = "quadrant_bl"
TYPE_QUADRANT_BR = "quadrant_br"
TYPE_HALF_LEFT = "half_left"
TYPE_HALF_RIGHT = "half_right"
TYPE_HALF_TOP = "half_top"
TYPE_HALF_BOTTOM = "half_bottom"
TYPE_MIDDLE_VERTICAL = "middle_vertical"
TYPE_MIDDLE_HORIZONTAL = "middle_horizontal"
TYPE_STRIP_VERTICAL = "strip_vertical"
TYPE_STRIP_HORIZONTAL = "strip_horizontal"
TYPE_RECT_INFILL = "rect_infill"

# Grouped by category for distribution
FULL_TYPES = [TYPE_FULL]
QUADRANT_TYPES = [
  TYPE_QUADRANT_TL,
  TYPE_QUADRANT_TR,
  TYPE_QUADRANT_BL,
  TYPE_QUADRANT_BR,
]
HALF_TYPES = [TYPE_HALF_LEFT, TYPE_HALF_RIGHT, TYPE_HALF_TOP, TYPE_HALF_BOTTOM]
MIDDLE_TYPES = [TYPE_MIDDLE_VERTICAL, TYPE_MIDDLE_HORIZONTAL]
STRIP_TYPES = [TYPE_STRIP_VERTICAL, TYPE_STRIP_HORIZONTAL]
INFILL_TYPES = [TYPE_RECT_INFILL]

# Distribution weights (must sum to 1.0)
DISTRIBUTION = {
  "full": 0.20,
  "quadrant": 0.18,
  "half": 0.17,
  "middle": 0.15,
  "strip": 0.10,
  "infill": 0.20,
}

# Alphabetical suffixes for variants
VARIANT_SUFFIXES = [
  "a",
  "b",
  "c",
  "d",
  "e",
  "f",
  "g",
  "h",
  "i",
  "j",
  "k",
  "l",
  "m",
  "n",
  "o",
  "p",
  "q",
  "r",
  "s",
  "t",
]

# Prompt template
PROMPT = (
  "Fill in the outlined section with the missing pixels corresponding to the "
  "<isometric nyc pixel art> style, removing the border and exactly following "
  "the shape/style/structure of the surrounding image (if present)."
)


def draw_outline(
  img: Image.Image,
  box: Tuple[int, int, int, int],
  color: Tuple[int, int, int] = OUTLINE_COLOR,
  width: int = OUTLINE_WIDTH,
) -> None:
  """
  Draw a solid red outline around a rectangular region.

  The outline is drawn ON TOP of the image (no pixel displacement).
  """
  draw = ImageDraw.Draw(img)
  x1, y1, x2, y2 = box

  for i in range(width):
    draw.rectangle(
      [x1 + i, y1 + i, x2 - 1 - i, y2 - 1 - i],
      outline=color,
      fill=None,
    )


def draw_multi_region_outline(
  img: Image.Image,
  boxes: List[Tuple[int, int, int, int]],
  color: Tuple[int, int, int] = OUTLINE_COLOR,
  width: int = OUTLINE_WIDTH,
) -> None:
  """Draw outlines around multiple rectangular regions."""
  for box in boxes:
    draw_outline(img, box, color, width)


def get_render_regions(
  gen_type: str, width: int, height: int, rng: random.Random
) -> List[Tuple[int, int, int, int]]:
  """
  Get the regions that should contain render pixels for a given generation type.

  Returns a list of (x1, y1, x2, y2) boxes.
  """
  half_w = width // 2
  half_h = height // 2

  if gen_type == TYPE_FULL:
    return [(0, 0, width, height)]

  # Quadrant types
  elif gen_type == TYPE_QUADRANT_TL:
    return [(0, 0, half_w, half_h)]
  elif gen_type == TYPE_QUADRANT_TR:
    return [(half_w, 0, width, half_h)]
  elif gen_type == TYPE_QUADRANT_BL:
    return [(0, half_h, half_w, height)]
  elif gen_type == TYPE_QUADRANT_BR:
    return [(half_w, half_h, width, height)]

  # Half types
  elif gen_type == TYPE_HALF_LEFT:
    return [(0, 0, half_w, height)]
  elif gen_type == TYPE_HALF_RIGHT:
    return [(half_w, 0, width, height)]
  elif gen_type == TYPE_HALF_TOP:
    return [(0, 0, width, half_h)]
  elif gen_type == TYPE_HALF_BOTTOM:
    return [(0, half_h, width, height)]

  # Middle types (centered strip, 50% of dimension)
  elif gen_type == TYPE_MIDDLE_VERTICAL:
    strip_width = width // 2
    x1 = (width - strip_width) // 2
    return [(x1, 0, x1 + strip_width, height)]
  elif gen_type == TYPE_MIDDLE_HORIZONTAL:
    strip_height = height // 2
    y1 = (height - strip_height) // 2
    return [(0, y1, width, y1 + strip_height)]

  # Strip types (25-60% of dimension, anywhere in image)
  elif gen_type == TYPE_STRIP_VERTICAL:
    min_pct, max_pct = 0.25, 0.60
    strip_pct = rng.uniform(min_pct, max_pct)
    strip_width = int(width * strip_pct)
    x1 = rng.randint(0, width - strip_width)
    return [(x1, 0, x1 + strip_width, height)]
  elif gen_type == TYPE_STRIP_HORIZONTAL:
    min_pct, max_pct = 0.25, 0.60
    strip_pct = rng.uniform(min_pct, max_pct)
    strip_height = int(height * strip_pct)
    y1 = rng.randint(0, height - strip_height)
    return [(0, y1, width, y1 + strip_height)]

  # Rectangle infill (25-60% area, anywhere)
  elif gen_type == TYPE_RECT_INFILL:
    total_area = width * height
    min_area = int(total_area * 0.25)
    max_area = int(total_area * 0.60)
    target_area = rng.randint(min_area, max_area)

    # Random aspect ratio between 0.5 and 2.0
    aspect = rng.uniform(0.5, 2.0)

    # Calculate dimensions
    rect_height = int((target_area / aspect) ** 0.5)
    rect_width = int(rect_height * aspect)

    # Clamp to image bounds
    rect_width = min(rect_width, width - 32)
    rect_height = min(rect_height, height - 32)
    rect_width = max(rect_width, width // 4)
    rect_height = max(rect_height, height // 4)

    # Random position
    margin = 16
    x1 = rng.randint(margin, max(margin, width - rect_width - margin))
    y1 = rng.randint(margin, max(margin, height - rect_height - margin))

    return [(x1, y1, x1 + rect_width, y1 + rect_height)]

  else:
    raise ValueError(f"Unknown generation type: {gen_type}")


def create_omni_image(
  img_gen: Image.Image,
  img_render: Image.Image,
  gen_type: str,
  rng: random.Random,
) -> Image.Image:
  """
  Create an omni training image.

  Takes the generation image and replaces specified regions with render pixels,
  then draws red outlines around the render regions.
  """
  width, height = TARGET_SIZE

  # Resize images if needed
  if img_gen.size != TARGET_SIZE:
    img_gen = img_gen.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
  if img_render.size != TARGET_SIZE:
    img_render = img_render.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

  # Get render regions
  regions = get_render_regions(gen_type, width, height, rng)

  # Start with generation image (or empty for full render)
  if gen_type == TYPE_FULL:
    final_image = img_render.copy()
    # Ensure we draw the border for full render
    # get_render_regions returns the full box, so draw_multi_region_outline will handle it
  else:
    final_image = img_gen.copy()
    # Paste render regions
    for x1, y1, x2, y2 in regions:
      render_crop = img_render.crop((x1, y1, x2, y2))
      final_image.paste(render_crop, (x1, y1))

  # Draw outlines
  draw_multi_region_outline(final_image, regions, width=OUTLINE_WIDTH)

  return final_image


def get_image_pairs(dataset_dir: Path) -> List[Tuple[str, Path, Path]]:
  """
  Get all matching generation/render image pairs.

  Returns:
      List of (image_number, generation_path, render_path) tuples.
  """
  generations_dir = dataset_dir / "generations"
  renders_dir = dataset_dir / "renders"

  pairs = []

  for gen_path in sorted(generations_dir.glob("*.png")):
    image_num = gen_path.stem
    render_path = renders_dir / f"{image_num}.png"

    if render_path.exists():
      pairs.append((image_num, gen_path, render_path))
    else:
      print(f"⚠️  No matching render for generation {image_num}")

  return pairs


def is_valid_assignment(current_types: List[str], new_type: str) -> bool:
  """Check if adding new_type to current_types satisfies constraints."""
  # Constraint 1: Max 1 "full"
  if new_type == TYPE_FULL and TYPE_FULL in current_types:
    return False

  # Constraint 2: Max 1 "middle" (any type)
  if new_type in MIDDLE_TYPES:
    for t in current_types:
      if t in MIDDLE_TYPES:
        return False

  # Constraint 3: Max 1 of SAME "half" type
  if new_type in HALF_TYPES and new_type in current_types:
    return False

  return True


def assign_generation_types(
  num_images: int, variants_per_image: int, rng: random.Random
) -> List[List[str]]:
  """
  Assign generation types to achieve the target distribution while respecting constraints.

  Constraints per image:
  1. Max 1 "full"
  2. Max 1 "middle" category variant
  3. Max 1 of the same "half" type
  """
  total_variants = num_images * variants_per_image

  # Calculate how many of each category
  category_counts = {
    cat: int(total_variants * weight) for cat, weight in DISTRIBUTION.items()
  }

  # Adjust for rounding errors - add to "infill" which is least constrained
  diff = total_variants - sum(category_counts.values())
  if diff > 0:
    category_counts["infill"] += diff

  # Create pool of generation types
  type_pool: List[str] = []

  # Full
  type_pool.extend([TYPE_FULL] * category_counts["full"])

  # Quadrant - distribute evenly among 4 quadrants
  quadrant_each = category_counts["quadrant"] // 4
  for qt in QUADRANT_TYPES:
    type_pool.extend([qt] * quadrant_each)
  # Add remainder
  remainder = category_counts["quadrant"] - (quadrant_each * 4)
  for i in range(remainder):
    type_pool.append(QUADRANT_TYPES[i % 4])

  # Half - distribute evenly among 4 halves
  half_each = category_counts["half"] // 4
  for ht in HALF_TYPES:
    type_pool.extend([ht] * half_each)
  remainder = category_counts["half"] - (half_each * 4)
  for i in range(remainder):
    type_pool.append(HALF_TYPES[i % 4])

  # Middle - distribute evenly
  middle_each = category_counts["middle"] // 2
  for mt in MIDDLE_TYPES:
    type_pool.extend([mt] * middle_each)
  remainder = category_counts["middle"] - (middle_each * 2)
  for i in range(remainder):
    type_pool.append(MIDDLE_TYPES[i % 2])

  # Strip - distribute evenly
  strip_each = category_counts["strip"] // 2
  for st in STRIP_TYPES:
    type_pool.extend([st] * strip_each)
  remainder = category_counts["strip"] - (strip_each * 2)
  for i in range(remainder):
    type_pool.append(STRIP_TYPES[i % 2])

  # Infill
  type_pool.extend([TYPE_RECT_INFILL] * category_counts["infill"])

  # Sort pool to prioritize constrained types
  # Priority: full > middle > half > others
  def get_priority(t: str) -> int:
    if t == TYPE_FULL:
      return 0
    if t in MIDDLE_TYPES:
      return 1
    if t in HALF_TYPES:
      return 2
    return 3

  type_pool.sort(key=get_priority)

  # Distribute to images
  assignments: List[List[str]] = [[] for _ in range(num_images)]

  # We use a round-robin approach with randomized start to distribute evenly
  # but fill constraints first
  image_indices = list(range(num_images))
  rng.shuffle(image_indices)

  # For each type in the pool, try to find a valid slot
  unassigned = []

  for gen_type in type_pool:
    assigned = False
    # Try to find a bucket that isn't full and satisfies constraints
    # Start checking from a random index to avoid bias
    start_idx = rng.randint(0, num_images - 1)

    for i in range(num_images):
      idx = (start_idx + i) % num_images
      image_idx = image_indices[idx]

      if len(assignments[image_idx]) < variants_per_image:
        if is_valid_assignment(assignments[image_idx], gen_type):
          assignments[image_idx].append(gen_type)
          assigned = True
          break

    if not assigned:
      unassigned.append(gen_type)

  # If we have unassigned items (should be rare/impossible with current distribution),
  # force assign them to any non-full bucket, ignoring constraints if necessary,
  # or try to swap. For now, we'll force assign to fill up.
  for gen_type in unassigned:
    for bucket in assignments:
      if len(bucket) < variants_per_image:
        bucket.append(gen_type)
        break

  # Shuffle variants within each image so the suffixes (a,b,c...) don't correlate with type
  for bucket in assignments:
    rng.shuffle(bucket)

  return assignments


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Create omni generation dataset combining all strategies"
  )
  parser.add_argument(
    "--dataset-dir",
    type=Path,
    default=Path("synthetic_data/datasets/v04"),
    help="Path to the v04 dataset directory",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Output directory for omni images (default: dataset_dir/omni)",
  )
  parser.add_argument(
    "--variants",
    type=int,
    default=5,
    help="Number of variants to create per image (default: 5)",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print what would be done without creating files",
  )
  args = parser.parse_args()

  # Set random seed
  random.seed(args.seed)
  rng = random.Random(args.seed)

  # Resolve paths
  dataset_dir = args.dataset_dir.resolve()
  if args.output_dir:
    output_dir = args.output_dir.resolve()
  else:
    output_dir = dataset_dir / "omni"

  if not dataset_dir.exists():
    print(f"❌ Dataset directory not found: {dataset_dir}")
    sys.exit(1)

  # Get image pairs
  pairs = get_image_pairs(dataset_dir)
  if not pairs:
    print("❌ No valid image pairs found in dataset")
    sys.exit(1)

  print("=" * 60)
  print("🖼️  CREATING OMNI GENERATION DATASET")
  print(f"   Dataset: {dataset_dir}")
  print(f"   Output: {output_dir}")
  print(f"   Images: {len(pairs)}")
  print(f"   Variants per image: {args.variants}")
  print(f"   Total examples: ~{len(pairs) * args.variants}")
  print("=" * 60)

  # Show distribution
  print("\nTarget distribution:")
  for cat, weight in DISTRIBUTION.items():
    print(f"   {cat}: {weight * 100:.0f}%")

  if args.dry_run:
    print("\n🔍 DRY RUN - No files will be created\n")

    if not args.test_only:
      type_assignments = assign_generation_types(len(pairs), args.variants, rng)

      # Count types
      type_counts: dict[str, int] = {}
      for image_types in type_assignments:
        for gen_type in image_types:
          type_counts[gen_type] = type_counts.get(gen_type, 0) + 1

      print("Type distribution:")
      total = sum(type_counts.values())
      for gen_type, count in sorted(type_counts.items()):
        pct = (count / total) * 100 if total > 0 else 0
        print(f"   {gen_type}: {count} ({pct:.1f}%)")

      print("\nSample assignments:")
      for i, (image_num, _, _) in enumerate(pairs[:3]):
        print(f"   {image_num}:")
        for j, gen_type in enumerate(type_assignments[i]):
          suffix = VARIANT_SUFFIXES[j]
          print(f"      {image_num}_{suffix}.png -> {gen_type}")

    sys.exit(0)

  # Create output directory
  output_dir.mkdir(parents=True, exist_ok=True)

  # Assign generation types
  type_assignments = assign_generation_types(len(pairs), args.variants, rng)

  # Track CSV rows
  csv_rows: List[dict] = []
  created_count = 0

  print(f"\n📦 Processing {len(pairs)} image pairs...")

  # Process each image pair
  for i, (image_num, gen_path, render_path) in enumerate(pairs):
    try:
      img_gen = Image.open(gen_path).convert("RGB")
      img_render = Image.open(render_path).convert("RGB")

      gen_types = type_assignments[i]

      for j, gen_type in enumerate(gen_types):
        suffix = VARIANT_SUFFIXES[j]
        output_filename = f"{image_num}_{suffix}.png"
        output_path = output_dir / output_filename

        # Create omni image
        omni_img = create_omni_image(img_gen, img_render, gen_type, rng)

        # Save
        omni_img.save(output_path)
        created_count += 1

        print(f"      - {output_filename}: {gen_type}")

        # Add to CSV data
        csv_rows.append(
          {
            "omni": f"omni/{output_filename}",
            "generation": f"generations/{image_num}.png",
            "prompt": PROMPT,
          }
        )

      print(f"✅ {image_num}: Created {len(gen_types)} variants")

    except Exception as e:
      print(f"❌ Error processing {image_num}: {e}")

  # Write CSV file
  csv_path = dataset_dir / "omni_v04.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["omni", "generation", "prompt"])
    writer.writeheader()
    writer.writerows(csv_rows)

  print("\n" + "=" * 60)
  print("✨ OMNI DATASET COMPLETE")
  print(f"   Created {created_count} omni images")
  print(f"   CSV saved to: {csv_path}")
  print("=" * 60)

  # Print distribution summary
  type_counts: dict[str, int] = {}
  for image_types in type_assignments:
    for gen_type in image_types:
      type_counts[gen_type] = type_counts.get(gen_type, 0) + 1

  print("\nGeneration type distribution:")
  total = sum(type_counts.values())

  # Group by category
  categories = {
    "full": [TYPE_FULL],
    "quadrant": QUADRANT_TYPES,
    "half": HALF_TYPES,
    "middle": MIDDLE_TYPES,
    "strip": STRIP_TYPES,
    "infill": INFILL_TYPES,
  }

  for cat, types in categories.items():
    cat_count = sum(type_counts.get(t, 0) for t in types)
    pct = (cat_count / total) * 100 if total > 0 else 0
    print(f"   {cat}: {cat_count} ({pct:.1f}%)")


if __name__ == "__main__":
  main()
