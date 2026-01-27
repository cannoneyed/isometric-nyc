"""
Create inpainting examples for fine-tuning a model to fill arbitrary rectangles.

This script generates ~120 training examples from the v04 dataset, where each example
shows a generation image with a rectangular region replaced by pixels from the
corresponding render image, outlined in red.

Rectangle types (20% each):
- Vertical band: full height, variable width
- Horizontal band: full width, variable height
- Vertical rectangle: taller than wide, some touching edges
- Horizontal rectangle: wider than tall, some touching edges
- Inner square: square in the middle of the tile

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inpainting_examples.py
  uv run python src/isometric_nyc/synthetic_data/create_inpainting_examples.py --dry-run
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

# Rectangle type constants
RECT_VERTICAL_BAND = "vertical_band"
RECT_HORIZONTAL_BAND = "horizontal_band"
RECT_VERTICAL_RECT = "vertical_rect"
RECT_HORIZONTAL_RECT = "horizontal_rect"
RECT_INNER_SQUARE = "inner_square"

# All rectangle types with equal distribution
RECT_TYPES = [
  RECT_VERTICAL_BAND,
  RECT_HORIZONTAL_BAND,
  RECT_VERTICAL_RECT,
  RECT_HORIZONTAL_RECT,
  RECT_INNER_SQUARE,
]

# Alphabetical suffixes for variants
VARIANT_SUFFIXES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

# Prompt template
PROMPT = (
  "Fill in the outlined section with the missing pixels corresponding to the "
  "<isometric nyc pixel art> style, removing the border and exactly following "
  "the shape/style/structure of the surrounding image."
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

  Args:
      img: The image to draw on (modified in place).
      box: (x1, y1, x2, y2) coordinates of the region.
      color: RGB color tuple for the outline.
      width: Width of the outline in pixels.
  """
  draw = ImageDraw.Draw(img)
  x1, y1, x2, y2 = box

  # Draw rectangle outline with specified width
  for i in range(width):
    draw.rectangle(
      [x1 + i, y1 + i, x2 - 1 - i, y2 - 1 - i],
      outline=color,
      fill=None,
    )


def get_max_area(width: int, height: int) -> int:
  """Calculate maximum allowed rectangle area (1/2 of total)."""
  return (width * height) // 2


def generate_vertical_band(
  width: int, height: int, rng: random.Random
) -> Tuple[int, int, int, int]:
  """
  Generate a vertical band (full height, variable width).

  The band spans from top to bottom and has width <= 1/2 of image width.
  """
  max_band_width = width // 2
  min_band_width = width // 8  # At least 128px for 1024 image

  band_width = rng.randint(min_band_width, max_band_width)

  # Random x position, ensuring it fits
  max_x = width - band_width
  x1 = rng.randint(0, max_x)
  x2 = x1 + band_width

  return (x1, 0, x2, height)


def generate_horizontal_band(
  width: int, height: int, rng: random.Random
) -> Tuple[int, int, int, int]:
  """
  Generate a horizontal band (full width, variable height).

  The band spans from left to right and has height <= 1/2 of image height.
  """
  max_band_height = height // 2
  min_band_height = height // 8  # At least 128px for 1024 image

  band_height = rng.randint(min_band_height, max_band_height)

  # Random y position, ensuring it fits
  max_y = height - band_height
  y1 = rng.randint(0, max_y)
  y2 = y1 + band_height

  return (0, y1, width, y2)


def generate_vertical_rect(
  width: int, height: int, rng: random.Random, touch_edge: bool = True
) -> Tuple[int, int, int, int]:
  """
  Generate a vertical rectangle (taller than wide).

  If touch_edge is True, the rectangle will touch at least one edge.
  Area must be <= 1/2 of total image area.
  """
  max_area = get_max_area(width, height)

  # Vertical rect: height > width
  min_rect_height = height // 3
  max_rect_height = height - 64  # Leave some margin

  rect_height = rng.randint(min_rect_height, max_rect_height)

  # Calculate max width based on area constraint
  max_rect_width = min(max_area // rect_height, rect_height - 1)
  min_rect_width = width // 8

  if max_rect_width < min_rect_width:
    max_rect_width = min_rect_width

  rect_width = rng.randint(min_rect_width, max_rect_width)

  if touch_edge:
    # Choose which edge(s) to touch
    edge_choice = rng.choice(["top", "bottom", "left", "right"])

    if edge_choice == "top":
      y1 = 0
      y2 = rect_height
      x1 = rng.randint(0, width - rect_width)
      x2 = x1 + rect_width
    elif edge_choice == "bottom":
      y2 = height
      y1 = height - rect_height
      x1 = rng.randint(0, width - rect_width)
      x2 = x1 + rect_width
    elif edge_choice == "left":
      x1 = 0
      x2 = rect_width
      y1 = rng.randint(0, height - rect_height)
      y2 = y1 + rect_height
    else:  # right
      x2 = width
      x1 = width - rect_width
      y1 = rng.randint(0, height - rect_height)
      y2 = y1 + rect_height
  else:
    # Don't touch any edge
    margin = 32
    x1 = rng.randint(margin, width - rect_width - margin)
    y1 = rng.randint(margin, height - rect_height - margin)
    x2 = x1 + rect_width
    y2 = y1 + rect_height

  return (x1, y1, x2, y2)


def generate_horizontal_rect(
  width: int, height: int, rng: random.Random, touch_edge: bool = True
) -> Tuple[int, int, int, int]:
  """
  Generate a horizontal rectangle (wider than tall).

  If touch_edge is True, the rectangle will touch at least one edge.
  Area must be <= 1/2 of total image area.
  """
  max_area = get_max_area(width, height)

  # Horizontal rect: width > height
  min_rect_width = width // 3
  max_rect_width = width - 64  # Leave some margin

  rect_width = rng.randint(min_rect_width, max_rect_width)

  # Calculate max height based on area constraint
  max_rect_height = min(max_area // rect_width, rect_width - 1)
  min_rect_height = height // 8

  if max_rect_height < min_rect_height:
    max_rect_height = min_rect_height

  rect_height = rng.randint(min_rect_height, max_rect_height)

  if touch_edge:
    # Choose which edge(s) to touch
    edge_choice = rng.choice(["top", "bottom", "left", "right"])

    if edge_choice == "top":
      y1 = 0
      y2 = rect_height
      x1 = rng.randint(0, width - rect_width)
      x2 = x1 + rect_width
    elif edge_choice == "bottom":
      y2 = height
      y1 = height - rect_height
      x1 = rng.randint(0, width - rect_width)
      x2 = x1 + rect_width
    elif edge_choice == "left":
      x1 = 0
      x2 = rect_width
      y1 = rng.randint(0, height - rect_height)
      y2 = y1 + rect_height
    else:  # right
      x2 = width
      x1 = width - rect_width
      y1 = rng.randint(0, height - rect_height)
      y2 = y1 + rect_height
  else:
    # Don't touch any edge
    margin = 32
    x1 = rng.randint(margin, width - rect_width - margin)
    y1 = rng.randint(margin, height - rect_height - margin)
    x2 = x1 + rect_width
    y2 = y1 + rect_height

  return (x1, y1, x2, y2)


def generate_inner_square(
  width: int, height: int, rng: random.Random
) -> Tuple[int, int, int, int]:
  """
  Generate an inner square somewhere in the middle of the tile.

  The square doesn't touch any edge and has area <= 1/2 of total.
  """
  max_area = get_max_area(width, height)
  max_side = int(max_area**0.5)  # sqrt for square

  # Allow larger squares - min is 1/4 of image, max is ~70% (constrained by area)
  min_side = min(width, height) // 4  # 256px for 1024 image
  max_side = min(max_side, int(min(width, height) * 0.7))  # ~716px max

  # Bias toward larger squares by using weighted choice
  # 30% small (min to 1/3), 30% medium (1/3 to 2/3), 40% large (2/3 to max)
  size_range = max_side - min_side
  small_max = min_side + size_range // 3
  medium_max = min_side + (2 * size_range) // 3

  size_category = rng.random()
  if size_category < 0.3:
    side = rng.randint(min_side, small_max)
  elif size_category < 0.6:
    side = rng.randint(small_max, medium_max)
  else:
    side = rng.randint(medium_max, max_side)

  # Keep away from edges with smaller margin to allow larger squares
  margin = 32
  x1 = rng.randint(margin, width - side - margin)
  y1 = rng.randint(margin, height - side - margin)
  x2 = x1 + side
  y2 = y1 + side

  return (x1, y1, x2, y2)


def generate_rectangle(
  rect_type: str, width: int, height: int, rng: random.Random
) -> Tuple[int, int, int, int]:
  """
  Generate a rectangle of the specified type.

  Args:
      rect_type: One of the RECT_* constants.
      width: Image width.
      height: Image height.
      rng: Random number generator for reproducibility.

  Returns:
      (x1, y1, x2, y2) coordinates of the rectangle.
  """
  if rect_type == RECT_VERTICAL_BAND:
    return generate_vertical_band(width, height, rng)
  elif rect_type == RECT_HORIZONTAL_BAND:
    return generate_horizontal_band(width, height, rng)
  elif rect_type == RECT_VERTICAL_RECT:
    # Half touch edge, half don't
    touch_edge = rng.choice([True, False])
    return generate_vertical_rect(width, height, rng, touch_edge)
  elif rect_type == RECT_HORIZONTAL_RECT:
    # Half touch edge, half don't
    touch_edge = rng.choice([True, False])
    return generate_horizontal_rect(width, height, rng, touch_edge)
  elif rect_type == RECT_INNER_SQUARE:
    return generate_inner_square(width, height, rng)
  else:
    raise ValueError(f"Unknown rectangle type: {rect_type}")


def create_inpainting_image(
  img_gen: Image.Image,
  img_render: Image.Image,
  rect_box: Tuple[int, int, int, int],
) -> Image.Image:
  """
  Create an inpainting training image.

  Takes the generation image and replaces the specified rectangle with
  pixels from the render image, then draws a red outline.

  Args:
      img_gen: The generation (pixel art) image.
      img_render: The render image.
      rect_box: (x1, y1, x2, y2) coordinates of the rectangle to replace.

  Returns:
      The composited image with red outline.
  """
  width, height = TARGET_SIZE

  # Resize images if needed
  if img_gen.size != TARGET_SIZE:
    img_gen = img_gen.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
  if img_render.size != TARGET_SIZE:
    img_render = img_render.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

  # Start with the generation image
  final_image = img_gen.copy()

  # Crop the rectangle from the render image and paste it
  x1, y1, x2, y2 = rect_box
  render_crop = img_render.crop((x1, y1, x2, y2))
  final_image.paste(render_crop, (x1, y1))

  # Draw the red outline on top
  draw_outline(final_image, rect_box)

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


def assign_rectangle_types(num_images: int, variants_per_image: int) -> List[List[str]]:
  """
  Assign rectangle types to achieve ~20% distribution for each type.

  Args:
      num_images: Number of source images.
      variants_per_image: Number of variants to create per image.

  Returns:
      List of lists, where each inner list contains rectangle types for one image.
  """
  total_variants = num_images * variants_per_image

  # Calculate how many of each type we need
  types_per_category = total_variants // len(RECT_TYPES)
  remainder = total_variants % len(RECT_TYPES)

  # Create pool of all rectangle types
  type_pool: List[str] = []
  for rect_type in RECT_TYPES:
    count = types_per_category + (1 if RECT_TYPES.index(rect_type) < remainder else 0)
    type_pool.extend([rect_type] * count)

  # Shuffle to distribute randomly
  random.shuffle(type_pool)

  # Distribute to images
  assignments: List[List[str]] = []
  pool_idx = 0

  for _ in range(num_images):
    image_types = type_pool[pool_idx : pool_idx + variants_per_image]
    assignments.append(image_types)
    pool_idx += variants_per_image

  return assignments


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Create inpainting training examples from v04 dataset"
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
    help="Output directory for inpainting images (default: dataset_dir/inpainting)",
  )
  parser.add_argument(
    "--variants",
    type=int,
    default=3,
    help="Number of variants to create per image (default: 3)",
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
    output_dir = dataset_dir / "inpainting"

  if not dataset_dir.exists():
    print(f"❌ Dataset directory not found: {dataset_dir}")
    sys.exit(1)

  # Get image pairs
  pairs = get_image_pairs(dataset_dir)
  if not pairs:
    print("❌ No valid image pairs found in dataset")
    sys.exit(1)

  print("=" * 60)
  print("🖼️  CREATING INPAINTING EXAMPLES")
  print(f"   Dataset: {dataset_dir}")
  print(f"   Output: {output_dir}")
  print(f"   Images: {len(pairs)}")
  print(f"   Variants per image: {args.variants}")
  print(f"   Total examples: ~{len(pairs) * args.variants}")
  print("=" * 60)

  # Assign rectangle types to achieve equal distribution
  type_assignments = assign_rectangle_types(len(pairs), args.variants)

  if args.dry_run:
    print("\n🔍 DRY RUN - No files will be created\n")

    type_counts = {t: 0 for t in RECT_TYPES}
    for image_types in type_assignments:
      for rect_type in image_types:
        type_counts[rect_type] += 1

    print("Distribution of rectangle types:")
    total = sum(type_counts.values())
    for rect_type, count in type_counts.items():
      pct = (count / total) * 100 if total > 0 else 0
      print(f"  {rect_type}: {count} ({pct:.1f}%)")

    print("\nSample assignments:")
    for i, (image_num, gen_path, render_path) in enumerate(pairs[:5]):
      print(f"  {image_num}:")
      for j, rect_type in enumerate(type_assignments[i]):
        suffix = VARIANT_SUFFIXES[j]
        print(f"    {image_num}_{suffix}.png -> {rect_type}")

    sys.exit(0)

  # Create output directory
  output_dir.mkdir(parents=True, exist_ok=True)

  # Track CSV rows
  csv_rows: List[dict] = []
  created_count = 0

  # Process each image pair
  for i, (image_num, gen_path, render_path) in enumerate(pairs):
    try:
      img_gen = Image.open(gen_path).convert("RGB")
      img_render = Image.open(render_path).convert("RGB")

      rect_types = type_assignments[i]

      for j, rect_type in enumerate(rect_types):
        suffix = VARIANT_SUFFIXES[j]
        output_filename = f"{image_num}_{suffix}.png"
        output_path = output_dir / output_filename

        # Generate rectangle
        width, height = TARGET_SIZE
        rect_box = generate_rectangle(rect_type, width, height, rng)

        # Create inpainting image
        inpainting_img = create_inpainting_image(img_gen, img_render, rect_box)

        # Save
        inpainting_img.save(output_path)
        created_count += 1

        # Add to CSV data
        csv_rows.append(
          {
            "inpainting": f"inpainting/{output_filename}",
            "generation": f"generations/{image_num}.png",
            "prompt": PROMPT,
          }
        )

      print(f"✅ {image_num}: Created {len(rect_types)} variants")

    except Exception as e:
      print(f"❌ Error processing {image_num}: {e}")

  # Write CSV file
  csv_path = dataset_dir / "inpainting_v04.csv"
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["inpainting", "generation", "prompt"])
    writer.writeheader()
    writer.writerows(csv_rows)

  print("\n" + "=" * 60)
  print("✨ INPAINTING EXAMPLES COMPLETE")
  print(f"   Created {created_count} inpainting images")
  print(f"   CSV saved to: {csv_path}")
  print("=" * 60)

  # Print distribution summary
  type_counts = {t: 0 for t in RECT_TYPES}
  for image_types in type_assignments:
    for rect_type in image_types:
      type_counts[rect_type] += 1

  print("\nRectangle type distribution:")
  total = sum(type_counts.values())
  for rect_type, count in type_counts.items():
    pct = (count / total) * 100 if total > 0 else 0
    print(f"  {rect_type}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
  main()
