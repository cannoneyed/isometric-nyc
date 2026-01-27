"""
Create omni generation dataset combining all generation strategies.

This script generates training examples combining:
1. Full generation (20%) - entire tile is input pixels
2. Quadrant generation (20%) - one quarter input, rest generation
3. Half generation (20%) - one half input, one half generation
4. Middle generation (15%) - middle strip (vertical/horizontal) is input
5. Rectangle strips (10%) - full horizontal/vertical strip of input (25-60%)
6. Rectangle infills (15%) - rectangle of input (25-60% area) anywhere

All input pixels are outlined with a 1px solid red border.

Transformations can be applied to input regions (in order):
- desaturation: Reduce color saturation (0.0 to 1.0) - kills the "satellite green"
- noise: Add multiplicative grain (0.0 to 1.0) - mimics texture loss, looks like gritty paper
- gamma_shift: Apply gamma crush (0.0 to 1.0) - THE SECRET SAUCE
  * Pushes dark greys to black while keeping lighter areas visible
  * Separates "tree tops" (visible) from "ground" (black)
  * Destroys the flat look of satellite photos
  * At intensity=1.0: gamma=1.8 + 0.7x brightness (the perfect corruption)

This "Perfect Corruption" recipe ensures inputs are dark, gritty, high-contrast versions
of the clean targets, forcing the model to become a "Light Resurrector"

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_omni_dataset.py
  uv run python src/isometric_nyc/synthetic_data/create_omni_dataset.py --dry-run

  # The "Perfect Corruption" recipe (recommended for training):
  uv run python src/isometric_nyc/synthetic_data/create_omni_dataset.py \
    --desaturation 0.5 --noise 1.0 --gamma-shift 1.0

  # With custom CSV specifying per-image settings:
  uv run python src/isometric_nyc/synthetic_data/create_omni_dataset.py --csv custom_settings.csv

CSV format (columns):
  name          - File name (without extension) in generations/inputs
  n_variants    - Number of variants to create for this image
  prompt        - Optional exact prompt to use (overrides default prompt entirely)
  prompt_suffix - Optional suffix to append to the standard prompt (ignored if prompt is set)
  noise         - Optional noise level (0.0 to 1.0)
  desaturation  - Optional desaturation level (0.0 to 1.0)
  gamma_shift   - Optional gamma shift level (0.0 to 1.0)
"""

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

from isometric_nyc.generation.image_preprocessing import (
  apply_desaturation,
  apply_gamma_shift,
  apply_noise,
)


@dataclass
class ImageSettings:
  """Per-image settings from CSV or defaults."""

  name: str
  n_variants: int
  prompt: str = ""  # If set, overrides the default prompt entirely
  prompt_suffix: str = (
    ""  # If set, appended to default prompt (ignored if prompt is set)
  )
  noise: float = 0.0
  desaturation: float = 0.0
  gamma_shift: float = 0.0


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


def get_input_regions(
  gen_type: str, width: int, height: int, rng: random.Random
) -> List[Tuple[int, int, int, int]]:
  """
  Get the regions that should contain input pixels for a given generation type.

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
  img_input: Image.Image,
  gen_type: str,
  rng: random.Random,
  noise: float = 0.0,
  desaturation: float = 0.0,
  gamma_shift: float = 0.0,
) -> Image.Image:
  """
  Create an omni training image.

  Takes the generation image and replaces specified regions with input pixels,
  applies transformations (noise, desaturation, gamma_shift) to input regions,
  then draws red outlines around the input regions.

  Args:
    img_gen: Generated image
    img_input: Input image
    gen_type: Type of generation layout
    rng: Random number generator
    noise: Noise intensity to apply to input regions (0.0 to 1.0)
    desaturation: Desaturation intensity for input regions (0.0 to 1.0)
    gamma_shift: Gamma shift intensity for input regions (0.0 to 1.0)

  Returns:
    Final omni image with transformations and outlines
  """
  width, height = TARGET_SIZE

  # Resize images if needed
  if img_gen.size != TARGET_SIZE:
    img_gen = img_gen.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
  if img_input.size != TARGET_SIZE:
    img_input = img_input.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

  # Get input regions
  regions = get_input_regions(gen_type, width, height, rng)

  # Apply transformations to input image if any are specified
  # Order: desaturation -> noise (grain) -> gamma_shift (crush)
  # The "Perfect Corruption" recipe:
  # 1. Desaturate to kill the satellite green and prevent color overfitting
  # 2. Add grain to mimic texture loss (gritty paper look)
  # 3. Gamma crush to separate highlights from shadows and destroy flatness
  transformed_input = img_input
  if noise > 0 or desaturation > 0 or gamma_shift > 0:
    if desaturation > 0:
      transformed_input = apply_desaturation(transformed_input, desaturation)
    if noise > 0:
      transformed_input = apply_noise(transformed_input, noise)
    if gamma_shift > 0:
      transformed_input = apply_gamma_shift(transformed_input, gamma_shift)

  # Start with generation image (or empty for full input)
  if gen_type == TYPE_FULL:
    final_image = transformed_input.copy()
    # Ensure we draw the border for full input
    # get_input_regions returns the full box, so draw_multi_region_outline will handle it
  else:
    final_image = img_gen.copy()
    # Paste transformed input regions
    for x1, y1, x2, y2 in regions:
      input_crop = transformed_input.crop((x1, y1, x2, y2))
      final_image.paste(input_crop, (x1, y1))

  # Draw outlines
  draw_multi_region_outline(final_image, regions, width=OUTLINE_WIDTH)

  return final_image


def get_image_pairs(dataset_dir: Path) -> List[Tuple[str, Path, Path]]:
  """
  Get all matching generation/input image pairs.

  Returns:
      List of (image_number, generation_path, input_path) tuples.
  """
  generations_dir = dataset_dir / "generations"
  inputs_dir = dataset_dir / "inputs"

  pairs = []

  for gen_path in sorted(generations_dir.glob("*.png")):
    image_num = gen_path.stem
    input_path = inputs_dir / f"{image_num}.png"

    if input_path.exists():
      pairs.append((image_num, gen_path, input_path))
    else:
      print(f"⚠️  No matching input for generation {image_num}")

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


def assign_generation_types_variable(
  variant_counts: List[int], rng: random.Random
) -> List[List[str]]:
  """
  Assign generation types when each image has a variable number of variants.

  This is similar to assign_generation_types but handles different variant counts
  per image rather than a fixed number.

  Constraints per image:
  1. Max 1 "full"
  2. Max 1 "middle" category variant
  3. Max 1 of the same "half" type
  """
  num_images = len(variant_counts)
  total_variants = sum(variant_counts)

  if total_variants == 0:
    return [[] for _ in range(num_images)]

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
  def get_priority(t: str) -> int:
    if t == TYPE_FULL:
      return 0
    if t in MIDDLE_TYPES:
      return 1
    if t in HALF_TYPES:
      return 2
    return 3

  type_pool.sort(key=get_priority)

  # Distribute to images (each with its own capacity)
  assignments: List[List[str]] = [[] for _ in range(num_images)]

  # Shuffle image order for fairer distribution
  image_indices = list(range(num_images))
  rng.shuffle(image_indices)

  # For each type in the pool, try to find a valid slot
  unassigned = []

  for gen_type in type_pool:
    assigned = False
    start_idx = rng.randint(0, num_images - 1)

    for i in range(num_images):
      idx = (start_idx + i) % num_images
      image_idx = image_indices[idx]

      if len(assignments[image_idx]) < variant_counts[image_idx]:
        if is_valid_assignment(assignments[image_idx], gen_type):
          assignments[image_idx].append(gen_type)
          assigned = True
          break

    if not assigned:
      unassigned.append(gen_type)

  # Force assign any remaining (should be rare)
  for gen_type in unassigned:
    for i, bucket in enumerate(assignments):
      if len(bucket) < variant_counts[i]:
        bucket.append(gen_type)
        break

  # Shuffle variants within each image
  for bucket in assignments:
    rng.shuffle(bucket)

  return assignments


def load_csv_settings(csv_path: Path) -> dict[str, ImageSettings]:
  """
  Load per-image settings from a CSV file.

  CSV columns:
    name          - File name (without extension) in generations/inputs
    n_variants    - Number of variants to create for this image
    prompt        - Optional exact prompt to use (overrides default prompt entirely)
    prompt_suffix - Optional suffix to append to the standard prompt (ignored if prompt is set)
    noise         - Optional noise level (0.0 to 1.0)
    desaturation  - Optional desaturation level (0.0 to 1.0)
    gamma_shift   - Optional gamma shift level (0.0 to 1.0)

  Returns:
    Dict mapping image name to ImageSettings.
  """
  settings: dict[str, ImageSettings] = {}

  with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)

    # Validate required columns
    if reader.fieldnames is None:
      raise ValueError("CSV file is empty or has no headers")

    required_cols = {"name", "n_variants"}
    missing = required_cols - set(reader.fieldnames)
    if missing:
      raise ValueError(f"CSV missing required columns: {missing}")

    for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
      name = row.get("name", "").strip()
      if not name:
        print(f"   ⚠️  Row {row_num}: Skipping empty name")
        continue

      # Remove file extension if present
      if name.endswith(".png"):
        name = name[:-4]

      try:
        n_variants = int(row.get("n_variants", "5"))
      except ValueError:
        print(
          f"   ⚠️  Row {row_num}: Invalid n_variants '{row.get('n_variants')}', using 5"
        )
        n_variants = 5

      # Get prompt (exact override) or prompt_suffix (appended to default)
      prompt = row.get("prompt", "") or ""
      prompt = prompt.strip()
      prompt_suffix = row.get("prompt_suffix", "") or ""
      prompt_suffix = prompt_suffix.strip()

      # Parse optional float parameters
      def parse_float_param(param_name: str, default: float = 0.0) -> float:
        value_str = row.get(param_name, "")
        # Handle None values from empty CSV cells
        if value_str is None:
          value_str = ""
        value_str = value_str.strip()
        if not value_str:
          return default
        try:
          value = float(value_str)
          return max(0.0, min(1.0, value))  # Clamp to [0, 1]
        except ValueError:
          print(
            f"   ⚠️  Row {row_num}: Invalid {param_name} '{value_str}', using {default}"
          )
          return default

      noise = parse_float_param("noise")
      desaturation = parse_float_param("desaturation")
      gamma_shift = parse_float_param("gamma_shift")

      settings[name] = ImageSettings(
        name=name,
        n_variants=n_variants,
        prompt=prompt,
        prompt_suffix=prompt_suffix,
        noise=noise,
        desaturation=desaturation,
        gamma_shift=gamma_shift,
      )

  return settings


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
    "--csv",
    type=Path,
    default=None,
    help="CSV file with per-image settings (columns: name, n_variants, prompt, prompt_suffix, noise, desaturation, gamma_shift)",
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
  parser.add_argument(
    "--noise",
    type=float,
    default=0.0,
    help="Add noise to input regions (0.0 to 1.0, default: 0.0)",
  )
  parser.add_argument(
    "--desaturation",
    type=float,
    default=0.0,
    help="Desaturate input regions (0.0 to 1.0, default: 0.0)",
  )
  parser.add_argument(
    "--gamma-shift",
    type=float,
    default=0.0,
    help="Darken input regions via gamma shift (0.0 to 1.0, default: 0.0)",
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

  # Load CSV settings if provided
  csv_settings: dict[str, ImageSettings] = {}
  if args.csv:
    if not args.csv.exists():
      print(f"❌ CSV file not found: {args.csv}")
      sys.exit(1)
    print(f"📋 Loading settings from: {args.csv}")
    csv_settings = load_csv_settings(args.csv)
    print(f"   Loaded settings for {len(csv_settings)} images")

  # Get image pairs
  all_pairs = get_image_pairs(dataset_dir)
  if not all_pairs:
    print("❌ No valid image pairs found in dataset")
    sys.exit(1)

  # Filter pairs to only those in CSV (if CSV provided)
  if csv_settings:
    pairs = [(num, gen, ren) for num, gen, ren in all_pairs if num in csv_settings]
    if not pairs:
      print("❌ No matching images found between CSV and dataset")
      print(f"   CSV names: {list(csv_settings.keys())[:5]}...")
      print(f"   Dataset names: {[p[0] for p in all_pairs[:5]]}...")
      sys.exit(1)
    print(f"   Matched {len(pairs)}/{len(all_pairs)} images from dataset")
  else:
    pairs = all_pairs

  # Build per-image settings (from CSV or defaults)
  image_settings_list: List[ImageSettings] = []
  total_variants = 0
  for image_num, _, _ in pairs:
    if image_num in csv_settings:
      settings = csv_settings[image_num]
    else:
      settings = ImageSettings(
        name=image_num,
        n_variants=args.variants,
        noise=args.noise,
        desaturation=args.desaturation,
        gamma_shift=args.gamma_shift,
      )
    image_settings_list.append(settings)
    total_variants += settings.n_variants

  print("=" * 60)
  print("🖼️  CREATING OMNI GENERATION DATASET")
  print(f"   Dataset: {dataset_dir}")
  print(f"   Output: {output_dir}")
  print(f"   Images: {len(pairs)}")
  if csv_settings:
    print(f"   Settings from CSV: {args.csv.name}")
    print(f"   Total examples: {total_variants}")
  else:
    print(f"   Variants per image: {args.variants}")
    print(f"   Total examples: ~{len(pairs) * args.variants}")
  print("=" * 60)

  # Show distribution
  print("\nTarget distribution:")
  for cat, weight in DISTRIBUTION.items():
    print(f"   {cat}: {weight * 100:.0f}%")

  # Build list of variant counts for type assignment
  variant_counts = [s.n_variants for s in image_settings_list]

  if args.dry_run:
    print("\n🔍 DRY RUN - No files will be created\n")

    type_assignments = assign_generation_types_variable(variant_counts, rng)

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
    for i, (image_num, _, _) in enumerate(pairs[:5]):
      settings = image_settings_list[i]
      if settings.prompt:
        prompt_str = " (custom prompt)"
      elif settings.prompt_suffix:
        prompt_str = f" (suffix: '{settings.prompt_suffix}')"
      else:
        prompt_str = ""
      print(f"   {image_num} ({settings.n_variants} variants){prompt_str}:")
      for j, gen_type in enumerate(type_assignments[i]):
        suffix = VARIANT_SUFFIXES[j]
        print(f"      {image_num}_{suffix}.png -> {gen_type}")

    sys.exit(0)

  # Create output directory
  output_dir.mkdir(parents=True, exist_ok=True)

  # Assign generation types with variable counts per image
  type_assignments = assign_generation_types_variable(variant_counts, rng)

  # Track CSV rows
  csv_rows: List[dict] = []
  created_count = 0

  print(f"\n📦 Processing {len(pairs)} image pairs...")

  # Process each image pair
  for i, (image_num, gen_path, input_path) in enumerate(pairs):
    try:
      img_gen = Image.open(gen_path).convert("RGB")
      img_input = Image.open(input_path).convert("RGB")

      gen_types = type_assignments[i]
      settings = image_settings_list[i]

      # Build the prompt
      # Priority:
      # 1. CSV `prompt` column (explicit user override - highest priority)
      # 2. `debug/<image_num>/prompt.txt` if present
      # 3. Default PROMPT + CSV `prompt_suffix`
      # 4. Default PROMPT
      if settings.prompt:
        full_prompt = settings.prompt
      else:
        # Check for prompt file in debug directory
        debug_prompt_path = dataset_dir / "debug" / image_num / "prompt.txt"
        if debug_prompt_path.exists():
          full_prompt = debug_prompt_path.read_text().strip()
        elif settings.prompt_suffix:
          full_prompt = f"{PROMPT} {settings.prompt_suffix}"
        else:
          full_prompt = PROMPT

      for j, gen_type in enumerate(gen_types):
        suffix = VARIANT_SUFFIXES[j]
        output_filename = f"{image_num}_{suffix}.png"
        output_path = output_dir / output_filename

        # Create omni image
        omni_img = create_omni_image(
          img_gen,
          img_input,
          gen_type,
          rng,
          noise=settings.noise,
          desaturation=settings.desaturation,
          gamma_shift=settings.gamma_shift,
        )

        # Save
        omni_img.save(output_path)
        created_count += 1

        print(f"      - {output_filename}: {gen_type}")

        # Add to CSV data
        # Escape prompt: replace newlines with literal "\n" and remove commas
        # Wrap in quotes for safe parsing
        escaped_prompt = full_prompt.replace("\n", "\\n").replace(",", "")
        csv_rows.append(
          {
            "omni": f"omni/{output_filename}",
            "generation": f"generations/{image_num}.png",
            "prompt": f'"{escaped_prompt}"',
          }
        )

      # Show prompt info in log
      if settings.prompt:
        prompt_info = " (custom prompt)"
      elif settings.prompt_suffix:
        prompt_info = f" (suffix: '{settings.prompt_suffix}')"
      else:
        prompt_info = ""
      print(f"✅ {image_num}: Created {len(gen_types)} variants{prompt_info}")

    except Exception as e:
      print(f"❌ Error processing {image_num}: {e}")

  # Write CSV file
  csv_path = dataset_dir / "omni.csv"
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
