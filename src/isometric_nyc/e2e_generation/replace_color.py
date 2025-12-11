"""
Replace a color in a quadrant's generation using soft blending.

This script performs color replacement while preserving anti-aliased edges
using a soft mask based on color distance (Euclidean norm in RGB space).

Usage:
  # Dry run - exports to exports subdir without saving to db
  uv run python src/isometric_nyc/e2e_generation/replace_color.py \\
    <generation_dir> -x 0 -y 0 \\
    --target-color "#4B697D" --replacement-color "#FF32FF" \\
    --dry-run

  # Apply to database
  uv run python src/isometric_nyc/e2e_generation/replace_color.py \\
    <generation_dir> -x 0 -y 0 \\
    --target-color "#4B697D" --replacement-color "#2A4A5F"

  # Custom softness (default is 60.0)
  uv run python src/isometric_nyc/e2e_generation/replace_color.py \\
    <generation_dir> -x 0 -y 0 \\
    --target-color "#4B697D" --replacement-color "#2A4A5F" \\
    --softness 80.0
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from isometric_nyc.e2e_generation.shared import (
  get_generation_config,
  get_quadrant_generation,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
  """
  Convert a hex color string to an RGB tuple.

  Args:
      hex_color: Hex color string (e.g., "#4B697D" or "4B697D")

  Returns:
      Tuple of (R, G, B) values (0-255)
  """
  hex_color = hex_color.lstrip("#")
  if len(hex_color) != 6:
    raise ValueError(f"Invalid hex color: {hex_color}. Expected 6 characters.")
  return (
    int(hex_color[0:2], 16),
    int(hex_color[2:4], 16),
    int(hex_color[4:6], 16),
  )


def soft_color_replace(
  image: np.ndarray,
  target_color: tuple[int, int, int],
  new_color: tuple[int, int, int],
  blend_softness: float,
) -> tuple[np.ndarray, np.ndarray]:
  """
  Replace a color while preserving anti-aliased edges using a soft mask.

  Uses Euclidean distance in RGB space to create a soft alpha mask,
  then performs alpha blending to smoothly replace the target color.

  Args:
      image: Input image (RGB, uint8).
      target_color: (R, G, B) color to replace (uint8).
      new_color: (R, G, B) replacement color (uint8).
      blend_softness: Controls the width of the blend transition.
                      Higher values = wider, softer edges. Try ranges 20.0 - 100.0.

  Returns:
      Tuple of (result_image_uint8, soft_mask_uint8)
  """
  # --- PREPARATION ---
  # Convert everything to floats between 0.0 and 1.0 for accurate math.
  img_float = image.astype(np.float32) / 255.0
  target_float = np.array(target_color, dtype=np.float32) / 255.0
  new_color_float = np.array(new_color, dtype=np.float32) / 255.0

  # Normalize softness parameter to 0-1 scale roughly matching color distance space
  # (Max Euclidean distance in unit RGB cube is sqrt(3) ~= 1.732)
  softness_scale = (blend_softness / 255.0) * np.sqrt(3)
  # Ensure it's not zero to avoid division errors
  softness_scale = max(softness_scale, 1e-5)

  # --- STEP 1: Calculate Color Distance Map ---
  # Find Euclidean distance between every pixel's color and the target color.
  # axis=2 calculates the norm across the R,G,B channels.
  # Result shape is (H, W). 0.0 means exact match, higher means very different.
  distances = np.linalg.norm(img_float - target_float, axis=2)

  # --- STEP 2: Create the Soft Mask (Alpha Matte) ---
  # We invert the distances: 0 distance should be 1.0 opacity.
  # We divide by softness_scale to control how quickly opacity drops off.
  alpha_mask = 1.0 - (distances / softness_scale)

  # Clip results to ensure the mask stays strictly between 0.0 and 1.0
  alpha_mask = np.clip(alpha_mask, 0.0, 1.0)

  # Reshape mask from (H, W) to (H, W, 1) so it can be multiplied with RGB images
  alpha_expanded = alpha_mask[:, :, np.newaxis]

  # --- STEP 3: Perform Alpha Blending ---
  # Create a solid image filled entirely with the new color
  solid_new_color_img = np.full_like(img_float, new_color_float)

  # Standard Alpha Compositing Formula:
  # Final = (Foreground * Alpha) + (Background * (1 - Alpha))
  # Foreground is the new color, Background is the original image.
  blended_float = (solid_new_color_img * alpha_expanded) + (
    img_float * (1.0 - alpha_expanded)
  )

  # --- FINALIZE ---
  # Convert back to uint8 format
  result_img = (blended_float * 255).astype(np.uint8)
  soft_mask_uint8 = (alpha_mask * 255).astype(np.uint8)

  return result_img, soft_mask_uint8


def process_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  target_color: tuple[int, int, int],
  replacement_color: tuple[int, int, int],
  softness: float,
  dry_run: bool,
  exports_dir: Path,
) -> bool:
  """
  Process a single quadrant, replacing the target color.

  Args:
      conn: Database connection
      config: Generation config
      x: Quadrant x coordinate
      y: Quadrant y coordinate
      target_color: RGB color to replace
      replacement_color: RGB color to use as replacement
      softness: Blend softness value
      dry_run: If True, export to file instead of saving to db
      exports_dir: Directory for dry-run exports

  Returns:
      True if successful, False otherwise
  """
  # Get the generation image
  generation_bytes = get_quadrant_generation(conn, x, y)
  if generation_bytes is None:
    print(f"❌ Error: No generation found for quadrant ({x}, {y})")
    return False

  # Convert to PIL Image, then to numpy array
  pil_image = png_bytes_to_image(generation_bytes)

  # Handle RGBA images - extract RGB channels for processing
  has_alpha = pil_image.mode == "RGBA"
  if has_alpha:
    # Store original alpha channel
    original_alpha = np.array(pil_image)[:, :, 3]
    # Convert to RGB for processing
    pil_image_rgb = pil_image.convert("RGB")
  else:
    pil_image_rgb = pil_image.convert("RGB")
    original_alpha = None

  img_array = np.array(pil_image_rgb)

  print(f"   🎨 Processing quadrant ({x}, {y})...")
  print(f"      Target color: RGB{target_color}")
  print(f"      Replacement color: RGB{replacement_color}")
  print(f"      Softness: {softness}")

  # Apply soft color replacement
  result_array, mask_array = soft_color_replace(
    img_array, target_color, replacement_color, softness
  )

  # Convert back to PIL Image
  result_image = Image.fromarray(result_array, mode="RGB")

  # Restore alpha channel if original had one
  if has_alpha and original_alpha is not None:
    result_rgba = np.zeros(
      (result_array.shape[0], result_array.shape[1], 4), dtype=np.uint8
    )
    result_rgba[:, :, :3] = result_array
    result_rgba[:, :, 3] = original_alpha
    result_image = Image.fromarray(result_rgba, mode="RGBA")

  # Calculate how many pixels were affected (mask > 0)
  affected_pixels = np.sum(mask_array > 0)
  total_pixels = mask_array.size
  affected_percent = (affected_pixels / total_pixels) * 100
  print(f"      Affected pixels: {affected_pixels:,} ({affected_percent:.1f}%)")

  if dry_run:
    # Export to file
    exports_dir.mkdir(parents=True, exist_ok=True)
    output_path = exports_dir / f"color_replace_{x}_{y}.png"
    mask_path = exports_dir / f"color_replace_{x}_{y}_mask.png"

    result_image.save(output_path, "PNG")
    Image.fromarray(mask_array, mode="L").save(mask_path, "PNG")

    print("   ✅ Dry run - exported to:")
    print(f"      Result: {output_path}")
    print(f"      Mask: {mask_path}")
  else:
    # Save back to database
    png_bytes = image_to_png_bytes(result_image)
    save_quadrant_generation(conn, config, x, y, png_bytes)
    print("   ✅ Saved to database")

  return True


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Replace a color in a quadrant's generation using soft blending."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "-x",
    type=int,
    required=True,
    help="X coordinate of the quadrant to edit",
  )
  parser.add_argument(
    "-y",
    type=int,
    required=True,
    help="Y coordinate of the quadrant to edit",
  )
  parser.add_argument(
    "--target-color",
    type=str,
    required=True,
    help="Hex code of the color to be replaced (e.g., '#4B697D' or '4B697D')",
  )
  parser.add_argument(
    "--replacement-color",
    type=str,
    required=True,
    help="Hex code of the replacement color (e.g., '#2A4A5F' or '2A4A5F')",
  )
  parser.add_argument(
    "--softness",
    type=float,
    default=60.0,
    help="Blend softness (higher = wider, softer edges). Default: 60.0. Try 20-100.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Export result to exports subdir without saving to database",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Parse colors
  try:
    target_color = hex_to_rgb(args.target_color)
  except ValueError as e:
    print(f"❌ Error parsing target color: {e}")
    return 1

  try:
    replacement_color = hex_to_rgb(args.replacement_color)
  except ValueError as e:
    print(f"❌ Error parsing replacement color: {e}")
    return 1

  exports_dir = generation_dir / "exports"

  print("\n🎨 Color Replacement Tool")
  print(f"   Generation dir: {generation_dir}")
  print(f"   Quadrant: ({args.x}, {args.y})")
  if args.dry_run:
    print(f"   Mode: DRY RUN (will export to {exports_dir})")
  else:
    print("   Mode: LIVE (will save to database)")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    success = process_quadrant(
      conn=conn,
      config=config,
      x=args.x,
      y=args.y,
      target_color=target_color,
      replacement_color=replacement_color,
      softness=args.softness,
      dry_run=args.dry_run,
      exports_dir=exports_dir,
    )

    return 0 if success else 1

  finally:
    conn.close()


if __name__ == "__main__":
  sys.exit(main())
