"""
Postprocess all tiles with pixelation and unified color quantization.

This script applies the pixelation effect from pixelate.py to all tiles
in the tileset, using a shared color palette across ALL tiles for consistency.

Usage:
    # Build palette and process all tiles:
    uv run python -m isometric_nyc.postprocess_tiles

    # Just build and save the palette:
    uv run python -m isometric_nyc.postprocess_tiles --build-palette-only

    # Process tiles using an existing palette:
    uv run python -m isometric_nyc.postprocess_tiles --palette palette.png
"""

import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# Default paths
DEFAULT_TILES_DIR = Path(__file__).parent.parent / "app" / "public" / "tiles"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "app" / "public" / "tiles_processed"
DEFAULT_PALETTE_PATH = Path(__file__).parent.parent / "app" / "public" / "palette.png"


def collect_all_tile_paths(tiles_dir: Path) -> list[Path]:
  """Collect all PNG tile paths from the tiles directory."""
  tile_paths: list[Path] = []
  for png_file in tiles_dir.rglob("*.png"):
    tile_paths.append(png_file)
  return sorted(tile_paths)


def sample_colors_from_tiles(
  tile_paths: list[Path],
  sample_size: int = 500,
  pixels_per_tile: int = 1000,
) -> list[tuple[int, int, int]]:
  """
  Sample colors from a subset of tiles to build a representative color set.

  Args:
      tile_paths: List of all tile paths
      sample_size: Number of tiles to sample from
      pixels_per_tile: Number of random pixels to sample from each tile

  Returns:
      List of RGB tuples representing sampled colors
  """
  # Sample a subset of tiles if we have more than sample_size
  if len(tile_paths) > sample_size:
    sampled_paths = random.sample(tile_paths, sample_size)
  else:
    sampled_paths = tile_paths

  all_colors: list[tuple[int, int, int]] = []

  for tile_path in sampled_paths:
    try:
      img = Image.open(tile_path).convert("RGB")
      width, height = img.size
      pixels = list(img.getdata())

      # Sample random pixels from this tile
      if len(pixels) > pixels_per_tile:
        sampled_pixels = random.sample(pixels, pixels_per_tile)
      else:
        sampled_pixels = pixels

      all_colors.extend(sampled_pixels)
    except Exception as e:
      print(f"Warning: Could not read {tile_path}: {e}")

  return all_colors


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = 32,
) -> Image.Image:
  """
  Build a unified palette image from sampled colors.

  Creates a small image with the quantized palette that can be used
  as a reference for applying to other images.

  Args:
      colors: List of RGB tuples
      num_colors: Target number of colors in the palette

  Returns:
      A palette image that can be used with Image.quantize()
  """
  # Create a composite image from all sampled colors
  # We'll make a square-ish image
  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  # Create image and populate with colors
  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  # Quantize this composite image to get our unified palette
  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)

  return palette_img


def save_palette(palette_img: Image.Image, output_path: Path) -> None:
  """Save the palette image to disk."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  palette_img.save(output_path)
  print(f"Saved palette to {output_path}")


def load_palette(palette_path: Path) -> Image.Image:
  """Load a palette image from disk."""
  return Image.open(palette_path)


def pixelate_and_quantize_with_palette(
  input_path: Path,
  output_path: Path,
  palette_img: Image.Image,
  pixel_scale: int = 2,
  dither: bool = True,
) -> bool:
  """
  Apply pixelation and quantization to a single tile using a shared palette.

  Args:
      input_path: Path to source tile
      output_path: Path to save processed tile
      palette_img: Palette image to use for quantization
      pixel_scale: Pixelation scale factor
      dither: Whether to apply dithering

  Returns:
      True if successful, False otherwise
  """
  try:
    # Load the image
    img = Image.open(input_path).convert("RGB")
    original_width, original_height = img.size

    # Skip pixelation if scale is 1
    if pixel_scale > 1:
      # Calculate small dimensions
      small_width = original_width // pixel_scale
      small_height = original_height // pixel_scale

      # Downscale with nearest neighbor
      img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
    else:
      img_small = img

    # Quantize using the shared palette
    img_quantized = img_small.quantize(
      palette=palette_img,
      dither=1 if dither else 0,
    )

    # Convert back to RGB
    img_quantized = img_quantized.convert("RGB")

    # Upscale back to original size if we downscaled
    if pixel_scale > 1:
      final_image = img_quantized.resize(
        (original_width, original_height), resample=Image.NEAREST
      )
    else:
      final_image = img_quantized

    # Save the result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path)
    return True

  except Exception as e:
    print(f"Error processing {input_path}: {e}")
    return False


def process_single_tile(args: tuple) -> tuple[Path, bool]:
  """
  Process a single tile (wrapper for multiprocessing).

  Args:
      args: Tuple of (input_path, output_path, palette_path, pixel_scale, dither)

  Returns:
      Tuple of (input_path, success)
  """
  input_path, output_path, palette_path, pixel_scale, dither = args

  # Load palette in each process
  palette_img = load_palette(palette_path)

  success = pixelate_and_quantize_with_palette(
    input_path=input_path,
    output_path=output_path,
    palette_img=palette_img,
    pixel_scale=pixel_scale,
    dither=dither,
  )
  return (input_path, success)


def process_all_tiles(
  tiles_dir: Path,
  output_dir: Path,
  palette_path: Path,
  pixel_scale: int = 2,
  dither: bool = True,
  num_workers: int = 8,
  in_place: bool = False,
) -> tuple[int, int]:
  """
  Process all tiles in parallel using the shared palette.

  Args:
      tiles_dir: Directory containing source tiles
      output_dir: Directory to save processed tiles
      palette_path: Path to the palette image
      pixel_scale: Pixelation scale factor
      dither: Whether to apply dithering
      num_workers: Number of parallel workers
      in_place: If True, overwrite original tiles instead of using output_dir

  Returns:
      Tuple of (successful_count, failed_count)
  """
  tile_paths = collect_all_tile_paths(tiles_dir)
  print(f"Found {len(tile_paths)} tiles to process")

  # Prepare task arguments
  tasks = []
  for tile_path in tile_paths:
    # Calculate relative path to preserve directory structure
    rel_path = tile_path.relative_to(tiles_dir)

    if in_place:
      output_path = tile_path
    else:
      output_path = output_dir / rel_path

    tasks.append((tile_path, output_path, palette_path, pixel_scale, dither))

  successful = 0
  failed = 0

  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_single_tile, task): task for task in tasks}

    for i, future in enumerate(as_completed(futures)):
      input_path, success = future.result()
      if success:
        successful += 1
      else:
        failed += 1

      # Progress update every 100 tiles
      if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(tasks)} tiles...")

  return successful, failed


def copy_manifest(tiles_dir: Path, output_dir: Path) -> None:
  """Copy the manifest.json to the output directory."""
  manifest_src = tiles_dir / "manifest.json"
  if manifest_src.exists():
    manifest_dst = output_dir / "manifest.json"
    manifest_dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(manifest_src, manifest_dst)
    print(f"Copied manifest.json to {manifest_dst}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Postprocess tiles with pixelation and unified color quantization."
  )
  parser.add_argument(
    "--tiles-dir",
    type=Path,
    default=DEFAULT_TILES_DIR,
    help=f"Directory containing source tiles (default: {DEFAULT_TILES_DIR})",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help=f"Directory for processed tiles (default: {DEFAULT_OUTPUT_DIR})",
  )
  parser.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use",
  )
  parser.add_argument(
    "--palette-output",
    type=Path,
    default=DEFAULT_PALETTE_PATH,
    help=f"Path to save generated palette (default: {DEFAULT_PALETTE_PATH})",
  )
  parser.add_argument(
    "--build-palette-only",
    action="store_true",
    help="Only build and save the palette, don't process tiles",
  )
  parser.add_argument(
    "-s",
    "--scale",
    type=int,
    default=2,
    help="Pixel scale factor. Higher = blockier (default: 2)",
  )
  parser.add_argument(
    "-c",
    "--colors",
    type=int,
    default=32,
    help="Number of colors in the palette (default: 32)",
  )
  parser.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  parser.add_argument(
    "-w",
    "--workers",
    type=int,
    default=8,
    help="Number of parallel workers (default: 8)",
  )
  parser.add_argument(
    "--sample-tiles",
    type=int,
    default=500,
    help="Number of tiles to sample for palette building (default: 500)",
  )
  parser.add_argument(
    "--in-place",
    action="store_true",
    help="Process tiles in place (overwrite originals)",
  )

  args = parser.parse_args()

  # Determine palette path
  if args.palette:
    palette_path = args.palette
    print(f"Using existing palette: {palette_path}")
  else:
    # Build palette from tiles
    print("Building unified palette from tiles...")
    tile_paths = collect_all_tile_paths(args.tiles_dir)
    print(f"Found {len(tile_paths)} tiles")

    print(f"Sampling colors from {args.sample_tiles} tiles...")
    colors = sample_colors_from_tiles(
      tile_paths,
      sample_size=args.sample_tiles,
      pixels_per_tile=1000,
    )
    print(f"Sampled {len(colors)} colors")

    print(f"Building palette with {args.colors} colors...")
    palette_img = build_unified_palette(colors, num_colors=args.colors)

    palette_path = args.palette_output
    save_palette(palette_img, palette_path)

  if args.build_palette_only:
    print("Palette built. Exiting (--build-palette-only specified).")
    return

  # Process all tiles
  print(f"\nProcessing tiles with scale={args.scale}, dither={not args.no_dither}")
  if args.in_place:
    print("WARNING: Processing in place - original tiles will be overwritten!")
  else:
    print(f"Output directory: {args.output_dir}")

  successful, failed = process_all_tiles(
    tiles_dir=args.tiles_dir,
    output_dir=args.output_dir,
    palette_path=palette_path,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    num_workers=args.workers,
    in_place=args.in_place,
  )

  # Copy manifest to output directory if not in-place
  if not args.in_place:
    copy_manifest(args.tiles_dir, args.output_dir)

  print(f"\nDone! Processed {successful} tiles successfully, {failed} failed.")


if __name__ == "__main__":
  main()
