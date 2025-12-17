"""
Export quadrants from the generation database to the web app's tile directory.

Exports a rectangular region of quadrants to the web app's public/tiles/ directory,
normalizing coordinates so the top-left of the region becomes (0, 0).
Generates multiple zoom levels for efficient tile loading.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  - Level 0: Base tiles (512x512 each)
  - Level 1: 2x2 base tiles combined into 1 (covers 1024x1024 world units)
  - Level 2: 4x4 base tiles combined into 1 (covers 2048x2048 world units)
  - Level 3: 8x8 base tiles combined into 1 (covers 4096x4096 world units)
  - Level 4: 16x16 base tiles combined into 1 (covers 8192x8192 world units)

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

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py generations/v01 --no-postprocess

  # Customize postprocessing parameters
  uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py generations/v01 --scale 4 --colors 64
"""

import argparse
import io
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # 16x16 tile combining

# Postprocessing defaults
DEFAULT_PIXEL_SCALE = 2
DEFAULT_NUM_COLORS = 32
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000


def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 >= n."""
  if n <= 0:
    return 1
  # If already power of 2, return it
  if n & (n - 1) == 0:
    return n
  # Find next power of 2
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.

  This ensures the grid divides evenly at all zoom levels.

  Args:
      width: Original grid width.
      height: Original grid height.
      max_zoom_level: Maximum zoom level to support.

  Returns:
      (padded_width, padded_height)
  """
  # Grid must be divisible by 2^max_zoom_level for perfect alignment
  alignment = 2**max_zoom_level
  padded_width = ((width + alignment - 1) // alignment) * alignment
  padded_height = ((height + alignment - 1) // alignment) * alignment
  return padded_width, padded_height


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


# =============================================================================
# Postprocessing functions (palette building and color quantization)
# =============================================================================


def sample_colors_from_database(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  use_render: bool = False,
  sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
  pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
  """
  Sample colors from quadrants in the database to build a representative color set.

  Args:
      db_path: Path to the quadrants.db file.
      tl: Top-left coordinate of the export region.
      br: Bottom-right coordinate of the export region.
      use_render: If True, sample from render images; otherwise from generations.
      sample_size: Number of quadrants to sample from.
      pixels_per_quadrant: Number of random pixels to sample from each quadrant.

  Returns:
      List of RGB tuples representing sampled colors.
  """
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

    # Get all quadrant coordinates with data in the range
    cursor.execute(
      f"""
            SELECT quadrant_x, quadrant_y FROM quadrants
            WHERE quadrant_x >= ? AND quadrant_x <= ?
              AND quadrant_y >= ? AND quadrant_y <= ?
              AND {column} IS NOT NULL
            """,
      (tl[0], br[0], tl[1], br[1]),
    )
    all_coords = cursor.fetchall()

    if not all_coords:
      return []

    # Sample a subset of quadrants if we have more than sample_size
    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      # Get the image data for this quadrant
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        # Load image from bytes
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

        # Sample random pixels from this quadrant
        if len(pixels) > pixels_per_quadrant:
          sampled_pixels = random.sample(pixels, pixels_per_quadrant)
        else:
          sampled_pixels = pixels

        all_colors.extend(sampled_pixels)
      except Exception as e:
        print(f"Warning: Could not read quadrant ({x},{y}): {e}")

    return all_colors
  finally:
    conn.close()


def build_unified_palette(
  colors: list[tuple[int, int, int]],
  num_colors: int = DEFAULT_NUM_COLORS,
) -> Image.Image:
  """
  Build a unified palette image from sampled colors.

  Creates a small image with the quantized palette that can be used
  as a reference for applying to other images.

  Args:
      colors: List of RGB tuples.
      num_colors: Target number of colors in the palette.

  Returns:
      A palette image that can be used with Image.quantize().
  """
  if not colors:
    # Return a simple grayscale palette if no colors provided
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

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


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """
  Apply pixelation and color quantization to an image.

  Args:
      img: Source image (will be converted to RGB).
      palette_img: Palette image to use for quantization.
      pixel_scale: Pixelation scale factor (1 = no pixelation).
      dither: Whether to apply dithering.

  Returns:
      Processed image in RGB mode.
  """
  img = img.convert("RGB")
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

  return final_image


def save_palette(palette_img: Image.Image, output_path: Path) -> None:
  """Save the palette image to disk."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  palette_img.save(output_path)
  print(f"📎 Saved palette to {output_path}")


# =============================================================================
# Export functions
# =============================================================================


def export_tiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_dir: Path,
  padded_width: int,
  padded_height: int,
  use_render: bool = False,
  skip_existing: bool = True,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> tuple[int, int, int, int]:
  """
  Export quadrants from the database to the output directory with padding.

  Coordinates are normalized so that tl becomes (0, 0) in the output.
  The grid is padded to padded_width x padded_height with black tiles.

  Args:
      db_path: Path to the quadrants.db file.
      tl: Top-left coordinate (x, y) of the region to export.
      br: Bottom-right coordinate (x, y) of the region to export.
      output_dir: Directory to save tiles (e.g., public/tiles/0/).
      padded_width: Padded grid width (power-of-2 aligned).
      padded_height: Padded grid height (power-of-2 aligned).
      use_render: If True, export render images; otherwise export generations.
      skip_existing: If True, skip tiles that already exist.
      palette_img: Palette image for postprocessing (None to skip postprocessing).
      pixel_scale: Pixelation scale factor for postprocessing.
      dither: Whether to apply dithering during postprocessing.

  Returns:
      Tuple of (exported_count, skipped_count, missing_count, padding_count)
  """
  output_dir.mkdir(parents=True, exist_ok=True)

  data_type = "render" if use_render else "generation"
  postprocess_mode = "with postprocessing" if palette_img else "raw"
  exported = 0
  skipped = 0
  missing = 0
  padding = 0

  # Calculate original grid dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  total = padded_width * padded_height

  print(
    f"📦 Exporting {orig_width}×{orig_height} tiles ({data_type}, {postprocess_mode})"
  )
  print(f"   Padded to: {padded_width}×{padded_height} = {total} tiles")
  print(f"   Source range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  print(f"   Output range: (0,0) to ({padded_width - 1},{padded_height - 1})")
  print(f"   Output dir: {output_dir}")
  if palette_img:
    print(f"   Postprocessing: scale={pixel_scale}, dither={dither}")
  print()

  # Create black tile for padding (postprocessed if using postprocessing)
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  black_tile_bytes = io.BytesIO()
  black_tile.save(black_tile_bytes, format="PNG")
  black_tile_data = black_tile_bytes.getvalue()

  # Iterate through all tiles in the padded grid
  for dst_y in range(padded_height):
    row_exported = 0
    row_missing = 0
    row_padding = 0

    for dst_x in range(padded_width):
      output_path = output_dir / f"{dst_x}_{dst_y}.png"

      # Check if output already exists
      if skip_existing and output_path.exists():
        skipped += 1
        continue

      # Check if this is a padding tile (outside original bounds)
      if dst_x >= orig_width or dst_y >= orig_height:
        # Write black padding tile
        output_path.write_bytes(black_tile_data)
        padding += 1
        row_padding += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      # Get quadrant data from database
      data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

      if data is None:
        # Missing tile - write black
        output_path.write_bytes(black_tile_data)
        missing += 1
        row_missing += 1
        continue

      # Apply postprocessing if palette is provided
      if palette_img:
        try:
          img = Image.open(io.BytesIO(data))
          processed = postprocess_image(img, palette_img, pixel_scale, dither)
          # Save processed image
          output_bytes = io.BytesIO()
          processed.save(output_bytes, format="PNG")
          output_path.write_bytes(output_bytes.getvalue())
        except Exception as e:
          print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
          output_path.write_bytes(data)
      else:
        # Save raw data
        output_path.write_bytes(data)

      exported += 1
      row_exported += 1

    # Print row progress
    progress = (dst_y + 1) / padded_height * 100
    status_parts = [f"Row {dst_y:3d}: {row_exported:3d} exported"]
    if row_missing > 0:
      status_parts.append(f"{row_missing} missing")
    if row_padding > 0:
      status_parts.append(f"{row_padding} padding")
    print(f"   [{progress:5.1f}%] {', '.join(status_parts)}")

  return exported, skipped, missing, padding


def load_tile_image(tile_path: Path) -> Image.Image:
  """
  Load a tile image, or create a black image if it doesn't exist.

  Args:
      tile_path: Path to the tile PNG file.

  Returns:
      PIL Image (RGBA, 512x512).
  """
  if tile_path.exists():
    img = Image.open(tile_path)
    if img.mode != "RGBA":
      img = img.convert("RGBA")
    return img
  else:
    # Return a black tile
    return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))


def generate_zoom_levels(
  base_dir: Path,
  padded_width: int,
  padded_height: int,
  max_zoom: int = MAX_ZOOM_LEVEL,
  skip_existing: bool = True,
) -> dict[int, tuple[int, int, int]]:
  """
  Generate zoomed-out tile levels by combining base tiles.

  Args:
      base_dir: Directory containing level 0 tiles (e.g., public/tiles/0/).
      padded_width: Padded width of the grid at level 0 (power-of-2 aligned).
      padded_height: Padded height of the grid at level 0 (power-of-2 aligned).
      max_zoom: Maximum zoom level to generate (1-4).
      skip_existing: If True, skip tiles that already exist.

  Returns:
      Dict mapping zoom level to (exported, skipped, total) counts.
  """
  tiles_root = base_dir.parent  # tiles/ directory
  results: dict[int, tuple[int, int, int]] = {}

  for zoom_level in range(1, max_zoom + 1):
    scale = 2**zoom_level  # How many base tiles fit in one zoomed tile

    # Calculate grid dimensions at this zoom level
    # Since padded dimensions are power-of-2 aligned, this divides evenly
    zoom_width = padded_width // scale
    zoom_height = padded_height // scale

    zoom_dir = tiles_root / str(zoom_level)
    zoom_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    total = zoom_width * zoom_height

    print(f"\n🔍 Generating zoom level {zoom_level} ({scale}×{scale} combining)")
    print(f"   Grid size: {zoom_width}×{zoom_height} = {total} tiles")

    for zy in range(zoom_height):
      row_exported = 0

      for zx in range(zoom_width):
        output_path = zoom_dir / f"{zx}_{zy}.png"

        if skip_existing and output_path.exists():
          skipped += 1
          continue

        # Create combined image
        combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

        # Load and combine base tiles
        # Each zoomed tile combines scale×scale base tiles
        for dy in range(scale):
          for dx in range(scale):
            base_x = zx * scale + dx
            base_y = zy * scale + dy

            # Load the base tile (or black if missing)
            base_tile_path = base_dir / f"{base_x}_{base_y}.png"
            base_tile = load_tile_image(base_tile_path)

            # Calculate position in combined image
            # Each base tile becomes (TILE_SIZE/scale) pixels
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = base_tile.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))

        # Save combined tile
        combined.save(output_path, "PNG")
        exported += 1
        row_exported += 1

      # Print row progress
      progress = (zy + 1) / zoom_height * 100
      print(f"   [{progress:5.1f}%] Row {zy:3d}: {row_exported:3d} exported")

    results[zoom_level] = (exported, skipped, total)
    print(f"   ✓ Zoom {zoom_level}: {exported} exported, {skipped} skipped")

  return results


def write_manifest(
  output_dir: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  tile_size: int = 512,
  max_zoom_level: int = MAX_ZOOM_LEVEL,
) -> None:
  """
  Write a manifest.json file with grid configuration.

  Args:
      output_dir: Directory containing tiles (e.g., public/tiles/0/).
      padded_width: Padded grid width in tiles (power-of-2 aligned).
      padded_height: Padded grid height in tiles (power-of-2 aligned).
      original_width: Original grid width before padding.
      original_height: Original grid height before padding.
      tile_size: Size of each tile in pixels.
      max_zoom_level: Maximum zoom level generated (0 = base only).
  """
  # Write manifest to parent directory (tiles/ not tiles/0/)
  manifest_path = output_dir.parent / "manifest.json"

  manifest = {
    "gridWidth": padded_width,
    "gridHeight": padded_height,
    "originalWidth": original_width,
    "originalHeight": original_height,
    "tileSize": tile_size,
    "totalTiles": padded_width * padded_height,
    "maxZoomLevel": max_zoom_level,
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

  # Export without postprocessing (raw tiles)
  %(prog)s generations/v01 --no-postprocess

  # Customize postprocessing
  %(prog)s generations/v01 --scale 4 --colors 64 --no-dither
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

  # Postprocessing arguments
  postprocess_group = parser.add_argument_group("postprocessing options")
  postprocess_group.add_argument(
    "--no-postprocess",
    action="store_true",
    help="Disable postprocessing (export raw tiles)",
  )
  postprocess_group.add_argument(
    "-s",
    "--scale",
    type=int,
    default=DEFAULT_PIXEL_SCALE,
    help=f"Pixel scale factor. Higher = blockier (default: {DEFAULT_PIXEL_SCALE})",
  )
  postprocess_group.add_argument(
    "-c",
    "--colors",
    type=int,
    default=DEFAULT_NUM_COLORS,
    help=f"Number of colors in the palette (default: {DEFAULT_NUM_COLORS})",
  )
  postprocess_group.add_argument(
    "--no-dither",
    action="store_true",
    help="Disable dithering for a cleaner look",
  )
  postprocess_group.add_argument(
    "--sample-quadrants",
    type=int,
    default=DEFAULT_SAMPLE_QUADRANTS,
    help=f"Number of quadrants to sample for palette building (default: {DEFAULT_SAMPLE_QUADRANTS})",
  )
  postprocess_group.add_argument(
    "--palette",
    type=Path,
    default=None,
    help="Path to existing palette image to use (skips palette building)",
  )
  postprocess_group.add_argument(
    "--save-palette",
    type=Path,
    default=None,
    help="Path to save the generated palette image",
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
      print(
        "❌ Error: Both --tl and --br must be provided together, or neither for auto-detect"
      )
      return 1
    tl = (bounds[0], bounds[1])
    br = (bounds[2], bounds[3])
    print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
  else:
    tl = args.tl
    br = args.br

  # Validate coordinate range
  if tl[0] > br[0] or tl[1] > br[1]:
    print(
      f"❌ Error: Top-left ({tl[0]},{tl[1]}) must be <= bottom-right ({br[0]},{br[1]})"
    )
    return 1

  # Count available data
  total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
  data_type = "render" if args.render else "generation"
  print(f"   Available {data_type} data: {available}/{total} quadrants")
  print()

  # Calculate original and padded dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Padding grid for zoom level alignment:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(
    f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})"
  )
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      # Load existing palette
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      # Build palette from database
      print(
        f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants..."
      )
      colors = sample_colors_from_database(
        db_path,
        tl,
        br,
        use_render=args.render,
        sample_size=args.sample_quadrants,
        pixels_per_quadrant=DEFAULT_PIXELS_PER_QUADRANT,
      )
      print(f"   Sampled {len(colors)} colors from quadrants")
      print(f"   Quantizing to {args.colors} colors...")
      palette_img = build_unified_palette(colors, num_colors=args.colors)

      # Save palette if requested
      if args.save_palette:
        save_palette(palette_img, args.save_palette)

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(
      f"   Would export: {padded_width}×{padded_height} = {padded_width * padded_height} tiles"
    )
    print(f"   To: {output_dir}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export tiles with padding and postprocessing
  exported, skipped, missing, padding = export_tiles(
    db_path,
    tl,
    br,
    output_dir,
    padded_width,
    padded_height,
    use_render=args.render,
    skip_existing=not args.overwrite,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
  )

  # Generate zoom levels 1-4 (combining tiles)
  print()
  print("=" * 50)
  print("🗺️  Generating zoom levels...")
  zoom_results = generate_zoom_levels(
    output_dir,
    padded_width,
    padded_height,
    max_zoom=MAX_ZOOM_LEVEL,
    skip_existing=not args.overwrite,
  )

  # Write manifest with grid configuration
  write_manifest(
    output_dir,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    max_zoom_level=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ Export complete!")
  print(
    f"   Level 0 (base): {exported} exported, {skipped} skipped, {missing} missing, {padding} padding"
  )
  for level, (exp, skip, total) in zoom_results.items():
    print(f"   Level {level} ({2**level}×{2**level}): {exp} exported, {skip} skipped")
  print(f"   Output: {output_dir.parent}")
  print(
    f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})"
  )
  print(f"   Zoom levels: 0-{MAX_ZOOM_LEVEL}")
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
