"""
Export quadrants from the generation database to a PMTiles archive.

Creates a single .pmtiles file containing all tiles at multiple zoom levels,
suitable for efficient serving from static storage or CDN.

Postprocessing:
  By default, tiles are exported with pixelation and color quantization applied.
  A unified color palette is built by sampling ~100 quadrants from the database
  before export, ensuring consistent colors across all tiles.

Zoom levels:
  PMTiles uses TMS-style zoom where z=0 is the entire world.
  We map our internal zoom levels to PMTiles:
  - Our Level 0: Base tiles (512x512 each) -> PMTiles z=maxZoom
  - Our Level 1: 2x2 combined -> PMTiles z=maxZoom-1
  - etc.

Usage:
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py <generation_dir> [options]

Examples:
  # Export ALL quadrants to PMTiles (auto-detect bounds)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01

  # Export with custom output file
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --output tiles.pmtiles

  # Export without postprocessing (raw tiles)
  uv run python src/isometric_nyc/e2e_generation/export_pmtiles.py generations/v01 --no-postprocess
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
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import write as pmtiles_write

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
  if n & (n - 1) == 0:
    return n
  power = 1
  while power < n:
    power *= 2
  return power


def calculate_padded_dimensions(
  width: int, height: int, max_zoom_level: int = MAX_ZOOM_LEVEL
) -> tuple[int, int]:
  """
  Calculate padded grid dimensions that are multiples of 2^max_zoom_level.
  """
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
  """Get the image bytes for a quadrant at position (x, y)."""
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
  """Get the bounding box of all quadrants in the database."""
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
  """Count total and generated quadrants in the specified range."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"
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
  """Sample colors from quadrants in the database to build a representative color set."""
  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    column = "render" if use_render else "generation"

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

    if len(all_coords) > sample_size:
      sampled_coords = random.sample(all_coords, sample_size)
    else:
      sampled_coords = all_coords

    all_colors: list[tuple[int, int, int]] = []

    for x, y in sampled_coords:
      cursor.execute(
        f"SELECT {column} FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (x, y),
      )
      row = cursor.fetchone()
      if not row or not row[0]:
        continue

      try:
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        pixels = list(img.getdata())

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
  """Build a unified palette image from sampled colors."""
  if not colors:
    gray_colors = [(i * 8, i * 8, i * 8) for i in range(num_colors)]
    composite = Image.new("RGB", (num_colors, 1), (0, 0, 0))
    pixels = composite.load()
    for i, color in enumerate(gray_colors):
      pixels[i, 0] = color
    return composite.quantize(colors=num_colors, method=1, dither=0)

  num_pixels = len(colors)
  side = int(num_pixels**0.5) + 1

  composite = Image.new("RGB", (side, side), (0, 0, 0))
  pixels = composite.load()

  for i, color in enumerate(colors):
    x = i % side
    y = i // side
    if y < side:
      pixels[x, y] = color

  palette_img = composite.quantize(colors=num_colors, method=1, dither=0)
  return palette_img


def postprocess_image(
  img: Image.Image,
  palette_img: Image.Image,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> Image.Image:
  """Apply pixelation and color quantization to an image."""
  img = img.convert("RGB")
  original_width, original_height = img.size

  if pixel_scale > 1:
    small_width = original_width // pixel_scale
    small_height = original_height // pixel_scale
    img_small = img.resize((small_width, small_height), resample=Image.NEAREST)
  else:
    img_small = img

  img_quantized = img_small.quantize(
    palette=palette_img,
    dither=1 if dither else 0,
  )
  img_quantized = img_quantized.convert("RGB")

  if pixel_scale > 1:
    final_image = img_quantized.resize(
      (original_width, original_height), resample=Image.NEAREST
    )
  else:
    final_image = img_quantized

  return final_image


# =============================================================================
# PMTiles export functions
# =============================================================================


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG", optimize=True)
  return buffer.getvalue()


def create_black_tile(
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
) -> bytes:
  """Create a black tile (postprocessed if palette provided)."""
  black_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
  if palette_img:
    black_tile = postprocess_image(black_tile, palette_img, pixel_scale, dither)
  return image_to_png_bytes(black_tile)


def load_and_process_tile(
  db_path: Path,
  src_x: int,
  src_y: int,
  use_render: bool,
  palette_img: Image.Image | None,
  pixel_scale: int,
  dither: bool,
  black_tile_bytes: bytes,
) -> bytes:
  """Load a tile from the database and optionally postprocess it."""
  data = get_quadrant_data(db_path, src_x, src_y, use_render=use_render)

  if data is None:
    return black_tile_bytes

  if palette_img:
    try:
      img = Image.open(io.BytesIO(data))
      processed = postprocess_image(img, palette_img, pixel_scale, dither)
      return image_to_png_bytes(processed)
    except Exception as e:
      print(f"Warning: Failed to postprocess ({src_x},{src_y}): {e}")
      return data
  else:
    return data


def combine_tiles_for_zoom(
  base_tiles: dict[tuple[int, int], bytes],
  padded_width: int,
  padded_height: int,
  zoom_level: int,
  black_tile_bytes: bytes,
) -> dict[tuple[int, int], bytes]:
  """
  Combine tiles from zoom level 0 to create tiles for a higher zoom level.

  Args:
      base_tiles: Dict mapping (x, y) to PNG bytes for the current level.
      padded_width: Grid width at level 0.
      padded_height: Grid height at level 0.
      zoom_level: Target zoom level (1-4).
      black_tile_bytes: Bytes for a black tile.

  Returns:
      Dict mapping (x, y) to PNG bytes for the new zoom level.
  """
  scale = 2**zoom_level
  new_width = padded_width // scale
  new_height = padded_height // scale

  result: dict[tuple[int, int], bytes] = {}

  for zy in range(new_height):
    for zx in range(new_width):
      # Create combined image
      combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))

      # Load and combine tiles from the previous level
      for dy in range(scale):
        for dx in range(scale):
          base_x = zx * scale + dx
          base_y = zy * scale + dy

          # Get tile data (from level 0)
          tile_data = base_tiles.get((base_x, base_y), black_tile_bytes)

          try:
            tile_img = Image.open(io.BytesIO(tile_data))
            if tile_img.mode != "RGBA":
              tile_img = tile_img.convert("RGBA")

            # Calculate position in combined image
            sub_size = TILE_SIZE // scale
            sub_x = dx * sub_size
            sub_y = dy * sub_size

            # Resize and paste
            resized = tile_img.resize((sub_size, sub_size), Image.Resampling.LANCZOS)
            combined.paste(resized, (sub_x, sub_y))
          except Exception as e:
            print(f"Warning: Failed to combine tile ({base_x},{base_y}): {e}")

      result[(zx, zy)] = image_to_png_bytes(combined.convert("RGB"))

  return result


def export_to_pmtiles(
  db_path: Path,
  tl: tuple[int, int],
  br: tuple[int, int],
  output_path: Path,
  padded_width: int,
  padded_height: int,
  original_width: int,
  original_height: int,
  use_render: bool = False,
  palette_img: Image.Image | None = None,
  pixel_scale: int = DEFAULT_PIXEL_SCALE,
  dither: bool = True,
  max_zoom: int = MAX_ZOOM_LEVEL,
) -> dict[str, int]:
  """
  Export all tiles to a PMTiles archive.

  Returns:
      Stats dict with counts.
  """
  stats = {
    "exported": 0,
    "missing": 0,
    "padding": 0,
    "zoom_levels": max_zoom + 1,
  }

  # Create black tile for padding/missing
  black_tile_bytes = create_black_tile(palette_img, pixel_scale, dither)

  # First, load all base (level 0) tiles into memory
  print(f"\n📦 Loading base tiles (level 0)...")
  base_tiles: dict[tuple[int, int], bytes] = {}

  for dst_y in range(padded_height):
    for dst_x in range(padded_width):
      # Check if this is a padding tile
      if dst_x >= original_width or dst_y >= original_height:
        base_tiles[(dst_x, dst_y)] = black_tile_bytes
        stats["padding"] += 1
        continue

      # Map to source coordinates
      src_x = tl[0] + dst_x
      src_y = tl[1] + dst_y

      tile_data = load_and_process_tile(
        db_path,
        src_x,
        src_y,
        use_render,
        palette_img,
        pixel_scale,
        dither,
        black_tile_bytes,
      )

      if tile_data == black_tile_bytes:
        stats["missing"] += 1
      else:
        stats["exported"] += 1

      base_tiles[(dst_x, dst_y)] = tile_data

    # Progress
    progress = (dst_y + 1) / padded_height * 100
    print(f"   [{progress:5.1f}%] Row {dst_y} loaded")

  # Generate zoom level tiles
  zoom_tiles: dict[int, dict[tuple[int, int], bytes]] = {0: base_tiles}

  for level in range(1, max_zoom + 1):
    print(f"\n🔍 Generating zoom level {level}...")
    zoom_tiles[level] = combine_tiles_for_zoom(
      base_tiles, padded_width, padded_height, level, black_tile_bytes
    )
    print(f"   Generated {len(zoom_tiles[level])} tiles")

  # Write to PMTiles
  print(f"\n📝 Writing PMTiles archive: {output_path}")

  # Ensure output directory exists
  output_path.parent.mkdir(parents=True, exist_ok=True)

  with pmtiles_write(str(output_path)) as writer:
    # Write tiles from all zoom levels
    # PMTiles uses standard web map tiling where at zoom z, x and y must be in [0, 2^z - 1]
    # We need to calculate the minimum PMTiles z that can fit our grid at each level
    #
    # For a grid of size N, we need z where 2^z >= N
    # z = ceil(log2(N))

    import math

    def min_zoom_for_grid(size: int) -> int:
      """Calculate minimum PMTiles zoom level to fit a grid of given size."""
      if size <= 1:
        return 0
      return math.ceil(math.log2(size))

    total_tiles = sum(len(tiles) for tiles in zoom_tiles.values())
    written = 0

    # Calculate PMTiles zoom for each of our levels
    # Our level 0 = most detail (largest grid), level max_zoom = least detail (smallest grid)
    pmtiles_zoom_map: dict[int, int] = {}
    for our_level in range(max_zoom + 1):
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale
      max_dim = max(level_width, level_height)
      pmtiles_zoom_map[our_level] = min_zoom_for_grid(max_dim)

    # Find the range of PMTiles zoom levels we'll use
    pmtiles_min_z = min(pmtiles_zoom_map.values())
    pmtiles_max_z = max(pmtiles_zoom_map.values())

    print(f"   PMTiles zoom range: {pmtiles_min_z} to {pmtiles_max_z}")

    # Write tiles starting from lowest zoom (least detail) to highest (most detail)
    for our_level in range(max_zoom, -1, -1):
      pmtiles_z = pmtiles_zoom_map[our_level]
      tiles = zoom_tiles[our_level]

      # Calculate grid size at this level
      scale = 2**our_level
      level_width = padded_width // scale
      level_height = padded_height // scale

      print(
        f"   Writing level {our_level} as PMTiles z={pmtiles_z} ({level_width}x{level_height} tiles)"
      )

      # Write tiles in order
      for y in range(level_height):
        for x in range(level_width):
          tile_data = tiles.get((x, y))
          if tile_data:
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_data)
            written += 1

      progress = written / total_tiles * 100
      print(f"   [{progress:5.1f}%] Level {our_level} complete")

    # Create header and metadata
    header = {
      "tile_type": TileType.PNG,
      "tile_compression": Compression.NONE,  # PNGs are already compressed
      "center_zoom": (pmtiles_min_z + pmtiles_max_z) // 2,
      "center_lon": 0,
      "center_lat": 0,
    }

    metadata = {
      "name": "Isometric NYC",
      "description": "Pixel art isometric view of New York City",
      "version": "1.0.0",
      "type": "raster",
      "format": "png",
      "tileSize": TILE_SIZE,
      "gridWidth": padded_width,
      "gridHeight": padded_height,
      "originalWidth": original_width,
      "originalHeight": original_height,
      "maxZoom": max_zoom,
      "pmtilesMinZoom": pmtiles_min_z,
      "pmtilesMaxZoom": pmtiles_max_z,
      "pmtilesZoomMap": pmtiles_zoom_map,
      "generated": datetime.now(timezone.utc).isoformat(),
    }

    writer.finalize(header, metadata)

  stats["total_tiles"] = total_tiles
  return stats


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Export quadrants from the generation database to a PMTiles archive.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Export ALL tiles to PMTiles (auto-detect bounds)
  %(prog)s generations/v01

  # Export with custom output file
  %(prog)s generations/v01 --output my-tiles.pmtiles

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
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Output PMTiles file path (default: src/app/public/tiles.pmtiles)",
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

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  # Default output path
  if args.output:
    output_path = args.output.resolve()
  else:
    project_root = generation_dir.parent
    while project_root != project_root.parent:
      if (project_root / "src" / "app").exists():
        break
      project_root = project_root.parent
    output_path = project_root / "src" / "app" / "public" / "tiles.pmtiles"

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

  # Use provided coordinates or auto-detect
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

  # Calculate dimensions
  orig_width = br[0] - tl[0] + 1
  orig_height = br[1] - tl[1] + 1
  padded_width, padded_height = calculate_padded_dimensions(
    orig_width, orig_height, MAX_ZOOM_LEVEL
  )

  print("📐 Grid dimensions:")
  print(f"   Original: {orig_width}×{orig_height}")
  print(f"   Padded:   {padded_width}×{padded_height} (multiple of {2**MAX_ZOOM_LEVEL})")
  print()

  # Build or load palette for postprocessing
  palette_img: Image.Image | None = None
  if not args.no_postprocess:
    if args.palette:
      print(f"🎨 Loading palette from {args.palette}...")
      palette_img = Image.open(args.palette)
    else:
      print(f"🎨 Building unified palette from {args.sample_quadrants} sampled quadrants...")
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

    print(
      f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={not args.no_dither}"
    )
    print()

  if args.dry_run:
    print("🔍 Dry run - no files will be written")
    print(f"   Would export: {padded_width}×{padded_height} base tiles")
    print(f"   Plus {MAX_ZOOM_LEVEL} zoom levels")
    print(f"   To: {output_path}")
    print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")
    return 0

  # Export to PMTiles
  stats = export_to_pmtiles(
    db_path,
    tl,
    br,
    output_path,
    padded_width,
    padded_height,
    orig_width,
    orig_height,
    use_render=args.render,
    palette_img=palette_img,
    pixel_scale=args.scale,
    dither=not args.no_dither,
    max_zoom=MAX_ZOOM_LEVEL,
  )

  # Print summary
  print()
  print("=" * 50)
  print("✅ PMTiles export complete!")
  print(f"   Output: {output_path}")
  print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
  print(f"   Total tiles: {stats['total_tiles']}")
  print(f"   Base tiles: {stats['exported']} exported, {stats['missing']} missing, {stats['padding']} padding")
  print(f"   Zoom levels: {stats['zoom_levels']} (0-{MAX_ZOOM_LEVEL})")
  print(f"   Grid size: {orig_width}×{orig_height} (padded to {padded_width}×{padded_height})")
  print(f"   Postprocessing: {'enabled' if palette_img else 'disabled'}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

