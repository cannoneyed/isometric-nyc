"""
Export quadrants from the generation database to DZI (Deep Zoom Image) format.

Creates a .dzi descriptor file and a _files directory containing the tile pyramid,
suitable for direct use with OpenSeadragon without any custom tile loading code.

Requirements:
  - libvips must be installed via Homebrew: brew install vips
  - The script automatically configures the library path for macOS

Run with:
  uv run python src/isometric_nyc/generation/export_dzi.py generations/nyc

This script handles:
  - Loading tiles from SQLite database
  - Optional postprocessing (palette quantization, pixelation, bounds clipping)
  - Assembling tiles into a composite image using pyvips (memory-efficient)
  - Generating DZI pyramid with dzsave()
  - Creating a metadata sidecar JSON file for frontend use

Image formats:
  - WebP (default): High-quality lossy (Q=95), smaller files, excellent browser support
  - PNG (--png): Lossless, larger files

Postprocessing:
  By default, tiles are exported without postprocessing to preserve colors.
  Use --postprocess to enable palette quantization (256 colors), which reduces
  file size but can cause color drift.

Bounds clipping:
  Use --bounds to specify a GeoJSON bounds file. Tiles at the edge of the bounds
  will have pixels inside the bounds shown normally and pixels outside blacked out.
"""

import argparse
import io
import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, shape

# Load environment variables from .env file
load_dotenv()

# Set library path for pyvips before import (required for Homebrew on macOS)
_homebrew_lib = "/opt/homebrew/lib"
_dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
if _homebrew_lib not in _dyld_path:
    os.environ["DYLD_LIBRARY_PATH"] = f"{_homebrew_lib}:{_dyld_path}".rstrip(":")

import pyvips

# Constants
TILE_SIZE = 512
MAX_ZOOM_LEVEL = 4  # Matches PMTiles export

# Postprocessing defaults (from export_pmtiles.py)
DEFAULT_PIXEL_SCALE = 1
DEFAULT_NUM_COLORS = 256
DEFAULT_DITHER = False
DEFAULT_SAMPLE_QUADRANTS = 100
DEFAULT_PIXELS_PER_QUADRANT = 1000

# Unfake postprocessing defaults
DEFAULT_UNFAKE_PIXEL_SIZE = 2
DEFAULT_UNFAKE_MAX_COLORS = 128
DEFAULT_UNFAKE_DOMINANT_THRESHOLD = 0.15
DEFAULT_UNFAKE_ALPHA_THRESHOLD = 128


def get_quadrant_bounds(db_path: Path) -> tuple[int, int, int, int] | None:
    """Get the bounding box of all quadrants in the database."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MIN(quadrant_x), MIN(quadrant_y), MAX(quadrant_x), MAX(quadrant_y)
            FROM quadrants
            WHERE generation IS NOT NULL
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


def get_all_quadrant_data_in_range(
    db_path: Path,
    tl: tuple[int, int],
    br: tuple[int, int],
    use_render: bool = False,
) -> dict[tuple[int, int], bytes]:
    """Load all tile data in range with a single query."""
    conn = sqlite3.connect(db_path)
    try:
        column = "render" if use_render else "generation"
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT quadrant_x, quadrant_y, {column}
            FROM quadrants
            WHERE quadrant_x >= ? AND quadrant_x <= ?
              AND quadrant_y >= ? AND quadrant_y <= ?
              AND {column} IS NOT NULL
            """,
            (tl[0], br[0], tl[1], br[1]),
        )
        return {(row[0], row[1]): row[2] for row in cursor.fetchall()}
    finally:
        conn.close()


# =============================================================================
# Bounds clipping (adapted from export_pmtiles.py)
# =============================================================================


def load_bounds_file(bounds_path: Path | str) -> dict[str, Any]:
    """Load a bounds GeoJSON file."""
    from isometric_nyc.generation.bounds import load_bounds

    return load_bounds(bounds_path)


def load_generation_config(generation_dir: Path) -> dict[str, Any]:
    """Load the generation configuration from a generation directory."""
    config_path = generation_dir / "generation_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Generation config not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def extract_polygon_from_geojson(geojson: dict) -> Polygon | None:
    """Extract the first polygon from a GeoJSON FeatureCollection."""
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if features:
            geometry = features[0].get("geometry")
            if geometry:
                return shape(geometry)
    elif geojson.get("type") == "Feature":
        geometry = geojson.get("geometry")
        if geometry:
            return shape(geometry)
    elif geojson.get("type") in ("Polygon", "MultiPolygon"):
        return shape(geojson)
    return None


def latlng_to_quadrant_coords(
    config: dict, lat: float, lng: float
) -> tuple[float, float]:
    """Convert a lat/lng position to quadrant (x, y) coordinates."""
    import math

    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]
    width_px = config["width_px"]
    height_px = config["height_px"]
    view_height_meters = config["view_height_meters"]
    azimuth = config["camera_azimuth_degrees"]
    elevation = config["camera_elevation_degrees"]
    tile_step = config.get("tile_step", 0.5)

    meters_per_pixel = view_height_meters / height_px
    delta_north_meters = (lat - seed_lat) * 111111.0
    delta_east_meters = (lng - seed_lng) * 111111.0 * math.cos(math.radians(seed_lat))

    azimuth_rad = math.radians(azimuth)
    cos_a = math.cos(azimuth_rad)
    sin_a = math.sin(azimuth_rad)

    delta_rot_x = delta_east_meters * cos_a - delta_north_meters * sin_a
    delta_rot_y = delta_east_meters * sin_a + delta_north_meters * cos_a

    elev_rad = math.radians(elevation)
    sin_elev = math.sin(elev_rad)

    shift_right_meters = delta_rot_x
    shift_up_meters = -delta_rot_y * sin_elev

    shift_x_px = shift_right_meters / meters_per_pixel
    shift_y_px = shift_up_meters / meters_per_pixel

    quadrant_step_x_px = width_px * tile_step
    quadrant_step_y_px = height_px * tile_step

    quadrant_x = shift_x_px / quadrant_step_x_px
    quadrant_y = -shift_y_px / quadrant_step_y_px

    return quadrant_x, quadrant_y


def convert_bounds_to_quadrant_coords(
    config: dict, bounds_polygon: Polygon
) -> list[tuple[float, float]]:
    """Convert a bounds polygon from lat/lng to quadrant coordinates."""
    exterior_coords = list(bounds_polygon.exterior.coords)
    quadrant_coords = []
    for lng, lat in exterior_coords:
        qx, qy = latlng_to_quadrant_coords(config, lat, lng)
        quadrant_coords.append((qx, qy))
    return quadrant_coords


def create_bounds_mask_for_tile(
    src_x: int,
    src_y: int,
    bounds_quadrant_coords: list[tuple[float, float]],
    tile_size: int = TILE_SIZE,
) -> Image.Image | None:
    """Create a mask for a tile based on bounds polygon."""
    pixel_coords = []
    for qx, qy in bounds_quadrant_coords:
        px = (qx - src_x) * tile_size
        py = (qy - src_y) * tile_size
        pixel_coords.append((px, py))

    if not pixel_coords:
        return None

    bounds_poly = Polygon(pixel_coords)
    tile_corners = [(0, 0), (tile_size, 0), (tile_size, tile_size), (0, tile_size)]
    tile_poly = Polygon(tile_corners)

    if bounds_poly.contains(tile_poly):
        return None

    if not bounds_poly.intersects(tile_poly):
        return Image.new("L", (tile_size, tile_size), 0)

    mask = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(mask)
    int_coords = [(int(round(x)), int(round(y))) for x, y in pixel_coords]
    if len(int_coords) >= 3:
        draw.polygon(int_coords, fill=255)

    return mask


def apply_bounds_mask(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply a bounds mask to an image."""
    img = img.convert("RGBA")
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    result = Image.composite(img, black, mask)
    return result.convert("RGB")


# =============================================================================
# Postprocessing (adapted from export_pmtiles.py)
# =============================================================================


def sample_colors_from_database(
    db_path: Path,
    tl: tuple[int, int],
    br: tuple[int, int],
    use_render: bool = False,
    sample_size: int = DEFAULT_SAMPLE_QUADRANTS,
    pixels_per_quadrant: int = DEFAULT_PIXELS_PER_QUADRANT,
) -> list[tuple[int, int, int]]:
    """Sample colors from quadrants in the database."""
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
# DZI export functions
# =============================================================================


def process_tile(
    raw_data: bytes | None,
    src_x: int,
    src_y: int,
    palette_img: Image.Image | None,
    pixel_scale: int,
    dither: bool,
    bounds_quadrant_coords: list[tuple[float, float]] | None,
    unfake_settings: dict[str, Any] | None = None,
) -> Image.Image:
    """Process a single tile with optional postprocessing and bounds clipping.

    Args:
        raw_data: Raw image bytes from database
        src_x: Quadrant X coordinate
        src_y: Quadrant Y coordinate
        palette_img: Optional palette for quantization
        pixel_scale: Pixelation scale factor (legacy mode)
        dither: Enable dithering (legacy mode)
        bounds_quadrant_coords: Optional bounds polygon for clipping
        unfake_settings: Optional dict with unfake processing settings:
            - pixel_size: int (default: 2)
            - max_colors: int (default: 128)
            - dominant_threshold: float (default: 0.15)
            - morph_cleanup: bool (default: True)
            - jaggy_cleanup: bool (default: True)
            - alpha_threshold: int (default: 128)
            - fixed_palette: list[tuple[int,int,int]] | None
    """
    if raw_data is None:
        return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))

    try:
        img = Image.open(io.BytesIO(raw_data))

        # Apply postprocessing
        if unfake_settings is not None:
            # Use unfake-style processing
            from isometric_nyc.generation.pixel_art_postprocess import (
                process_tile_unfake,
            )

            img = process_tile_unfake(
                img,
                pixel_size=unfake_settings.get("pixel_size", DEFAULT_UNFAKE_PIXEL_SIZE),
                max_colors=unfake_settings.get("max_colors", DEFAULT_UNFAKE_MAX_COLORS),
                dominant_threshold=unfake_settings.get(
                    "dominant_threshold", DEFAULT_UNFAKE_DOMINANT_THRESHOLD
                ),
                morph_cleanup=unfake_settings.get("morph_cleanup", True),
                jaggy_cleanup=unfake_settings.get("jaggy_cleanup", True),
                alpha_threshold=unfake_settings.get(
                    "alpha_threshold", DEFAULT_UNFAKE_ALPHA_THRESHOLD
                ),
                fixed_palette=unfake_settings.get("fixed_palette"),
                snap_grid=True,
            )
            img = img.convert("RGB")
        elif palette_img:
            # Legacy postprocessing
            img = postprocess_image(img, palette_img, pixel_scale, dither)
        else:
            img = img.convert("RGB")

        # Apply bounds mask
        if bounds_quadrant_coords:
            bounds_mask = create_bounds_mask_for_tile(
                src_x, src_y, bounds_quadrant_coords, TILE_SIZE
            )
            if bounds_mask is not None:
                img = apply_bounds_mask(img, bounds_mask)

        return img
    except Exception as e:
        print(f"Warning: Failed to process tile ({src_x},{src_y}): {e}")
        return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))


def assemble_tiles_to_pyvips(
    raw_tiles: dict[tuple[int, int], bytes],
    tl: tuple[int, int],
    br: tuple[int, int],
    palette_img: Image.Image | None,
    pixel_scale: int,
    dither: bool,
    bounds_quadrant_coords: list[tuple[float, float]] | None,
    unfake_settings: dict[str, Any] | None = None,
    num_workers: int = 8,
) -> pyvips.Image:
    """
    Assemble all tiles into a single pyvips image.

    Uses pyvips.arrayjoin for efficient memory handling.
    Fast path when no postprocessing: loads directly with pyvips (skips PIL).
    When postprocessing is needed, tiles are processed in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1
    total_tiles = width * height

    # Check if we need postprocessing (slow path with PIL)
    needs_postprocessing = (
        palette_img is not None
        or bounds_quadrant_coords is not None
        or unfake_settings is not None
    )

    if needs_postprocessing:
        print(f"  Processing {width}x{height} grid with {num_workers} workers...")
        if bounds_quadrant_coords:
            print(f"  Bounds clipping enabled with {len(bounds_quadrant_coords)} vertices")
    else:
        print(f"  Assembling {width}x{height} grid ({width * TILE_SIZE}x{height * TILE_SIZE} pixels)...")

    # Create black tile for missing data
    black_tile_bytes = None  # Lazy init

    def get_black_tile_bytes() -> bytes:
        nonlocal black_tile_bytes
        if black_tile_bytes is None:
            black_img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
            buf = io.BytesIO()
            black_img.save(buf, format="PNG")
            black_tile_bytes = buf.getvalue()
        return black_tile_bytes

    # Process a single tile (for parallel execution)
    def process_single_tile(coords: tuple[int, int]) -> tuple[tuple[int, int], bytes]:
        x, y = coords
        raw_data = raw_tiles.get((x, y))

        if raw_data is None:
            return (coords, get_black_tile_bytes())

        if needs_postprocessing:
            processed_pil = process_tile(
                raw_data,
                x,
                y,
                palette_img,
                pixel_scale,
                dither,
                bounds_quadrant_coords,
                unfake_settings,
            )
            buf = io.BytesIO()
            processed_pil.save(buf, format="PNG")
            return (coords, buf.getvalue())
        else:
            return (coords, raw_data)

    # Build list of all tile coordinates
    all_coords = [
        (x, y)
        for y in range(tl[1], br[1] + 1)
        for x in range(tl[0], br[0] + 1)
    ]

    # Process tiles in parallel if postprocessing is needed
    processed_tiles: dict[tuple[int, int], bytes] = {}

    if needs_postprocessing:
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_tile, coords): coords for coords in all_coords}
            for future in as_completed(futures):
                coords, tile_bytes = future.result()
                processed_tiles[coords] = tile_bytes
                completed += 1
                # Progress update (every 5%)
                progress = int(completed / total_tiles * 20) * 5
                print(f"\r    [{progress:3d}%] {completed}/{total_tiles} tiles processed", end="", flush=True)
        print()  # Newline after progress
    else:
        # No postprocessing - just use raw tiles directly
        for coords in all_coords:
            _, tile_bytes = process_single_tile(coords)
            processed_tiles[coords] = tile_bytes

    # Assemble into pyvips image
    print("  Assembling tile pyramid...")
    black_tile = pyvips.Image.black(TILE_SIZE, TILE_SIZE, bands=3)
    rows = []

    for y in range(tl[1], br[1] + 1):
        row_tiles = []
        for x in range(tl[0], br[0] + 1):
            tile_bytes = processed_tiles.get((x, y))
            if tile_bytes:
                tile_img = pyvips.Image.new_from_buffer(tile_bytes, "")
            else:
                tile_img = black_tile

            # Ensure RGB (3 bands)
            if tile_img.bands == 4:
                tile_img = tile_img.extract_band(0, n=3)
            elif tile_img.bands == 1:
                tile_img = tile_img.bandjoin([tile_img, tile_img])

            row_tiles.append(tile_img)

        # Join row horizontally
        row_img = pyvips.Image.arrayjoin(row_tiles, across=len(row_tiles))
        rows.append(row_img)

    # Join all rows vertically
    result = pyvips.Image.arrayjoin(rows, across=1)
    return result


def export_to_dzi(
    db_path: Path,
    tl: tuple[int, int],
    br: tuple[int, int],
    output_base: Path,
    use_render: bool = False,
    palette_img: Image.Image | None = None,
    pixel_scale: int = DEFAULT_PIXEL_SCALE,
    dither: bool = False,
    image_format: str = "png",
    webp_quality: int = 85,
    bounds_quadrant_coords: list[tuple[float, float]] | None = None,
    unfake_settings: dict[str, Any] | None = None,
    num_workers: int = 8,
) -> dict[str, Any]:
    """
    Export all tiles to DZI format.

    Args:
        db_path: Path to quadrants.db
        tl: Top-left coordinate (x, y)
        br: Bottom-right coordinate (x, y)
        output_base: Base path for output (without extension)
        use_render: Use render column instead of generation
        palette_img: Optional palette for color quantization
        pixel_scale: Pixelation scale factor
        dither: Enable dithering
        image_format: Output format ("png" or "webp")
        webp_quality: Quality for WebP (0-100)
        bounds_quadrant_coords: Optional bounds polygon for clipping

    Returns:
        Stats dict with counts and timing.
    """
    total_start = time.time()

    # Phase 1: Load all tiles
    print("\n📥 Loading tiles from database...")
    load_start = time.time()
    raw_tiles = get_all_quadrant_data_in_range(db_path, tl, br, use_render)
    load_time = time.time() - load_start
    print(f"   Loaded {len(raw_tiles)} tiles in {load_time:.1f}s")

    # Phase 2: Assemble tiles
    print("\n🔧 Assembling tiles...")
    assemble_start = time.time()
    composite = assemble_tiles_to_pyvips(
        raw_tiles,
        tl,
        br,
        palette_img,
        pixel_scale,
        dither,
        bounds_quadrant_coords,
        unfake_settings,
        num_workers,
    )
    assemble_time = time.time() - assemble_start
    print(f"   Assembled in {assemble_time:.1f}s")
    print(f"   Image size: {composite.width} x {composite.height} pixels")

    # Phase 3: Generate DZI
    print(f"\n📝 Generating DZI pyramid: {output_base}...")
    dzi_start = time.time()

    # Ensure output directory exists
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Set suffix based on format
    if image_format == "webp":
        suffix = f".webp[Q={webp_quality}]"
    else:
        suffix = ".png"

    # Progress tracking for dzsave
    last_percent = [-1]  # Use list to allow mutation in closure

    def on_progress(image: Any, progress: Any) -> None:
        percent = progress.percent
        # Only print on whole percent changes to avoid spam
        if percent != last_percent[0]:
            last_percent[0] = percent
            # Carriage return to overwrite line
            print(f"\r   [{percent:3d}%] Generating pyramid tiles...", end="", flush=True)

    # Enable progress reporting and connect callback
    composite.set_progress(True)
    composite.signal_connect("eval", on_progress)

    composite.dzsave(
        str(output_base),
        tile_size=TILE_SIZE,
        overlap=0,
        suffix=suffix,
        depth="onetile",  # Generate pyramid until image fits in one tile
        background=[0, 0, 0],
        region_shrink="nearest",  # Use nearest-neighbor for pixel art (no color blending)
    )

    # Clear the progress line and print completion
    print()  # Newline after progress
    dzi_time = time.time() - dzi_start
    print(f"   Generated in {dzi_time:.1f}s")

    # Count output files
    dzi_file = Path(f"{output_base}.dzi")
    files_dir = Path(f"{output_base}_files")

    tile_count = 0
    level_count = 0
    if files_dir.exists():
        for level_dir in files_dir.iterdir():
            if level_dir.is_dir():
                level_count += 1
                ext = ".webp" if image_format == "webp" else ".png"
                tile_count += len(list(level_dir.glob(f"*{ext}")))

    total_time = time.time() - total_start

    return {
        "source_tiles": len(raw_tiles),
        "output_tiles": tile_count,
        "zoom_levels": level_count,
        "image_width": composite.width,
        "image_height": composite.height,
        "load_time": load_time,
        "assemble_time": assemble_time,
        "dzi_time": dzi_time,
        "total_time": total_time,
        "dzi_file": str(dzi_file),
        "files_dir": str(files_dir),
    }


def compute_app_defaults(image_width: int, image_height: int) -> dict[str, Any]:
    """
    Compute sensible default view settings for the app.

    Returns x, y centered on the image, and zoom at approximately 4:1
    (each screen pixel shows ~4 image pixels on a typical display).
    """
    import math

    # Center of the image
    x = image_width // 2
    y = image_height // 2

    # Zoom calculation: empirically, zoom = log2(max_dimension) - 8.5 gives ~4:1
    # on a typical 1920px wide display. This formula:
    # - 10752px image → zoom ~5
    # - 50000px image → zoom ~7
    # - 100000px image → zoom ~8
    max_dim = max(image_width, image_height)
    zoom = round(math.log2(max_dim) - 8.5, 2)

    return {"x": x, "y": y, "zoom": zoom}


def create_metadata_sidecar(
    output_path: Path,
    tl: tuple[int, int],
    br: tuple[int, int],
    image_width: int,
    image_height: int,
    image_format: str,
    app_defaults: dict[str, Any] | None = None,
) -> None:
    """
    Create a metadata sidecar JSON file for the DZI.

    This provides information the frontend needs that isn't in the DZI file itself.
    If app_defaults is not provided, sensible defaults are computed automatically.
    """
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1

    # Compute app defaults if not provided
    if app_defaults is None:
        app_defaults = compute_app_defaults(image_width, image_height)

    metadata: dict[str, Any] = {
        "originX": tl[0],
        "originY": tl[1],
        "gridWidth": width,
        "gridHeight": height,
        "tileSize": TILE_SIZE,
        "maxZoom": MAX_ZOOM_LEVEL,
        "imageWidth": image_width,
        "imageHeight": image_height,
        "format": image_format,
        "generated": datetime.now(timezone.utc).isoformat(),
        "appDefaults": app_defaults,
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export quadrants from the generation database to DZI format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export using MAP_ID env var (reads from generations/tiny-nyc, writes to dzi/tiny-nyc)
  MAP_ID=tiny-nyc %(prog)s

  # Export with explicit generation dir (output defaults to dir name)
  %(prog)s generations/tiny-nyc

  # Export with custom output subdirectory
  %(prog)s generations/nyc --output custom-name

  # Export with postprocessing (256-color palette)
  MAP_ID=tiny-nyc %(prog)s --postprocess

  # Export with PNG format (lossless, larger files)
  %(prog)s generations/nyc --png

  # Export with bounds clipping
  %(prog)s generations/nyc --bounds v1.json
        """,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the generation directory containing quadrants.db (default: generations/$MAP_ID)",
    )
    parser.add_argument(
        "--tl",
        type=parse_coordinate,
        default=None,
        metavar="X,Y",
        help="Top-left coordinate (auto-detect if omitted)",
    )
    parser.add_argument(
        "--br",
        type=parse_coordinate,
        default=None,
        metavar="X,Y",
        help="Bottom-right coordinate (auto-detect if omitted)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Export render images instead of generation images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Subdirectory name inside src/app/public/dzi/ (e.g., 'tiny-nyc' saves to src/app/public/dzi/tiny-nyc)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without actually exporting",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Export a small 20x20 test subset centered on (-55, 63) for quick iteration",
    )

    # Bounds clipping
    bounds_group = parser.add_argument_group("bounds clipping options")
    bounds_group.add_argument(
        "--bounds",
        type=str,
        default=None,
        metavar="FILE",
        help="GeoJSON bounds file for clipping (default: bounds.json in generation dir if it exists)",
    )

    # Postprocessing
    postprocess_group = parser.add_argument_group("postprocessing options")
    postprocess_group.add_argument(
        "--postprocess",
        action="store_true",
        help="Enable postprocessing (palette quantization to 256 colors)",
    )
    postprocess_group.add_argument(
        "-s",
        "--scale",
        type=int,
        default=DEFAULT_PIXEL_SCALE,
        help=f"Pixel scale factor (default: {DEFAULT_PIXEL_SCALE})",
    )
    postprocess_group.add_argument(
        "-c",
        "--colors",
        type=int,
        default=DEFAULT_NUM_COLORS,
        help=f"Number of colors in palette (default: {DEFAULT_NUM_COLORS})",
    )
    postprocess_group.add_argument(
        "--dither",
        action="store_true",
        help="Enable dithering",
    )
    postprocess_group.add_argument(
        "--sample-quadrants",
        type=int,
        default=DEFAULT_SAMPLE_QUADRANTS,
        help=f"Quadrants to sample for palette (default: {DEFAULT_SAMPLE_QUADRANTS})",
    )
    postprocess_group.add_argument(
        "--palette",
        type=Path,
        default=None,
        help="Path to existing palette image",
    )

    # Unfake postprocessing (new pixel art style)
    unfake_group = parser.add_argument_group("unfake postprocessing options")
    unfake_group.add_argument(
        "--unfake",
        action="store_true",
        help="Enable unfake-style pixel art postprocessing (grid snap, dominant downscale, cleanup)",
    )
    unfake_group.add_argument(
        "--pixel-size",
        type=int,
        default=DEFAULT_UNFAKE_PIXEL_SIZE,
        help=f"Native pixel size for unfake processing (default: {DEFAULT_UNFAKE_PIXEL_SIZE})",
    )
    unfake_group.add_argument(
        "--unfake-colors",
        type=int,
        default=DEFAULT_UNFAKE_MAX_COLORS,
        help=f"Max colors for unfake processing (default: {DEFAULT_UNFAKE_MAX_COLORS})",
    )
    unfake_group.add_argument(
        "--dominant-threshold",
        type=float,
        default=DEFAULT_UNFAKE_DOMINANT_THRESHOLD,
        help=f"Dominant color threshold (default: {DEFAULT_UNFAKE_DOMINANT_THRESHOLD})",
    )
    unfake_group.add_argument(
        "--morph-cleanup",
        action="store_true",
        default=True,
        help="Enable morphological cleanup (default: enabled)",
    )
    unfake_group.add_argument(
        "--no-morph-cleanup",
        action="store_false",
        dest="morph_cleanup",
        help="Disable morphological cleanup",
    )
    unfake_group.add_argument(
        "--jaggy-cleanup",
        action="store_true",
        default=True,
        help="Enable jaggy edge cleanup (default: enabled)",
    )
    unfake_group.add_argument(
        "--no-jaggy-cleanup",
        action="store_false",
        dest="jaggy_cleanup",
        help="Disable jaggy edge cleanup",
    )
    unfake_group.add_argument(
        "--alpha-threshold",
        type=int,
        default=DEFAULT_UNFAKE_ALPHA_THRESHOLD,
        help=f"Alpha binarization threshold (default: {DEFAULT_UNFAKE_ALPHA_THRESHOLD})",
    )
    unfake_group.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for tile processing (default: 8)",
    )

    # Image format
    format_group = parser.add_argument_group("image format options")
    format_group.add_argument(
        "--png",
        action="store_true",
        help="Use PNG format instead of WebP (larger files, lossless)",
    )
    format_group.add_argument(
        "--webp-quality",
        type=int,
        default=95,
        help="WebP quality (0-100, default: 95)",
    )

    args = parser.parse_args()

    # Determine MAP_ID from environment or generation_dir
    map_id = os.environ.get("MAP_ID")

    # Resolve generation directory
    if args.generation_dir:
        generation_dir = args.generation_dir.resolve()
        # Infer map_id from directory name if not set
        if not map_id:
            map_id = generation_dir.name
    elif map_id:
        # Use MAP_ID to construct generation path
        generation_dir = Path(f"generations/{map_id}").resolve()
    else:
        print("❌ Error: Either provide a generation_dir argument or set MAP_ID env var")
        print("   Example: MAP_ID=tiny-nyc uv run python src/isometric_nyc/generation/export_dzi.py")
        print("   Example: uv run python src/isometric_nyc/generation/export_dzi.py generations/tiny-nyc")
        return 1

    # Find project root
    project_root = Path.cwd()
    while project_root != project_root.parent:
        if (project_root / "src" / "app").exists():
            break
        project_root = project_root.parent

    # Determine output directory
    # --output specifies a subdirectory name inside src/app/public/dzi/
    # Default to map_id (derived from generation_dir name or MAP_ID env var)
    if args.output:
        subdir_name = args.output
    elif args.test:
        subdir_name = "test"
    else:
        subdir_name = map_id

    export_dir = project_root / "src" / "app" / "public" / "dzi" / subdir_name
    output_base = export_dir / "tiles"

    print(f"🗺️  MAP_ID: {map_id}")

    # Validate inputs
    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Load generation config (needed for bounds)
    try:
        config = load_generation_config(generation_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1

    # Load bounds file
    # Priority: --bounds argument > bounds.json in generation dir > no bounds
    bounds_quadrant_coords: list[tuple[float, float]] | None = None
    bounds_path: Path | None = None

    if args.bounds:
        # Explicit bounds file specified
        bounds_path = Path(args.bounds)
    else:
        # Check for bounds.json in generation directory
        default_bounds = generation_dir / "bounds.json"
        if default_bounds.exists():
            bounds_path = default_bounds

    if bounds_path:
        try:
            print(f"📍 Loading bounds from: {bounds_path}")
            bounds_geojson = load_bounds_file(bounds_path)
            bounds_polygon = extract_polygon_from_geojson(bounds_geojson)
            if bounds_polygon is None:
                print("❌ Error: Could not extract polygon from bounds file")
                return 1
            bounds_quadrant_coords = convert_bounds_to_quadrant_coords(config, bounds_polygon)
            print(f"   Bounds polygon has {len(bounds_quadrant_coords)} vertices")
        except Exception as e:
            print(f"❌ Error loading bounds: {e}")
            return 1

    # Get database bounds
    bounds = get_quadrant_bounds(db_path)
    if not bounds:
        print("❌ Error: No quadrants found in database")
        return 1

    print(f"📊 Database bounds: ({bounds[0]},{bounds[1]}) to ({bounds[2]},{bounds[3]})")

    # Use provided coordinates, test subset, or auto-detect
    if args.test:
        # Small 20x20 test subset centered on (-55, 63) - a tile with water
        # that showed splotchy color artifacts during debugging
        center_x, center_y = -55, 63
        half_size = 10
        tl = (center_x - half_size, center_y - half_size)
        br = (center_x + half_size - 1, center_y + half_size - 1)
        print(f"🧪 TEST MODE: Using 20x20 subset centered on ({center_x}, {center_y})")
        print(f"   Range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    elif args.tl is None or args.br is None:
        if args.tl is not None or args.br is not None:
            print("❌ Error: Both --tl and --br must be provided together")
            return 1
        tl = (bounds[0], bounds[1])
        br = (bounds[2], bounds[3])
        print(f"   Auto-detected range: ({tl[0]},{tl[1]}) to ({br[0]},{br[1]})")
    else:
        tl = args.tl
        br = args.br

    # Validate coordinate range
    if tl[0] > br[0] or tl[1] > br[1]:
        print("❌ Error: Invalid coordinate range")
        return 1

    # Count available data
    total, available = count_generated_quadrants(db_path, tl, br, use_render=args.render)
    data_type = "render" if args.render else "generation"
    print(f"   Available {data_type} data: {available}/{total} quadrants")

    # Calculate dimensions
    width = br[0] - tl[0] + 1
    height = br[1] - tl[1] + 1

    print()
    print("📐 Grid dimensions:")
    print(f"   Tiles: {width}×{height}")
    print(f"   Pixels: {width * TILE_SIZE}×{height * TILE_SIZE}")
    print()

    # Build palette and settings for postprocessing
    palette_img: Image.Image | None = None
    unfake_settings: dict[str, Any] | None = None

    if args.unfake:
        # Unfake-style postprocessing
        print("🎮 Unfake postprocessing enabled")

        # Extract fixed palette from palette image if provided
        fixed_palette: list[tuple[int, int, int]] | None = None
        if args.palette:
            print(f"🎨 Loading palette from {args.palette}...")
            palette_img = Image.open(args.palette)
            # Extract colors from the palette image
            from isometric_nyc.generation.pixel_art_postprocess import (
                extract_palette_from_image,
            )

            fixed_palette = extract_palette_from_image(palette_img)
            print(f"   Loaded {len(fixed_palette)} colors from palette")
        else:
            # Build palette from sampled colors
            print(f"🎨 Building unified palette from {args.sample_quadrants} quadrants...")
            colors = sample_colors_from_database(
                db_path,
                tl,
                br,
                use_render=args.render,
                sample_size=args.sample_quadrants,
            )
            print(f"   Sampled {len(colors)} colors")
            print(f"   Quantizing to {args.unfake_colors} colors...")
            palette_img = build_unified_palette(colors, num_colors=args.unfake_colors)
            from isometric_nyc.generation.pixel_art_postprocess import (
                extract_palette_from_image,
            )

            fixed_palette = extract_palette_from_image(palette_img)

        unfake_settings = {
            "pixel_size": args.pixel_size,
            "max_colors": args.unfake_colors,
            "dominant_threshold": args.dominant_threshold,
            "morph_cleanup": args.morph_cleanup,
            "jaggy_cleanup": args.jaggy_cleanup,
            "alpha_threshold": args.alpha_threshold,
            "fixed_palette": fixed_palette,
        }

        print(f"   Settings: pixel_size={args.pixel_size}, colors={args.unfake_colors}")
        print(f"   Threshold: {args.dominant_threshold}, alpha={args.alpha_threshold}")
        print(f"   Cleanup: morph={args.morph_cleanup}, jaggy={args.jaggy_cleanup}")
        print()

    elif args.postprocess:
        # Legacy postprocessing
        if args.palette:
            print(f"🎨 Loading palette from {args.palette}...")
            palette_img = Image.open(args.palette)
        else:
            print(f"🎨 Building unified palette from {args.sample_quadrants} quadrants...")
            colors = sample_colors_from_database(
                db_path,
                tl,
                br,
                use_render=args.render,
                sample_size=args.sample_quadrants,
            )
            print(f"   Sampled {len(colors)} colors")
            print(f"   Quantizing to {args.colors} colors...")
            palette_img = build_unified_palette(colors, num_colors=args.colors)

        print(f"   Postprocessing: scale={args.scale}, colors={args.colors}, dither={args.dither}")
        print()

    # Determine image format (WebP is default, use --png for lossless)
    image_format = "png" if args.png else "webp"
    print(f"🖼️  Output format: {image_format.upper()}")
    if image_format == "webp":
        print(f"   WebP quality: {args.webp_quality}")
    print()

    if bounds_quadrant_coords:
        print("✂️  Bounds clipping: enabled")
        print()

    print(f"📂 Export directory: {export_dir}")
    print()

    if args.dry_run:
        print("🔍 Dry run - no files will be written")
        print(f"   Would export: {width}×{height} tiles")
        print(f"   To: {export_dir}/")
        print(f"       tiles.dzi")
        print(f"       tiles_files/")
        print(f"       metadata.json")
        return 0

    # Export to DZI
    stats = export_to_dzi(
        db_path,
        tl,
        br,
        output_base,
        use_render=args.render,
        palette_img=palette_img if not args.unfake else None,  # Don't use legacy palette with unfake
        pixel_scale=args.scale,
        dither=args.dither,
        image_format=image_format,
        webp_quality=args.webp_quality,
        bounds_quadrant_coords=bounds_quadrant_coords,
        unfake_settings=unfake_settings,
        num_workers=args.workers,
    )

    # Create metadata sidecar in the export directory
    metadata_path = export_dir / "metadata.json"
    print(f"\n📋 Creating metadata sidecar: {metadata_path}")

    # Use app_defaults from config if provided, otherwise compute automatically
    app_defaults = config.get("app_defaults")
    if app_defaults:
        print(f"   Using app_defaults from config: x={app_defaults.get('x')}, y={app_defaults.get('y')}, zoom={app_defaults.get('zoom')}")
    else:
        app_defaults = compute_app_defaults(stats["image_width"], stats["image_height"])
        print(f"   Computed app_defaults: x={app_defaults['x']}, y={app_defaults['y']}, zoom={app_defaults['zoom']}")

    create_metadata_sidecar(
        metadata_path, tl, br, stats["image_width"], stats["image_height"], image_format, app_defaults
    )

    # Print summary
    print()
    print("=" * 60)
    print("✅ DZI export complete!")
    print(f"   DZI file: {stats['dzi_file']}")
    print(f"   Tiles directory: {stats['files_dir']}")
    print(f"   Metadata: {metadata_path}")
    print()
    print(f"   Source tiles: {stats['source_tiles']}")
    print(f"   Output tiles: {stats['output_tiles']}")
    print(f"   Zoom levels: {stats['zoom_levels']}")
    print(f"   Image size: {stats['image_width']} × {stats['image_height']} pixels")
    print()
    print("⏱️  Performance:")
    print(f"   Database load: {stats['load_time']:.1f}s")
    print(f"   Tile assembly: {stats['assemble_time']:.1f}s")
    print(f"   DZI generation: {stats['dzi_time']:.1f}s")
    print(f"   Total time: {stats['total_time']:.1f}s ({stats['total_time']/60:.1f} minutes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
