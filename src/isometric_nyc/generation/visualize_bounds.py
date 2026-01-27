"""
Visualize the bounding box for a full-scale isometric pixel art generation.

This script renders a web view of the specified bounding box at ~2000px width
to preview what a full generation would cover.

Usage:
  uv run python src/isometric_nyc/generation/visualize_bounds.py <generation_dir> --top-left "(x,y)" --bottom-right "(x,y)"
  uv run python src/isometric_nyc/generation/visualize_bounds.py <generation_dir> --top-right "(x,y)" --bottom-left "(x,y)"

Examples:
  # Visualize bounds using top-left and bottom-right
  uv run python src/isometric_nyc/generation/visualize_bounds.py generations/test --top-left "(0,0)" --bottom-right "(10,10)"

  # Visualize bounds using top-right and bottom-left
  uv run python src/isometric_nyc/generation/visualize_bounds.py generations/test --top-right "(10,0)" --bottom-left "(0,10)"

  # Visualize a larger area
  uv run python src/isometric_nyc/generation/visualize_bounds.py generations/test --top-left "(-5,-5)" --bottom-right "(20,20)"
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from isometric_nyc.generation.shared import (
  CHROMIUM_ARGS,
  DEFAULT_WEB_PORT,
  WEB_DIR,
  calculate_offset,
  start_web_server,
)

# Target width for the visualization
TARGET_WIDTH_PX = 2000


def parse_coordinate_tuple(coord_str: str) -> tuple[int, int]:
  """
  Parse a coordinate string into an (x, y) tuple.

  Args:
      coord_str: String in format "(x,y)"

  Returns:
      Tuple of (x, y) integers

  Raises:
      ValueError: If the string format is invalid
  """
  pattern = r"\((-?\d+),\s*(-?\d+)\)"
  match = re.match(pattern, coord_str.strip())

  if not match:
    raise ValueError(
      f"Invalid coordinate format: '{coord_str}'. "
      "Expected format: '(x,y)' e.g. '(0,0)' or '(-5,10)'"
    )

  return (int(match.group(1)), int(match.group(2)))


def load_generation_config(generation_dir: Path) -> dict:
  """Load the generation config from the generation directory."""
  config_path = generation_dir / "generation_config.json"
  if not config_path.exists():
    raise FileNotFoundError(f"Generation config not found: {config_path}")

  with open(config_path) as f:
    return json.load(f)


def compute_bounds_from_corners(
  top_left: tuple[int, int] | None = None,
  top_right: tuple[int, int] | None = None,
  bottom_left: tuple[int, int] | None = None,
  bottom_right: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
  """
  Compute the bounding box (top_left, bottom_right) from any two opposite corners.

  Args:
      top_left: (x, y) of top-left corner
      top_right: (x, y) of top-right corner
      bottom_left: (x, y) of bottom-left corner
      bottom_right: (x, y) of bottom-right corner

  Returns:
      Tuple of (top_left, bottom_right) coordinates

  Raises:
      ValueError: If invalid corner combination is provided
  """
  provided = []
  if top_left is not None:
    provided.append(("top_left", top_left))
  if top_right is not None:
    provided.append(("top_right", top_right))
  if bottom_left is not None:
    provided.append(("bottom_left", bottom_left))
  if bottom_right is not None:
    provided.append(("bottom_right", bottom_right))

  if len(provided) != 2:
    raise ValueError(
      f"Expected exactly 2 corners, got {len(provided)}. "
      "Provide either (--top-left, --bottom-right) or (--top-right, --bottom-left)"
    )

  names = {p[0] for p in provided}

  # Valid combinations: TL+BR or TR+BL
  if names == {"top_left", "bottom_right"}:
    tl = top_left
    br = bottom_right
  elif names == {"top_right", "bottom_left"}:
    # Compute TL and BR from TR and BL
    tr = top_right
    bl = bottom_left
    tl = (bl[0], tr[1])  # left x from BL, top y from TR
    br = (tr[0], bl[1])  # right x from TR, bottom y from BL
  else:
    raise ValueError(
      f"Invalid corner combination: {names}. "
      "Provide either (--top-left, --bottom-right) or (--top-right, --bottom-left)"
    )

  # Validate that TL is actually top-left of BR
  if tl[0] > br[0] or tl[1] > br[1]:
    raise ValueError(
      f"Invalid bounds: computed top-left {tl} must be <= bottom-right {br}. "
      "Check your corner coordinates."
    )

  return tl, br


def calculate_bounds_center(
  config: dict,
  top_left: tuple[int, int],
  bottom_right: tuple[int, int],
) -> tuple[float, float]:
  """
  Calculate the lat/lng center point of the bounding box.

  Args:
      config: Generation config dictionary
      top_left: (x, y) quadrant coordinates of the top-left corner
      bottom_right: (x, y) quadrant coordinates of the bottom-right corner

  Returns:
      Tuple of (lat, lng) for the center of the bounding box
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Calculate pixel offset from seed to the center of the bounding box
  # Quadrant (0,0) has anchor at seed position
  # Center of box is midpoint between top_left and bottom_right
  center_qx = (top_left[0] + bottom_right[0]) / 2.0
  center_qy = (top_left[1] + bottom_right[1]) / 2.0

  # Convert quadrant position to pixel offset
  quadrant_step_x_px = width_px * tile_step
  quadrant_step_y_px = height_px * tile_step

  shift_x_px = center_qx * quadrant_step_x_px
  shift_y_px = -center_qy * quadrant_step_y_px  # Negative because y increases downward

  return calculate_offset(
    seed_lat,
    seed_lng,
    shift_x_px,
    shift_y_px,
    view_height_meters,
    height_px,
    azimuth,
    elevation,
  )


def calculate_view_params(
  config: dict,
  top_left: tuple[int, int],
  bottom_right: tuple[int, int],
  target_width: int = TARGET_WIDTH_PX,
) -> dict:
  """
  Calculate the view parameters to show the full bounding box.

  Args:
      config: Generation config dictionary
      top_left: (x, y) quadrant coordinates of the top-left corner
      bottom_right: (x, y) quadrant coordinates of the bottom-right corner
      target_width: Target width in pixels for the output image

  Returns:
      Dictionary with lat, lon, width, height, view_height, azimuth, elevation
  """
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  tile_step = config.get("tile_step", 0.5)

  # Calculate the size of the bounding box in quadrants
  num_quadrants_x = bottom_right[0] - top_left[0] + 1
  num_quadrants_y = bottom_right[1] - top_left[1] + 1

  # Calculate the box size in pixels at the original zoom level
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  box_width_px = num_quadrants_x * quadrant_width_px
  box_height_px = num_quadrants_y * quadrant_height_px

  # Calculate the aspect ratio of the box
  box_aspect = box_width_px / box_height_px

  # Determine output dimensions based on aspect ratio
  # Target ~2000px on the larger dimension
  if box_aspect >= 1:
    # Wider than tall - constrain by width
    output_width = target_width
    output_height = int(target_width / box_aspect)
  else:
    # Taller than wide - constrain by height
    output_height = target_width
    output_width = int(target_width * box_aspect)

  # Ensure dimensions are even (helps with rendering)
  output_width = (output_width // 2) * 2
  output_height = (output_height // 2) * 2

  # Calculate the scale factor from original to output
  scale_factor = box_width_px / output_width

  # The view_height_meters needs to be scaled proportionally
  # Original: view_height_meters covers height_px pixels of world
  # We want box_height_px (at original zoom) to fit in output_height pixels
  # The world height we need to show = box_height_px * (view_height_meters / height_px)
  # new_view_height = view_height_meters * (box_height_px / height_px)
  new_view_height = view_height_meters * (box_height_px / height_px)

  # Calculate center coordinates
  center_lat, center_lng = calculate_bounds_center(config, top_left, bottom_right)

  return {
    "lat": center_lat,
    "lon": center_lng,
    "width": output_width,
    "height": output_height,
    "view_height": new_view_height,
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "box_width_quadrants": num_quadrants_x,
    "box_height_quadrants": num_quadrants_y,
    "scale_factor": scale_factor,
  }


def render_bounds(
  view_params: dict,
  output_path: Path,
  port: int = DEFAULT_WEB_PORT,
) -> Path:
  """
  Render the bounding box visualization using the web renderer.

  Args:
      view_params: Dictionary with view parameters (lat, lon, width, height, etc.)
      output_path: Path to save the rendered image
      port: Web server port

  Returns:
      Path to the saved image
  """
  # Build URL parameters
  params = {
    "export": "true",
    "lat": view_params["lat"],
    "lon": view_params["lon"],
    "width": view_params["width"],
    "height": view_params["height"],
    "azimuth": view_params["azimuth"],
    "elevation": view_params["elevation"],
    "view_height": view_params["view_height"],
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  print("\n🌐 Rendering bounds visualization...")
  print(f"   URL: {url}")

  with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)

    context = browser.new_context(
      viewport={"width": view_params["width"], "height": view_params["height"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    # Navigate to the page
    page.goto(url, wait_until="networkidle")

    # Wait for tiles to load
    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=120000)
    except Exception as e:
      print(f"   ⚠️  Timeout waiting for tiles: {e}")
      print("   📸 Taking screenshot anyway...")

    # Take screenshot
    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"✅ Rendered to {output_path}")
  return output_path


def main():
  parser = argparse.ArgumentParser(
    description="Visualize the bounding box for a full-scale isometric generation."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing generation_config.json",
  )
  parser.add_argument(
    "--top-left",
    type=str,
    help='Top-left corner of the bounding box in format "(x,y)"',
  )
  parser.add_argument(
    "--bottom-right",
    type=str,
    help='Bottom-right corner of the bounding box in format "(x,y)"',
  )
  parser.add_argument(
    "--top-right",
    type=str,
    help='Top-right corner of the bounding box in format "(x,y)"',
  )
  parser.add_argument(
    "--bottom-left",
    type=str,
    help='Bottom-left corner of the bounding box in format "(x,y)"',
  )
  parser.add_argument(
    "--output",
    type=Path,
    help="Output path for the rendered image (default: generation_dir/bounds_preview.png)",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--target-width",
    type=int,
    default=TARGET_WIDTH_PX,
    help=f"Target width in pixels for the output (default: {TARGET_WIDTH_PX})",
  )

  args = parser.parse_args()

  # Parse coordinate tuples (only parse if provided)
  try:
    tl = parse_coordinate_tuple(args.top_left) if args.top_left else None
    tr = parse_coordinate_tuple(args.top_right) if args.top_right else None
    bl = parse_coordinate_tuple(args.bottom_left) if args.bottom_left else None
    br = parse_coordinate_tuple(args.bottom_right) if args.bottom_right else None
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1

  # Compute bounds from provided corners
  try:
    top_left, bottom_right = compute_bounds_from_corners(
      top_left=tl, top_right=tr, bottom_left=bl, bottom_right=br
    )
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  # Load config
  try:
    config = load_generation_config(generation_dir)
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1

  print(f"\n{'=' * 60}")
  print("🗺️  Bounds Visualization")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")
  print(f"   Top-left: {top_left}")
  print(f"   Bottom-right: {bottom_right}")
  print(f"   Target width: {args.target_width}px")

  # Calculate view parameters
  view_params = calculate_view_params(config, top_left, bottom_right, args.target_width)

  print("\n📐 Bounding Box:")
  print(
    f"   Quadrants: {view_params['box_width_quadrants']} × {view_params['box_height_quadrants']}"
  )
  print(f"   Output size: {view_params['width']} × {view_params['height']} px")
  print(f"   Scale factor: {view_params['scale_factor']:.2f}x")
  print("\n📍 Center:")
  print(f"   Lat: {view_params['lat']:.6f}")
  print(f"   Lng: {view_params['lon']:.6f}")
  print(f"   View height: {view_params['view_height']:.2f} meters")

  # Determine output path
  if args.output:
    output_path = args.output.resolve()
  else:
    output_path = generation_dir / "bounds_preview.png"

  web_server = None

  try:
    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    # Render the bounds
    render_bounds(view_params, output_path, args.port)

    print(f"\n{'=' * 60}")
    print("✅ Bounds visualization complete!")
    print(f"   Output: {output_path}")
    print(f"{'=' * 60}")

    return 0

  except Exception as e:
    print(f"❌ Error: {e}")
    raise

  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())
