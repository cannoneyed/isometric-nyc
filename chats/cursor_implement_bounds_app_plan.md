# Implement bounds app plan
_Exported on 12/8/2025 at 17:27:04 EST from Cursor (2.1.39)_

---

**User**

Implement the plan in @tasks/014_bounds_app.md 

---

**Cursor**

I'll implement this task. Let me first explore the codebase to understand the existing structure, configuration format, and rendering mechanisms.

Now I have a good understanding of the codebase. Let me create the `visualize_bounds.py` script:

```python
"""
Visualize the bounding box for a full-scale isometric pixel art generation.

This script renders a web view of the specified bounding box at ~2000px width
to preview what a full generation would cover.

Usage:
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-left "(x,y)" --bottom-right "(x,y)"

Examples:
  # Visualize bounds from quadrant (0,0) to (10,10)
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(0,0)" --bottom-right "(10,10)"

  # Visualize a larger area
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(-5,-5)" --bottom-right "(20,20)"
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
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
    # Original: view_height_meters covers height_px pixels
    # We want the box_height_px to fit in output_height pixels
    # So: new_view_height = view_height_meters * (box_height_px / height_px) * (height_px / output_height)
    #                     = view_height_meters * (box_height_px / output_height)
    new_view_height = view_height_meters * (box_height_px / output_height)

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

    print(f"\n🌐 Rendering bounds visualization...")
    print(f"   URL: {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--enable-webgl",
                "--use-gl=angle",
                "--ignore-gpu-blocklist",
            ],
        )

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
        required=True,
        help='Top-left corner of the bounding box in format "(x,y)"',
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help='Bottom-right corner of the bounding box in format "(x,y)"',
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

    # Parse coordinate tuples
    try:
        top_left = parse_coordinate_tuple(args.top_left)
        bottom_right = parse_coordinate_tuple(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounds
    if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
        print(f"❌ Error: Invalid bounds - top-left {top_left} must be <= bottom-right {bottom_right}")
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

    print(f"\n📐 Bounding Box:")
    print(f"   Quadrants: {view_params['box_width_quadrants']} × {view_params['box_height_quadrants']}")
    print(f"   Output size: {view_params['width']} × {view_params['height']} px")
    print(f"   Scale factor: {view_params['scale_factor']:.2f}x")
    print(f"\n📍 Center:")
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

```

Let me verify the script runs correctly by checking if we can at least get the help output:

The script loads correctly. Let me test it with the test generation directory:

The script executed successfully. Let me verify the output image:

The image was created at 2000×2000 pixels. Let me view the rendered image to verify it shows the correct area:

Excellent! The visualization shows a beautiful isometric render of NYC buildings. Let me also test with a larger bounding box to ensure the scaling works correctly:

The implementation is complete and working. Here's a summary of what was created:

## Summary

I created `/Users/andycoenen/cannoneyed/isometric-nyc/src/isometric_nyc/e2e_generation/visualize_bounds.py` that:

### Parameters
- **`generation_dir`** - Directory containing `generation_config.json`
- **`--top-left "(x,y)"`** - Top-left quadrant coordinate of the bounding box
- **`--bottom-right "(x,y)"`** - Bottom-right quadrant coordinate of the bounding box
- **`--output`** (optional) - Custom output path (default: `bounds_preview.png` in generation dir)
- **`--target-width`** (optional) - Target width in pixels (default: 2000)
- **`--port`** (optional) - Web server port (default: 5173)
- **`--no-start-server`** - Skip auto-starting the web server

### Features
- Reads generation config (seed center, azimuth, elevation, etc.) from `generation_config.json`
- Calculates the appropriate lat/lng center and view height to show the full bounding box
- Automatically maintains aspect ratio based on the selected quadrant bounds
- Renders the preview at ~2000px width using the web renderer with Playwright
- Outputs dimension information, scale factor, and center coordinates

### Usage Examples

```bash
# Visualize a 4×4 quadrant area
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation --top-left "(0,0)" --bottom-right "(3,3)"

# Visualize a larger area with custom output
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation --top-left "(-2,-2)" --bottom-right "(10,8)" --output bounds_large.png
```

```shellscript
# Visualize a 4×4 quadrant area
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation --top-left "(0,0)" --bottom-right "(3,3)"

# Visualize a larger area with custom output
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation --top-left "(-2,-2)" --bottom-right "(10,8)" --output bounds_large.png
```

---

**User**

hmm.... it doesn't really look like the math is working correctly - e.g. when I run a with (-20,-20) and (60,60) i get a view that doesn't any longer seem to include Manhattan though Empire State Building is 0,0

---

**Cursor**

```python
center_qx = (top_left[0] + bottom_right[0]) / 2.0
center_qy = (top_left[1] + bottom_right[1]) / 2.0
```

```python
shift_y_px = -quadrant_y * quadrant_step_y_px  # Negative because y increases downward
```

```python
shift_y_px = -center_qy * quadrant_step_y_px  # Negative because y increases downward
```

Let me debug this. I'll run the script with your parameters and check the math:

```python
new_view_height = view_height_meters * (box_height_px / height_px)
```

```python
new_view_height = view_height_meters * (box_height_px / output_height)  # What I had
```

I found the bug! My `view_height` calculation is wrong. I'm using `output_height` when I should be using the original config's `height_px`. Let me fix it:

```python
"""
Visualize the bounding box for a full-scale isometric pixel art generation.

This script renders a web view of the specified bounding box at ~2000px width
to preview what a full generation would cover.

Usage:
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-left "(x,y)" --bottom-right "(x,y)"

Examples:
  # Visualize bounds from quadrant (0,0) to (10,10)
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(0,0)" --bottom-right "(10,10)"

  # Visualize a larger area
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(-5,-5)" --bottom-right "(20,20)"
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
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
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

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
    required=True,
    help='Top-left corner of the bounding box in format "(x,y)"',
  )
  parser.add_argument(
    "--bottom-right",
    type=str,
    required=True,
    help='Bottom-right corner of the bounding box in format "(x,y)"',
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

  # Parse coordinate tuples
  try:
    top_left = parse_coordinate_tuple(args.top_left)
    bottom_right = parse_coordinate_tuple(args.bottom_right)
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1

  # Validate bounds
  if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
    print(
      f"❌ Error: Invalid bounds - top-left {top_left} must be <= bottom-right {bottom_right}"
    )
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

```

Now let me test with your parameters again:

The view_height is now correctly calculated at 12150m (was 6220m before). Let me check the result:

Now Manhattan is clearly visible in the upper-left area! The fix was correcting the `view_height` calculation formula.

The issue was that I was dividing `box_height_px` by `output_height` instead of by `height_px` (the original config's height). This was causing the view to be zoomed in about 2x more than it should be.

Now for your `(-20,-20)` to `(60,60)` bounds:
- The center is correctly at quadrant `(20, 20)` - which is why Manhattan appears in the upper-left (since Empire State Building at `(0,0)` is offset from the center)
- You can now see Manhattan, Brooklyn, Queens, and parts of the Bronx

If you want the Empire State Building (0,0) to be more centered, you could use symmetric bounds like `(-40,-40)` to `(40,40)` instead.

---

**User**

OK - now let's add two more parameters "top-right" and "bottom-left" so that we can make a rectangle

---

**Cursor**

I'll add `--top-right` and `--bottom-left` parameters so you can specify the rectangle using any two opposite corners:

```python
"""
Visualize the bounding box for a full-scale isometric pixel art generation.

This script renders a web view of the specified bounding box at ~2000px width
to preview what a full generation would cover.

Usage:
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-left "(x,y)" --bottom-right "(x,y)"
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-right "(x,y)" --bottom-left "(x,y)"

Examples:
  # Visualize bounds using top-left and bottom-right
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(0,0)" --bottom-right "(10,10)"

  # Visualize bounds using top-right and bottom-left
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-right "(10,0)" --bottom-left "(0,10)"

  # Visualize a larger area
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(-5,-5)" --bottom-right "(20,20)"
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
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
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

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
    required=True,
    help='Top-left corner of the bounding box in format "(x,y)"',
  )
  parser.add_argument(
    "--bottom-right",
    type=str,
    required=True,
    help='Bottom-right corner of the bounding box in format "(x,y)"',
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

  # Parse coordinate tuples
  try:
    top_left = parse_coordinate_tuple(args.top_left)
    bottom_right = parse_coordinate_tuple(args.bottom_right)
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1

  # Validate bounds
  if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
    print(
      f"❌ Error: Invalid bounds - top-left {top_left} must be <= bottom-right {bottom_right}"
    )
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

```

```python
"""
Visualize the bounding box for a full-scale isometric pixel art generation.

This script renders a web view of the specified bounding box at ~2000px width
to preview what a full generation would cover.

Usage:
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-left "(x,y)" --bottom-right "(x,y)"
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-right "(x,y)" --bottom-left "(x,y)"

Examples:
  # Visualize bounds using top-left and bottom-right
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(0,0)" --bottom-right "(10,10)"

  # Visualize bounds using top-right and bottom-left
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-right "(10,0)" --bottom-left "(0,10)"

  # Visualize a larger area
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(-5,-5)" --bottom-right "(20,20)"
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
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
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

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
    required=True,
    help='Top-left corner of the bounding box in format "(x,y)"',
  )
  parser.add_argument(
    "--bottom-right",
    type=str,
    required=True,
    help='Bottom-right corner of the bounding box in format "(x,y)"',
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

  # Parse coordinate tuples
  try:
    top_left = parse_coordinate_tuple(args.top_left)
    bottom_right = parse_coordinate_tuple(args.bottom_right)
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1

  # Validate bounds
  if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
    print(
      f"❌ Error: Invalid bounds - top-left {top_left} must be <= bottom-right {bottom_right}"
    )
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

```

```python
"""
Visualize the bounding box for a full-scale isometric pixel art generation.

This script renders a web view of the specified bounding box at ~2000px width
to preview what a full generation would cover.

Usage:
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-left "(x,y)" --bottom-right "(x,y)"
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py <generation_dir> --top-right "(x,y)" --bottom-left "(x,y)"

Examples:
  # Visualize bounds using top-left and bottom-right
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(0,0)" --bottom-right "(10,10)"

  # Visualize bounds using top-right and bottom-left
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-right "(10,0)" --bottom-left "(0,10)"

  # Visualize a larger area
  uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test --top-left "(-5,-5)" --bottom-right "(20,20)"
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
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
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

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

```

Let me test the updated script:

Now let me test using the new `--top-right` and `--bottom-left` parameters:

Let me also test error handling for invalid combinations:

Done! The script now supports four corner parameters:

- **`--top-left "(x,y)"`** + **`--bottom-right "(x,y)"`** (original)
- **`--top-right "(x,y)"`** + **`--bottom-left "(x,y)"`** (new)

You must provide exactly two opposite corners. The script will compute the bounding box from whichever pair you provide.

**Example usage:**

```bash
# Using top-left and bottom-right (original way)
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation \
  --top-left "(-20,-20)" --bottom-right "(60,60)"

# Using top-right and bottom-left (new way)
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation \
  --top-right "(60,-20)" --bottom-left "(-20,60)"
```

Both commands above would produce the same result since they define the same rectangle.

```shellscript
# Using top-left and bottom-right (original way)
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation \
  --top-left "(-20,-20)" --bottom-right "(60,60)"

# Using top-right and bottom-left (new way)
uv run python src/isometric_nyc/e2e_generation/visualize_bounds.py generations/test_generation \
  --top-right "(60,-20)" --bottom-left "(-20,60)"
```

