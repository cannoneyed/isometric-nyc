# Debug map plan
_Exported on 12/12/2025 at 10:06:12 EST from Cursor (2.1.39)_

---

**User**

Implement the plan in @tasks/022_debug_map.md 

---

**Cursor**

I'll implement the debug_map script. Let me first explore the codebase to understand the database structure, generation config, and how quadrants are defined.

Let me explore more of the shared utilities and understand the coordinate calculations:

Let me also look at the visualize_bounds.py script and check how to calculate quadrant corners for the geographic footprint:

Now I have a good understanding. Let me check what packages are available in the project:

Now I'll create the `debug_map.py` script. I'll use matplotlib for plotting (adding it as a dependency) and contextily for fetching map tiles as background:

Now let me create the `debug_map.py` script:

```python
"""
Debug Map - Visualize generated quadrants on a real map of NYC.

This script creates a visualization showing which quadrants have been generated
by overlaying 25% alpha red rectangles on a real map. The isometric projection
is accounted for, so the quadrants appear as parallelograms on the standard map.

Usage:
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)

Output:
  Creates debug_map.png in the generation directory
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pyproj import Transformer

from isometric_nyc.e2e_generation.shared import calculate_offset


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_generated_quadrants(conn: sqlite3.Connection) -> list[tuple[int, int]]:
  """
  Get all quadrants that have been generated (have generation data).

  Returns list of (quadrant_x, quadrant_y) tuples.
  """
  cursor = conn.cursor()
  cursor.execute("""
    SELECT quadrant_x, quadrant_y
    FROM quadrants
    WHERE generation IS NOT NULL
    ORDER BY quadrant_x, quadrant_y
  """)
  return [(row[0], row[1]) for row in cursor.fetchall()]


def calculate_quadrant_corners(
  config: dict, quadrant_x: int, quadrant_y: int
) -> list[tuple[float, float]]:
  """
  Calculate the 4 geographic corners of a quadrant.

  The isometric projection means the quadrant appears as a parallelogram
  in geographic space. Returns corners in order: TL, TR, BR, BL.

  Args:
    config: Generation config dictionary
    quadrant_x: X coordinate of the quadrant
    quadrant_y: Y coordinate of the quadrant

  Returns:
    List of 4 (lat, lng) tuples representing the corners
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant is half a tile in size
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  # Calculate the pixel offsets for the 4 corners of this quadrant
  # Quadrant (0,0) has its bottom-right corner at the seed position
  # So we need to calculate corners relative to the seed
  base_x_px = quadrant_x * quadrant_width_px
  base_y_px = -quadrant_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels (relative to the quadrant's anchor)
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL: left and up from anchor
    (0, quadrant_height_px),  # TR: up from anchor
    (0, 0),  # BR: at anchor
    (-quadrant_width_px, 0),  # BL: left from anchor
  ]

  corners = []
  for dx, dy in corner_offsets:
    shift_x_px = base_x_px + dx
    shift_y_px = base_y_px + dy

    lat, lng = calculate_offset(
      seed_lat,
      seed_lng,
      shift_x_px,
      shift_y_px,
      view_height_meters,
      height_px,
      azimuth,
      elevation,
    )
    corners.append((lat, lng))

  return corners


def create_debug_map(
  generation_dir: Path,
  output_path: Path | None = None,
) -> Path:
  """
  Create a debug map showing generated quadrants overlaid on NYC map.

  Args:
    generation_dir: Path to the generation directory
    output_path: Optional output path (default: generation_dir/debug_map.png)

  Returns:
    Path to the generated image
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"📂 Loaded generation config: {config.get('name', 'unnamed')}")
    print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
    print(f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
          f"elevation={config['camera_elevation_degrees']}°")

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"\n📊 Found {len(generated)} generated quadrants")

    if not generated:
      print("⚠️  No generated quadrants found. Creating empty map.")

    # Calculate corners for all generated quadrants
    print("\n🔲 Calculating quadrant geographic footprints...")
    quadrant_polygons = []
    all_lats = []
    all_lngs = []

    for qx, qy in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners))

      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate bounding box with padding
    if all_lats and all_lngs:
      min_lat, max_lat = min(all_lats), max(all_lats)
      min_lng, max_lng = min(all_lngs), max(all_lngs)

      # Add 10% padding
      lat_range = max_lat - min_lat
      lng_range = max_lng - min_lng
      padding = 0.1

      min_lat -= lat_range * padding
      max_lat += lat_range * padding
      min_lng -= lng_range * padding
      max_lng += lng_range * padding
    else:
      # Default to NYC area around the seed
      seed_lat = config["seed"]["lat"]
      seed_lng = config["seed"]["lng"]
      min_lat = seed_lat - 0.01
      max_lat = seed_lat + 0.01
      min_lng = seed_lng - 0.01
      max_lng = seed_lng + 0.01

    print(f"\n🗺️  Map bounds:")
    print(f"   Lat: {min_lat:.6f} to {max_lat:.6f}")
    print(f"   Lng: {min_lng:.6f} to {max_lng:.6f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))

    # Set up coordinate transformer (WGS84 to Web Mercator)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Transform bounds to Web Mercator for contextily
    x_min, y_min = transformer.transform(min_lng, min_lat)
    x_max, y_max = transformer.transform(max_lng, max_lat)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add quadrant polygons
    print("\n🎨 Drawing quadrant overlays...")
    for qx, qy, corners in quadrant_polygons:
      # Transform corners to Web Mercator
      web_mercator_corners = []
      for lat, lng in corners:
        x, y = transformer.transform(lng, lat)
        web_mercator_corners.append((x, y))

      # Create polygon
      polygon = Polygon(
        web_mercator_corners,
        closed=True,
        facecolor=(1, 0, 0, 0.25),  # 25% alpha red
        edgecolor=(1, 0, 0, 0.5),  # 50% alpha red edge
        linewidth=1,
      )
      ax.add_patch(polygon)

    # Add the seed point marker
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]
    seed_x, seed_y = transformer.transform(seed_lng, seed_lat)
    ax.plot(seed_x, seed_y, 'go', markersize=10, markeredgecolor='black',
            markeredgewidth=2, label='Seed point', zorder=10)

    # Add base map tiles
    print("🌍 Fetching map tiles...")
    try:
      ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        zoom="auto",
      )
    except Exception as e:
      print(f"   ⚠️  Could not fetch tiles: {e}")
      print("   Using plain background instead.")
      ax.set_facecolor('#f0f0f0')

    # Add title and legend
    ax.set_title(
      f"Generated Quadrants: {len(generated)} total\n"
      f"{config.get('name', 'Generation')}",
      fontsize=14,
      fontweight='bold',
    )

    # Create legend
    red_patch = mpatches.Patch(
      color=(1, 0, 0, 0.25),
      edgecolor=(1, 0, 0, 0.5),
      label=f'Generated ({len(generated)} quadrants)',
    )
    seed_marker = plt.Line2D(
      [0], [0], marker='o', color='w',
      markerfacecolor='green', markeredgecolor='black',
      markeredgewidth=2, markersize=10, label='Seed point',
    )
    ax.legend(handles=[red_patch, seed_marker], loc='upper left')

    # Remove axis labels (they're Web Mercator coordinates)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.png"

    # Save figure
    print(f"\n💾 Saving to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Debug map created: {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create a debug map showing generated quadrants overlaid on NYC."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: generation_dir/debug_map.png)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None

  print(f"\n{'=' * 60}")
  print("🗺️  Debug Map Generator")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")

  try:
    create_debug_map(generation_dir, output_path)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

The script ran successfully. Let me fix the warning about the legend patch by updating that line:

```python
"""
Debug Map - Visualize generated quadrants on a real map of NYC.

This script creates a visualization showing which quadrants have been generated
by overlaying 25% alpha red rectangles on a real map. The isometric projection
is accounted for, so the quadrants appear as parallelograms on the standard map.

Usage:
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)

Output:
  Creates debug_map.png in the generation directory
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pyproj import Transformer

from isometric_nyc.e2e_generation.shared import calculate_offset


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_generated_quadrants(conn: sqlite3.Connection) -> list[tuple[int, int]]:
  """
  Get all quadrants that have been generated (have generation data).

  Returns list of (quadrant_x, quadrant_y) tuples.
  """
  cursor = conn.cursor()
  cursor.execute("""
    SELECT quadrant_x, quadrant_y
    FROM quadrants
    WHERE generation IS NOT NULL
    ORDER BY quadrant_x, quadrant_y
  """)
  return [(row[0], row[1]) for row in cursor.fetchall()]


def calculate_quadrant_corners(
  config: dict, quadrant_x: int, quadrant_y: int
) -> list[tuple[float, float]]:
  """
  Calculate the 4 geographic corners of a quadrant.

  The isometric projection means the quadrant appears as a parallelogram
  in geographic space. Returns corners in order: TL, TR, BR, BL.

  Args:
    config: Generation config dictionary
    quadrant_x: X coordinate of the quadrant
    quadrant_y: Y coordinate of the quadrant

  Returns:
    List of 4 (lat, lng) tuples representing the corners
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant is half a tile in size
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  # Calculate the pixel offsets for the 4 corners of this quadrant
  # Quadrant (0,0) has its bottom-right corner at the seed position
  # So we need to calculate corners relative to the seed
  base_x_px = quadrant_x * quadrant_width_px
  base_y_px = -quadrant_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels (relative to the quadrant's anchor)
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL: left and up from anchor
    (0, quadrant_height_px),  # TR: up from anchor
    (0, 0),  # BR: at anchor
    (-quadrant_width_px, 0),  # BL: left from anchor
  ]

  corners = []
  for dx, dy in corner_offsets:
    shift_x_px = base_x_px + dx
    shift_y_px = base_y_px + dy

    lat, lng = calculate_offset(
      seed_lat,
      seed_lng,
      shift_x_px,
      shift_y_px,
      view_height_meters,
      height_px,
      azimuth,
      elevation,
    )
    corners.append((lat, lng))

  return corners


def create_debug_map(
  generation_dir: Path,
  output_path: Path | None = None,
) -> Path:
  """
  Create a debug map showing generated quadrants overlaid on NYC map.

  Args:
    generation_dir: Path to the generation directory
    output_path: Optional output path (default: generation_dir/debug_map.png)

  Returns:
    Path to the generated image
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"📂 Loaded generation config: {config.get('name', 'unnamed')}")
    print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
    print(f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
          f"elevation={config['camera_elevation_degrees']}°")

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"\n📊 Found {len(generated)} generated quadrants")

    if not generated:
      print("⚠️  No generated quadrants found. Creating empty map.")

    # Calculate corners for all generated quadrants
    print("\n🔲 Calculating quadrant geographic footprints...")
    quadrant_polygons = []
    all_lats = []
    all_lngs = []

    for qx, qy in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners))

      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate bounding box with padding
    if all_lats and all_lngs:
      min_lat, max_lat = min(all_lats), max(all_lats)
      min_lng, max_lng = min(all_lngs), max(all_lngs)

      # Add 10% padding
      lat_range = max_lat - min_lat
      lng_range = max_lng - min_lng
      padding = 0.1

      min_lat -= lat_range * padding
      max_lat += lat_range * padding
      min_lng -= lng_range * padding
      max_lng += lng_range * padding
    else:
      # Default to NYC area around the seed
      seed_lat = config["seed"]["lat"]
      seed_lng = config["seed"]["lng"]
      min_lat = seed_lat - 0.01
      max_lat = seed_lat + 0.01
      min_lng = seed_lng - 0.01
      max_lng = seed_lng + 0.01

    print(f"\n🗺️  Map bounds:")
    print(f"   Lat: {min_lat:.6f} to {max_lat:.6f}")
    print(f"   Lng: {min_lng:.6f} to {max_lng:.6f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))

    # Set up coordinate transformer (WGS84 to Web Mercator)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Transform bounds to Web Mercator for contextily
    x_min, y_min = transformer.transform(min_lng, min_lat)
    x_max, y_max = transformer.transform(max_lng, max_lat)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add quadrant polygons
    print("\n🎨 Drawing quadrant overlays...")
    for qx, qy, corners in quadrant_polygons:
      # Transform corners to Web Mercator
      web_mercator_corners = []
      for lat, lng in corners:
        x, y = transformer.transform(lng, lat)
        web_mercator_corners.append((x, y))

      # Create polygon
      polygon = Polygon(
        web_mercator_corners,
        closed=True,
        facecolor=(1, 0, 0, 0.25),  # 25% alpha red
        edgecolor=(1, 0, 0, 0.5),  # 50% alpha red edge
        linewidth=1,
      )
      ax.add_patch(polygon)

    # Add the seed point marker
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]
    seed_x, seed_y = transformer.transform(seed_lng, seed_lat)
    ax.plot(seed_x, seed_y, 'go', markersize=10, markeredgecolor='black',
            markeredgewidth=2, label='Seed point', zorder=10)

    # Add base map tiles
    print("🌍 Fetching map tiles...")
    try:
      ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        zoom="auto",
      )
    except Exception as e:
      print(f"   ⚠️  Could not fetch tiles: {e}")
      print("   Using plain background instead.")
      ax.set_facecolor('#f0f0f0')

    # Add title and legend
    ax.set_title(
      f"Generated Quadrants: {len(generated)} total\n"
      f"{config.get('name', 'Generation')}",
      fontsize=14,
      fontweight='bold',
    )

    # Create legend
    red_patch = mpatches.Patch(
      facecolor=(1, 0, 0, 0.25),
      edgecolor=(1, 0, 0, 0.5),
      label=f'Generated ({len(generated)} quadrants)',
    )
    seed_marker = plt.Line2D(
      [0], [0], marker='o', color='w',
      markerfacecolor='green', markeredgecolor='black',
      markeredgewidth=2, markersize=10, label='Seed point',
    )
    ax.legend(handles=[red_patch, seed_marker], loc='upper left')

    # Remove axis labels (they're Web Mercator coordinates)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.png"

    # Save figure
    print(f"\n💾 Saving to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Debug map created: {output_path}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create a debug map showing generated quadrants overlaid on NYC."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: generation_dir/debug_map.png)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None

  print(f"\n{'=' * 60}")
  print("🗺️  Debug Map Generator")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")

  try:
    create_debug_map(generation_dir, output_path)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())


```

Now let me view the generated debug map to verify it looks correct:

The debug map script is working. The visualization shows:

1. **4050 generated quadrants** overlaid as red rectangles (25% alpha) on a real map of NYC
2. The **seed point** (green marker) located around the Empire State Building area
3. The **isometric projection** is correctly accounted for - notice how the quadrants appear as parallelograms tilted at the camera azimuth angle (-15°), matching the isometric view angle
4. The white gaps visible in the coverage show areas that haven't been generated yet
5. Coverage extends from upper Manhattan down through Midtown, into the Financial District, and into Brooklyn

The script successfully:
- Loads the generation config from `generation_config.json` (stored in database metadata)
- Queries the SQLite database for all quadrants with generation data
- Calculates the geographic footprint of each quadrant accounting for the isometric camera projection
- Overlays the quadrants on a CartoDB basemap using contextily
- Outputs `debug_map.png` to the generation directory

You can run it with:

```bash
uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]
```

Where `generation_dir` defaults to `generations/v01`.

```shellscript
uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]
```

---

**User**

Update README.md to document this functionality

---

**Cursor**

```markdown
# Isometric NYC

This project aims to generate a massive isometric pixel art view of New York
City using the latest and greatest AI tools available.

## Project Structure

```
isometric-nyc/
├── src/
│   ├── isometric_nyc/       # Python tile generation pipeline
│   ├── app/                 # Web viewer (React + Deck.gl)
│   ├── water_shader_demo/   # Water shader demo (React + WebGL)
│   └── web/                 # 3D tiles viewer (Three.js)
├── synthetic_data/          # Training datasets
├── tile_plans/              # Tile generation plans
└── tasks/                   # Task documentation
```

---

## Web Viewer

A high-performance tiled map viewer for exploring the isometric pixel art map.

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **Deck.gl** — WebGL tile rendering with OrthographicView
- **Bun** — Package management

### Running the Web App

```bash
cd src/app

# Install dependencies
bun install

# Start development server
bun run dev
```

The app will open at **http://localhost:3000**

---

## Water Shader Demo

A WebGL shader demo for adding animated water effects to the isometric map
tiles.

### Features

- **Shoreline foam**: Subtle animated foam effect at water edges using a
  distance mask
- **Water ripples**: Gentle darkening ripples across open water areas
- **Adjustable parameters**: Real-time sliders for tweaking all effects
- **Pixel art aesthetic**: Quantized effects that maintain the retro look

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **WebGL** — Custom GLSL shaders for water animation
- **Bun** — Package management

### Running the Water Shader Demo

```bash
cd src/water_shader_demo

# Install dependencies
bun install

# Start development server
bun dev
```

The app will open at **http://localhost:5173** (or next available port)

### How It Works

The shader uses a **distance mask** approach:

- **Black pixels** in the mask = land (pass through unchanged)
- **White pixels** = deep water (receives ripple effects)
- **Gradient transition** = shoreline (receives foam effects)

The mask is a blurred version of the land/water boundary, providing smooth
distance information for wave animation.

### Shader Controls

| Control         | Description                        |
| --------------- | ---------------------------------- |
| Wave Speed      | Animation speed of shoreline waves |
| Wave Frequency  | Density of wave bands              |
| Foam Threshold  | How far from shore foam appears    |
| Water Darkness  | Overall brightness of water pixels |
| Ripple Darkness | Intensity of deep water ripples    |

### Adding Tile Assets

Place tile images and corresponding masks in:

```
src/water_shader_demo/public/
├── tiles/
│   └── 0_0.png          # The tile image
└── masks/
    └── 0_0.png          # Distance mask (black=land, white=water)
```

---

### Adding Tiles

Place your tile images in the `src/app/public/tiles/` directory:

```
src/app/public/tiles/
├── 0/                    # Zoom level 0 = native resolution (max zoom in)
│   ├── 0_0.png          # Tile at x=0, y=0
│   ├── 0_1.png          # Tile at x=0, y=1
│   ├── 1_0.png          # Tile at x=1, y=0
│   └── ...
├── 1/                    # Zoom level 1 = 2× zoomed out
├── 2/                    # Zoom level 2 = 4× zoomed out
└── info.json
```

**Tile naming:** `{zoom_level}/{x}_{y}.png`

**Zoom convention:** | Level | Resolution | Grid Size (20×20 base) |
|-------|------------|------------------------| | 0 | Native (512×512 per tile)
| 20×20 tiles | | 1 | 2× zoomed out | 10×10 tiles | | 2 | 4× zoomed out | 5×5
tiles |

---

## Python Pipeline

### Setup

Create a `.env` file in the root directory with API credentials:

```bash
GOOGLE_MAPS_API_KEY=...
NYC_OPENDATA_APP_TOKEN=...
```

Install dependencies using `uv`:

```bash
uv sync
```

### Scripts

```bash
# Generate tiles for an address
uv run isometric-nyc "350 5th Ave, New York, NY"

# Run the e2e generation viewer
uv run python src/isometric_nyc/e2e_generation/app.py

# Export tiles to the web app
**uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19**
```

### Exporting Tiles for the Web App

The `export_tiles_for_app.py` script exports quadrants from the SQLite database
to the web app's tile directory:

```bash
# Export a 20x20 region, normalizing coordinates so (0,0) is top-left
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19

# Export a specific region (e.g., quadrants 10-29 → tiles 0-19)
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 10,10 --br 29,29

# Export render images instead of generations
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --render

# Custom output directory
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --output-dir ./my-tiles/0

# Dry run to see what would be exported
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --dry-run
```

**Coordinate normalization:** The `--tl` coordinate becomes `0_0.png` in the
output. For example, exporting `--tl 10,15 --br 20,25` produces tiles `0_0.png`
through `10_10.png`.

### Debug Map

The `debug_map.py` script generates a visualization showing generation progress
by overlaying generated quadrants on a real map of NYC.

```bash
# Generate debug map for the default generation directory
uv run python src/isometric_nyc/e2e_generation/debug_map.py

# Generate for a specific generation directory
uv run python src/isometric_nyc/e2e_generation/debug_map.py generations/v01

# Custom output path
uv run python src/isometric_nyc/e2e_generation/debug_map.py generations/v01 -o ./my_debug_map.png
```

**Features:**

- Overlays 25% alpha red rectangles for each generated quadrant
- Shows the seed point (green marker) as the generation origin
- Accounts for the isometric projection — quadrants appear as parallelograms
  matching the camera azimuth angle
- Uses CartoDB basemap tiles for geographic context
- Outputs to `debug_map.png` in the generation directory by default

**Output example:** Shows coverage gaps, generation progress, and geographic
extent of the current generation.

---

## Development

| Task                  | Command                                     |
| --------------------- | ------------------------------------------- |
| Run Python tests      | `uv run pytest`                             |
| Format Python code    | `uv run ruff format .`                      |
| Lint Python code      | `uv run ruff check .`                       |
| Run web app           | `cd src/app && bun run dev`                 |
| Build web app         | `cd src/app && bun run build`               |
| Run water shader demo | `cd src/water_shader_demo && bun dev`       |
| Build water shader    | `cd src/water_shader_demo && bun run build` |

```

The README has been updated with documentation for the debug_map script. The new section includes:

- Basic usage examples
- Command-line options for specifying generation directory and output path
- List of features including the isometric projection handling
- Description of what the output shows

