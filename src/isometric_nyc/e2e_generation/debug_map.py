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
    print(
      f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
      f"elevation={config['camera_elevation_degrees']}°"
    )

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

    print("\n🗺️  Map bounds:")
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
    ax.plot(
      seed_x,
      seed_y,
      "go",
      markersize=10,
      markeredgecolor="black",
      markeredgewidth=2,
      label="Seed point",
      zorder=10,
    )

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
      ax.set_facecolor("#f0f0f0")

    # Add title and legend
    ax.set_title(
      f"Generated Quadrants: {len(generated)} total\n"
      f"{config.get('name', 'Generation')}",
      fontsize=14,
      fontweight="bold",
    )

    # Create legend
    red_patch = mpatches.Patch(
      facecolor=(1, 0, 0, 0.25),
      edgecolor=(1, 0, 0, 0.5),
      label=f"Generated ({len(generated)} quadrants)",
    )
    seed_marker = plt.Line2D(
      [0],
      [0],
      marker="o",
      color="w",
      markerfacecolor="green",
      markeredgecolor="black",
      markeredgewidth=2,
      markersize=10,
      label="Seed point",
    )
    ax.legend(handles=[red_patch, seed_marker], loc="upper left")

    # Remove axis labels (they're Web Mercator coordinates)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.png"

    # Save figure
    print(f"\n💾 Saving to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
