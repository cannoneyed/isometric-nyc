"""
Debug visualization for parallel map layer generation.

This script creates an interactive web visualization showing the generation plan
overlayed on a real map. Each generation block is color-coded by step:
- Red: Step 1 (2x2 tiles)
- Green: Step 2 (1x2 and 2x1 strips)
- Blue: Step 3 (1x1 corners)

Usage:
  uv run python src/isometric_nyc/generation/debug_generate_full_map_layer.py \
    --layer-dir layers/snow

  # With custom output path:
  uv run python src/isometric_nyc/generation/debug_generate_full_map_layer.py \
    --layer-dir layers/snow \
    --output debug_layer_plan.html
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.generation.bounds import load_bounds
from isometric_nyc.generation.shared import calculate_offset


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def load_layer_config(layer_dir: Path) -> dict:
  """Load the layer configuration."""
  config_path = layer_dir / "layer_config.json"
  if not config_path.exists():
    raise FileNotFoundError(f"Layer config not found: {config_path}")
  with open(config_path) as f:
    return json.load(f)


def get_generation_plan(layer_dir: Path) -> list[dict]:
  """Load all items from the generation plan."""
  db_path = layer_dir / "progress.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Progress database not found: {db_path}")

  conn = sqlite3.connect(db_path)
  try:
    cursor = conn.cursor()
    cursor.execute("""
            SELECT id, step, block_type, top_left_x, top_left_y, width, height, status
            FROM generation_plan
            ORDER BY step, id
        """)
    items = []
    for row in cursor.fetchall():
      items.append(
        {
          "id": row[0],
          "step": row[1],
          "block_type": row[2],
          "top_left_x": row[3],
          "top_left_y": row[4],
          "width": row[5],
          "height": row[6],
          "status": row[7],
        }
      )
    return items
  finally:
    conn.close()


def calculate_block_corners(
  config: dict,
  top_left_x: int,
  top_left_y: int,
  width: int,
  height: int,
) -> list[tuple[float, float]]:
  """
  Calculate the geographic corners of a block.

  Returns corners in order: TL, TR, BR, BL for Leaflet polygon.
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

  # Block dimensions in pixels
  block_width_px = width * quadrant_width_px
  block_height_px = height * quadrant_height_px

  # Calculate the pixel offsets for the block corners
  # The block's top-left quadrant has its bottom-right at a position
  # derived from the quadrant coordinates
  base_x_px = top_left_x * quadrant_width_px
  base_y_px = -top_left_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels relative to the top-left quadrant's anchor
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL of block
    (-quadrant_width_px + block_width_px, quadrant_height_px),  # TR of block
    (-quadrant_width_px + block_width_px, quadrant_height_px - block_height_px),  # BR
    (-quadrant_width_px, quadrant_height_px - block_height_px),  # BL
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


def generate_html(
  layer_config: dict,
  source_config: dict,
  plan_items: list[dict],
  seed_lat: float,
  seed_lng: float,
  center_lat: float,
  center_lng: float,
  boundary_geojson: dict | None = None,
) -> str:
  """Generate the HTML content for the debug map."""

  # Process plan items into polygon data
  polygons_data = []
  for item in plan_items:
    corners = calculate_block_corners(
      source_config,
      item["top_left_x"],
      item["top_left_y"],
      item["width"],
      item["height"],
    )
    polygons_data.append(
      {
        "id": item["id"],
        "step": item["step"],
        "block_type": item["block_type"],
        "x": item["top_left_x"],
        "y": item["top_left_y"],
        "width": item["width"],
        "height": item["height"],
        "status": item["status"],
        "corners": [[lat, lng] for lat, lng in corners],
      }
    )

  polygons_json = json.dumps(polygons_data)

  # Count items by step and status
  step_counts = {
    1: {"pending": 0, "complete": 0, "error": 0},
    2: {"pending": 0, "complete": 0, "error": 0},
    3: {"pending": 0, "complete": 0, "error": 0},
  }
  for item in plan_items:
    status = item["status"] if item["status"] in ["complete", "error"] else "pending"
    step_counts[item["step"]][status] += 1

  # Boundary GeoJSON for the map
  if boundary_geojson is None:
    boundary_geojson = load_bounds()
  nyc_boundary_json = json.dumps(boundary_geojson)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Layer Plan - {layer_config.get("name", "Generation")}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600&display=swap');

    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body {{
      font-family: 'Space Grotesk', sans-serif;
      background: #0a0a0f;
    }}

    #map {{
      width: 100vw;
      height: 100vh;
    }}

    .info-panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      color: #fff;
      min-width: 200px;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .info-panel h3 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 8px;
    }}

    .coords {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 20px;
      font-weight: 600;
      color: #fff;
      display: flex;
      gap: 8px;
      align-items: baseline;
    }}

    .coords .label {{
      font-size: 12px;
      color: rgba(255, 255, 255, 0.4);
    }}

    .block-info {{
      margin-top: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: rgba(255, 255, 255, 0.7);
    }}

    .no-hover {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: rgba(255, 255, 255, 0.3);
      font-style: italic;
    }}

    .stats-panel {{
      position: fixed;
      bottom: 16px;
      left: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .stats-panel .title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 8px;
    }}

    .stats-panel .stat-row {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: rgba(255, 255, 255, 0.7);
      margin: 4px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .stats-panel .stat-color {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}

    .legend {{
      position: fixed;
      bottom: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
    }}

    .legend-swatch {{
      width: 16px;
      height: 16px;
      border-radius: 4px;
    }}

    .legend-swatch.step1 {{
      background: rgba(255, 99, 71, 0.5);
      border: 2px solid rgba(255, 99, 71, 0.8);
    }}

    .legend-swatch.step2 {{
      background: rgba(50, 205, 50, 0.5);
      border: 2px solid rgba(50, 205, 50, 0.8);
    }}

    .legend-swatch.step3 {{
      background: rgba(30, 144, 255, 0.5);
      border: 2px solid rgba(30, 144, 255, 0.8);
    }}

    .legend-swatch.complete {{
      background: rgba(147, 51, 234, 0.5);
      border: 2px solid rgba(147, 51, 234, 0.8);
    }}

    .legend-swatch.seed {{
      background: #10b981;
      border: 2px solid #fff;
      border-radius: 50%;
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="info-panel">
    <h3>Block Info</h3>
    <div id="block-display" class="no-hover">Hover over a block</div>
  </div>

  <div class="stats-panel">
    <div class="title">{layer_config.get("name", "Layer Plan")}</div>
    <div class="stat-row">
      <div class="stat-color" style="background: rgba(255, 99, 71, 0.7);"></div>
      <span>Step 1 (2x2): {step_counts[1]["pending"]} pending, {step_counts[1]["complete"]} complete</span>
    </div>
    <div class="stat-row">
      <div class="stat-color" style="background: rgba(50, 205, 50, 0.7);"></div>
      <span>Step 2 (strips): {step_counts[2]["pending"]} pending, {step_counts[2]["complete"]} complete</span>
    </div>
    <div class="stat-row">
      <div class="stat-color" style="background: rgba(30, 144, 255, 0.7);"></div>
      <span>Step 3 (1x1): {step_counts[3]["pending"]} pending, {step_counts[3]["complete"]} complete</span>
    </div>
    <div class="stat-row" style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">
      <span>Total: {len(plan_items)} blocks</span>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch step1"></div>
      <span>Step 1: 2x2 tiles</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch step2"></div>
      <span>Step 2: 1x2/2x1 strips</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch step3"></div>
      <span>Step 3: 1x1 corners</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch complete"></div>
      <span>Generated (complete)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch seed"></div>
      <span>Seed Point</span>
    </div>
  </div>

  <script>
    const polygons = {polygons_json};
    const nycBoundary = {nyc_boundary_json};
    const seedLat = {seed_lat};
    const seedLng = {seed_lng};
    const centerLat = {center_lat};
    const centerLng = {center_lng};

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView([centerLat, centerLng], 15);

    // Add dark tile layer
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }}).addTo(map);

    // Add NYC boundary layer
    const nycBoundaryStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.05,
      color: '#3b82f6',
      weight: 1,
      opacity: 0.3
    }};

    L.geoJSON(nycBoundary, {{
      style: nycBoundaryStyle
    }}).addTo(map);

    const blockDisplay = document.getElementById('block-display');

    // Color schemes for each step
    const stepColors = {{
      1: {{ fill: '#ff6347', border: '#ff6347' }},  // Tomato red
      2: {{ fill: '#32cd32', border: '#32cd32' }},  // Lime green
      3: {{ fill: '#1e90ff', border: '#1e90ff' }},  // Dodger blue
    }};

    // Purple color for completed items
    const completeColor = '#9333ea';  // Purple

    // Get style based on step and status
    function getStyle(step, status) {{
      const colors = stepColors[step];
      const isComplete = status === 'complete';
      return {{
        fillColor: isComplete ? completeColor : colors.fill,
        fillOpacity: isComplete ? 0.4 : 0.35,
        color: isComplete ? completeColor : colors.border,
        weight: 1.5,
        opacity: isComplete ? 0.7 : 0.7,
      }};
    }}

    function getHoverStyle(step, status) {{
      const colors = stepColors[step];
      const isComplete = status === 'complete';
      return {{
        fillColor: isComplete ? completeColor : colors.fill,
        fillOpacity: isComplete ? 0.6 : 0.6,
        color: '#fff',
        weight: 2,
        opacity: 1,
      }};
    }}

    // Add block polygons
    polygons.forEach(p => {{
      const style = getStyle(p.step, p.status);
      const hStyle = getHoverStyle(p.step, p.status);
      const polygon = L.polygon(p.corners, style).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(hStyle);
        const stepName = ['', '2x2 tile', '1x2/2x1 strip', '1x1 corner'][p.step];
        const statusEmoji = p.status === 'complete' ? '✅' : p.status === 'error' ? '❌' : '⏳';
        blockDisplay.innerHTML = `
          <div class="coords">
            <span class="label">pos:</span>
            <span>(${{p.x}}, ${{p.y}})</span>
          </div>
          <div class="block-info">
            Step ${{p.step}}: ${{stepName}} (${{p.width}}x${{p.height}})<br>
            Status: ${{p.status}} ${{statusEmoji}}
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(style);
        blockDisplay.innerHTML = '<span class="no-hover">Hover over a block</span>';
      }});
    }});

    // Add seed point marker
    const seedIcon = L.divIcon({{
      className: 'seed-marker',
      html: `<div style="
        width: 14px;
        height: 14px;
        background: #10b981;
        border: 3px solid #fff;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      "></div>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    }});

    L.marker([seedLat, seedLng], {{ icon: seedIcon }})
      .addTo(map)
      .bindPopup('Seed Point');

    // Fit map to polygon bounds
    if (polygons.length > 0) {{
      const allLatLngs = polygons.flatMap(p => p.corners);
      const bounds = L.latLngBounds(allLatLngs);
      map.fitBounds(bounds.pad(0.1));
    }}
  </script>
</body>
</html>
"""
  return html


def create_debug_map(
  layer_dir: Path,
  output_path: Path | None = None,
  bounds_path: Path | None = None,
) -> Path:
  """
  Create an interactive debug map showing the generation plan.

  Args:
      layer_dir: Path to the layer directory
      output_path: Optional output path (default: layer_dir/debug_map.html)
      bounds_path: Optional path to custom bounds GeoJSON file

  Returns:
      Path to the generated HTML file
  """
  print(f"\n{'=' * 60}")
  print("🗺️  Layer Plan Debug Map Generator")
  print(f"{'=' * 60}")
  print(f"   Layer dir: {layer_dir}")

  # Load layer config
  layer_config = load_layer_config(layer_dir)
  print(f"   Layer name: {layer_config.get('name', 'unnamed')}")

  # Load source generation config
  source_dir = Path(layer_config["generation_dir"])
  if not source_dir.is_absolute():
    # Try relative to current directory
    if not source_dir.exists():
      # Try relative to layer directory
      source_dir = layer_dir.parent.parent / layer_config["generation_dir"]

  source_db_path = source_dir / "quadrants.db"
  if not source_db_path.exists():
    raise FileNotFoundError(f"Source database not found: {source_db_path}")

  print(f"   Source: {source_dir}")

  source_conn = sqlite3.connect(source_db_path)
  try:
    source_config = get_generation_config(source_conn)
    print(f"   Source config: {source_config.get('name', 'unnamed')}")
  finally:
    source_conn.close()

  # Load generation plan
  plan_items = get_generation_plan(layer_dir)
  print(f"\n📊 Loaded {len(plan_items)} plan items")

  # Load boundary GeoJSON
  boundary_geojson = load_bounds(bounds_path)

  # Calculate all corners for bounds
  all_lats = []
  all_lngs = []
  for item in plan_items:
    corners = calculate_block_corners(
      source_config,
      item["top_left_x"],
      item["top_left_y"],
      item["width"],
      item["height"],
    )
    for lat, lng in corners:
      all_lats.append(lat)
      all_lngs.append(lng)

  # Calculate center
  seed_lat = source_config["seed"]["lat"]
  seed_lng = source_config["seed"]["lng"]

  if all_lats and all_lngs:
    center_lat = (min(all_lats) + max(all_lats)) / 2
    center_lng = (min(all_lngs) + max(all_lngs)) / 2
  else:
    center_lat = seed_lat
    center_lng = seed_lng

  # Generate HTML
  print("\n🎨 Generating interactive map...")
  html_content = generate_html(
    layer_config=layer_config,
    source_config=source_config,
    plan_items=plan_items,
    seed_lat=seed_lat,
    seed_lng=seed_lng,
    center_lat=center_lat,
    center_lng=center_lng,
    boundary_geojson=boundary_geojson,
  )

  # Set output path
  if output_path is None:
    output_path = layer_dir / "debug_map.html"

  # Save HTML file
  print(f"\n💾 Saving to {output_path}")
  output_path.write_text(html_content)

  print(f"\n✅ Debug map created: {output_path}")
  print(f"   Open in browser: file://{output_path.resolve()}")
  return output_path


def main():
  parser = argparse.ArgumentParser(
    description="Create an interactive debug map showing the layer generation plan.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "--layer-dir",
    type=Path,
    required=True,
    help="Path to the layer directory",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: layer_dir/debug_map.html)",
  )
  parser.add_argument(
    "--bounds",
    "-b",
    type=Path,
    help="Path to custom bounds GeoJSON file (default: NYC boundary)",
  )

  args = parser.parse_args()

  layer_dir = args.layer_dir.resolve()
  if not layer_dir.exists():
    print(f"❌ Error: Layer directory not found: {layer_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None
  bounds_path = args.bounds.resolve() if args.bounds else None

  try:
    create_debug_map(layer_dir, output_path, bounds_path)
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
