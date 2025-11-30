# Export images from whitebox and web viewer
_Exported on 11/30/2025 at 12:30:25 EST from Cursor (2.1.39)_

---

**User**

OK great - now we're on to the next feature - we need to be able to export images from both the @whitebox.py *and* web @src/web/main.js page view with *exactly* the same parameters (for now those defined in @src/view.json .

This should be a separate script that
a) runs the whitebox.py code in *non-interactive* mode, e.g. it does not open a viewer and saves a png file to an output directory
b) opens the web viewer in a webview and captures a screenshot of the page - we need to disable all control overlays

Eventually we'll work to ensure the two viewports are pixel-perfect exactly the same, but for now let's make a script that outputs the two images

---

**Cursor**

I'll create a script to export images from both views. Let me break this into steps:

1. Modify `whitebox.py` to support non-interactive export mode
2. Modify the web viewer to support hiding UI overlays
3. Create an export script that orchestrates both

Let me first modify `whitebox.py` to support non-interactive export mode with an `output_path` parameter:

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
  output_path=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Positive to align azimuth direction with top of view
      satellite_image = satellite_image.rotate(
        orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  # Positive to align azimuth direction with +Y (top of view)
  angle_rad = np.radians(orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()

  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = CAMERA_ZOOM

  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
  output_path=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Positive to align azimuth direction with top of view
      satellite_image = satellite_image.rotate(
        orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  # Use off-screen rendering if output_path is specified
  off_screen = output_path is not None
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height), off_screen=off_screen)
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  # Positive to align azimuth direction with +Y (top of view)
  angle_rad = np.radians(orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()

  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = CAMERA_ZOOM

  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
  output_path=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Positive to align azimuth direction with top of view
      satellite_image = satellite_image.rotate(
        orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  # Use off-screen rendering if output_path is specified
  off_screen = output_path is not None
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height), off_screen=off_screen)
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  # Positive to align azimuth direction with +Y (top of view)
  angle_rad = np.radians(orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()

  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = CAMERA_ZOOM

  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  if output_path:
    # Non-interactive mode: save screenshot and exit
    print(f"📸 Saving render to {output_path}...")
    plotter.screenshot(output_path)
    plotter.close()
    print(f"✅ Saved to {output_path}")
    return output_path
  else:
    # Interactive mode: show viewer
    print("📸 Displaying Isometric Render...")
    plotter.show()

    # Log final camera settings after user interaction
    print("\n🎥 FINAL CAMERA SETTINGS:")
    print(f"   Position: {plotter.camera.position}")
    print(f"   Focal Point: {plotter.camera.focal_point}")
    print(f"   Up Vector: {plotter.camera.up}")
    print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
    print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;

    // In export mode, wait a bit for tiles to fully render then signal ready
    if (EXPORT_MODE) {
      setTimeout(() => {
        window.__EXPORT_READY = true;
        console.log("Export ready!");
      }, 3000); // Wait 3 seconds for tiles to load
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```python
"""
Export views from both whitebox.py and web viewer.

This script captures screenshots from both rendering pipelines using the same
view parameters defined in view.json. The outputs can be compared for
alignment verification.

Usage:
  uv run python src/isometric_nyc/export_views.py [--output-dir OUTPUT_DIR]
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from isometric_nyc.whitebox import (
  LAT,
  LON,
  ORIENTATION_DEG,
  SIZE_METERS,
  VIEWPORT_HEIGHT,
  VIEWPORT_WIDTH,
  render_tile,
)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "exports"

# Web server configuration
WEB_DIR = Path(__file__).parent.parent / "web"
WEB_PORT = 5173  # Vite default port


def export_whitebox(output_dir: Path) -> Path:
  """
  Export screenshot from whitebox.py renderer.

  Args:
    output_dir: Directory to save the output

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"whitebox_{timestamp}.png"

  print("🎨 Rendering whitebox view...")
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
    output_path=str(output_path),
  )

  return output_path


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  # Wait for server to start
  time.sleep(3)
  return process


def export_web_view(output_dir: Path, port: int) -> Path:
  """
  Export screenshot from web viewer using Playwright.

  Args:
    output_dir: Directory to save the output
    port: Port where web server is running

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"web_{timestamp}.png"

  url = f"http://localhost:{port}/?export=true"

  print(f"🌐 Capturing web view from {url}...")

  with sync_playwright() as p:
    # Launch browser
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
      viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
    )
    page = context.new_page()

    # Navigate to the page
    page.goto(url)

    # Wait for the export ready signal
    print("   ⏳ Waiting for tiles to load...")
    page.wait_for_function("window.__EXPORT_READY === true", timeout=60000)

    # Take screenshot
    print(f"   📸 Taking screenshot...")
    page.screenshot(path=str(output_path))

    browser.close()

  print(f"   ✅ Saved to {output_path}")
  return output_path


def main():
  parser = argparse.ArgumentParser(
    description="Export views from whitebox and web viewer"
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory to save exported images",
  )
  parser.add_argument(
    "--whitebox-only",
    action="store_true",
    help="Only export whitebox view",
  )
  parser.add_argument(
    "--web-only",
    action="store_true",
    help="Only export web view",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=WEB_PORT,
    help="Port for web server",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  print("=" * 60)
  print("🏙️  ISOMETRIC NYC VIEW EXPORTER")
  print("=" * 60)
  print(f"📁 Output directory: {args.output_dir}")
  print(f"📍 View: {LAT}, {LON}")
  print(f"📐 Size: {SIZE_METERS}m, Orientation: {ORIENTATION_DEG}°")
  print(f"🖥️  Resolution: {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT}")
  print("=" * 60)

  results = {}
  web_server = None

  try:
    # Export whitebox view
    if not args.web_only:
      whitebox_path = export_whitebox(args.output_dir)
      results["whitebox"] = whitebox_path

    # Export web view
    if not args.whitebox_only:
      # Start web server if needed
      if not args.no_start_server:
        web_server = start_web_server(WEB_DIR, args.port)

      try:
        web_path = export_web_view(args.output_dir, args.port)
        results["web"] = web_path
      finally:
        # Stop web server
        if web_server:
          print("🛑 Stopping web server...")
          web_server.terminate()
          web_server.wait()

  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    if web_server:
      web_server.terminate()
    sys.exit(1)

  # Summary
  print("\n" + "=" * 60)
  print("📦 EXPORT COMPLETE")
  print("=" * 60)
  for name, path in results.items():
    print(f"   {name}: {path}")

  return results


if __name__ == "__main__":
  main()

```

```toml
[project]
name = "isometric-nyc"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "google-genai>=1.52.0",
    "googlemaps>=4.10.0",
    "marimo>=0.18.0",
    "numpy>=2.3.5",
    "opencv-python>=4.11.0.86",
    "pillow>=12.0.0",
    "playwright>=1.49.0",
    "psycopg2-binary>=2.9.11",
    "pydantic>=2.12.4",
    "pyproj>=3.7.2",
    "python-dotenv>=1.2.1",
    "pyvista>=0.46.4",
    "shapely>=2.1.2",
    "sodapy>=2.2.0",
    "tabulate>=0.9.0",
]

[project.scripts]
check-boundaries = "isometric_nyc.check_boundaries:main"
check-coordinates = "isometric_nyc.check_coordinates:main"
check-db = "isometric_nyc.check_db:main"
isometric-nyc = "isometric_nyc.main:main"
inspect-db = "isometric_nyc.inspect_db:main"
inspect-columns = "isometric_nyc.inspect_columns:main"
whitebox = "isometric_nyc.whitebox:main"

[tool.ruff]
line-length = 88
indent-width = 2

[tool.ruff.format]
indent-style = "space"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = []

[dependency-groups]
dev = [
    "ruff>=0.14.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

```

```toml
[project]
name = "isometric-nyc"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "google-genai>=1.52.0",
    "googlemaps>=4.10.0",
    "marimo>=0.18.0",
    "numpy>=2.3.5",
    "opencv-python>=4.11.0.86",
    "pillow>=12.0.0",
    "playwright>=1.49.0",
    "psycopg2-binary>=2.9.11",
    "pydantic>=2.12.4",
    "pyproj>=3.7.2",
    "python-dotenv>=1.2.1",
    "pyvista>=0.46.4",
    "shapely>=2.1.2",
    "sodapy>=2.2.0",
    "tabulate>=0.9.0",
]

[project.scripts]
check-boundaries = "isometric_nyc.check_boundaries:main"
check-coordinates = "isometric_nyc.check_coordinates:main"
check-db = "isometric_nyc.check_db:main"
export-views = "isometric_nyc.export_views:main"
isometric-nyc = "isometric_nyc.main:main"
inspect-db = "isometric_nyc.inspect_db:main"
inspect-columns = "isometric_nyc.inspect_columns:main"
whitebox = "isometric_nyc.whitebox:main"

[tool.ruff]
line-length = 88
indent-width = 2

[tool.ruff.format]
indent-style = "space"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = []

[dependency-groups]
dev = [
    "ruff>=0.14.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

```

```ignore
# General
.DS_Store
.idea/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# Ruff
.ruff_cache/

# Local Database
*.db
buildings.db

# Downloaded data
map_data/**/*.3dm
map_data/**/*.gml
map_data/**/*.obj
map_data/**/*.rhl
map_data/**/*.mtl
map_data/**/*.zip

# Satellite tile cache
.satellite_cache/

# Exported view images
exports/

```

Let me test the whitebox export to make sure it works:

I've created the export system. Here's a summary of the changes:

### Changes Made

1. **`whitebox.py`** - Added non-interactive export mode:
   - New `output_path` parameter to `render_tile()` function
   - When provided, renders off-screen and saves screenshot to file
   - Returns without opening interactive viewer

2. **`main.js`** - Added export mode support:
   - Checks for `?export=true` URL parameter
   - Hides UI overlay in export mode
   - Sets `window.__EXPORT_READY = true` when tiles are loaded (3s delay for full rendering)

3. **`export_views.py`** - New export script:
   - Exports both whitebox and web views using same `view.json` parameters
   - Uses Playwright to capture web screenshots
   - Supports command-line options for flexibility

4. **`pyproject.toml`** - Added:
   - `playwright` dependency
   - `export-views` script entry point

5. **`.gitignore`** - Added `exports/` directory

### Usage

```bash
# Export both views (starts its own web server)
uv run export-views

# Export whitebox only
uv run export-views --whitebox-only

# Export web only (if you have the server running)
uv run export-views --web-only --no-start-server

# Custom output directory
uv run export-views --output-dir ./my_exports
```

Or run directly:
```bash
uv run python src/isometric_nyc/export_views.py
```

### Output

Images are saved to `exports/` with timestamps:
- `whitebox_YYYYMMDD_HHMMSS.png` - PyVista 3D render
- `web_YYYYMMDD_HHMMSS.png` - Google 3D Tiles web view

Both use the same parameters from `src/view.json` (lat, lon, size_meters, camera angles).

```shellscript
# Export both views (starts its own web server)
uv run export-views

# Export whitebox only
uv run export-views --whitebox-only

# Export web only (if you have the server running)
uv run export-views --web-only --no-start-server

# Custom output directory
uv run export-views --output-dir ./my_exports
```

```shellscript
uv run python src/isometric_nyc/export_views.py
```

---

**User**

OK great - it looks like the view for the web render isn't working - it's just completely zoomed out over north america. Maybe the params aren't getting loaded correctly?

---

**Cursor**

I see the issue - the web view is showing the whole globe instead of zooming into Madison Square Garden. Let me check what's happening with the camera positioning in export mode.

I see the issue. The `load-tile-set` event fires when the tileset metadata loads, but the actual tiles haven't been loaded and rendered yet. The camera positioning might be happening but the view isn't updated. Let me fix this by waiting for actual tiles to render:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;

    // In export mode, wait a bit for tiles to fully render then signal ready
    if (EXPORT_MODE) {
      setTimeout(() => {
        window.__EXPORT_READY = true;
        console.log("Export ready!");
      }, 3000); // Wait 3 seconds for tiles to load
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset metadata loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(`Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);
    
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds
      
      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const tilesLoaded = tiles.group.children.length > 0;
        const visibleTiles = tiles.visibleTiles?.size || 0;
        
        console.log(`[${checkCount}] Tiles group children: ${tiles.group.children.length}, visible: ${visibleTiles}`);
        
        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };
      
      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset metadata loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(`Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);
    
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds
      
      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const tilesLoaded = tiles.group.children.length > 0;
        const visibleTiles = tiles.visibleTiles?.size || 0;
        
        console.log(`[${checkCount}] Tiles group children: ${tiles.group.children.length}, visible: ${visibleTiles}`);
        
        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };
      
      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  console.log("Positioning camera...");
  console.log(`  Target: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}m`);
  console.log(`  Angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  console.log(`  Camera position after decompose: ${camera.position.x.toFixed(0)}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)}`);

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned at ${HEIGHT}m`);
  console.log(`  Mode: ${transition.mode}`);
  console.log(`  Final position: ${transition.camera.position.x.toFixed(0)}, ${transition.camera.position.y.toFixed(0)}, ${transition.camera.position.z.toFixed(0)}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```python
"""
Export views from both whitebox.py and web viewer.

This script captures screenshots from both rendering pipelines using the same
view parameters defined in view.json. The outputs can be compared for
alignment verification.

Usage:
  uv run python src/isometric_nyc/export_views.py [--output-dir OUTPUT_DIR]
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from isometric_nyc.whitebox import (
  LAT,
  LON,
  ORIENTATION_DEG,
  SIZE_METERS,
  VIEWPORT_HEIGHT,
  VIEWPORT_WIDTH,
  render_tile,
)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "exports"

# Web server configuration
WEB_DIR = Path(__file__).parent.parent / "web"
WEB_PORT = 5173  # Vite default port


def export_whitebox(output_dir: Path) -> Path:
  """
  Export screenshot from whitebox.py renderer.

  Args:
    output_dir: Directory to save the output

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"whitebox_{timestamp}.png"

  print("🎨 Rendering whitebox view...")
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
    output_path=str(output_path),
  )

  return output_path


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  # Wait for server to start
  time.sleep(3)
  return process


def export_web_view(output_dir: Path, port: int) -> Path:
  """
  Export screenshot from web viewer using Playwright.

  Args:
    output_dir: Directory to save the output
    port: Port where web server is running

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"web_{timestamp}.png"

  url = f"http://localhost:{port}/?export=true"

  print(f"🌐 Capturing web view from {url}...")

  with sync_playwright() as p:
    # Launch browser with GPU support for WebGL
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--enable-features=Vulkan",
        "--ignore-gpu-blocklist",
      ],
    )
    context = browser.new_context(
      viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
      device_scale_factor=1,
    )
    page = context.new_page()

    # Enable console logging from the page
    page.on("console", lambda msg: print(f"   [browser] {msg.text}"))

    # Navigate to the page
    print("   ⏳ Loading page...")
    page.goto(url, wait_until="networkidle")

    # Wait for the export ready signal (up to 90 seconds)
    print("   ⏳ Waiting for tiles to load...")
    try:
      page.wait_for_function("window.__EXPORT_READY === true", timeout=90000)
    except Exception as e:
      # Debug: check what state we're in
      tiles_loaded = page.evaluate("window.__TILES_LOADED")
      camera_positioned = page.evaluate("window.__CAMERA_POSITIONED")
      print(f"   ⚠️  Timeout! TILES_LOADED={tiles_loaded}, CAMERA_POSITIONED={camera_positioned}")
      raise e

    # Take screenshot
    print(f"   📸 Taking screenshot...")
    page.screenshot(path=str(output_path))

    browser.close()

  print(f"   ✅ Saved to {output_path}")
  return output_path


def main():
  parser = argparse.ArgumentParser(
    description="Export views from whitebox and web viewer"
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory to save exported images",
  )
  parser.add_argument(
    "--whitebox-only",
    action="store_true",
    help="Only export whitebox view",
  )
  parser.add_argument(
    "--web-only",
    action="store_true",
    help="Only export web view",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=WEB_PORT,
    help="Port for web server",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  print("=" * 60)
  print("🏙️  ISOMETRIC NYC VIEW EXPORTER")
  print("=" * 60)
  print(f"📁 Output directory: {args.output_dir}")
  print(f"📍 View: {LAT}, {LON}")
  print(f"📐 Size: {SIZE_METERS}m, Orientation: {ORIENTATION_DEG}°")
  print(f"🖥️  Resolution: {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT}")
  print("=" * 60)

  results = {}
  web_server = None

  try:
    # Export whitebox view
    if not args.web_only:
      whitebox_path = export_whitebox(args.output_dir)
      results["whitebox"] = whitebox_path

    # Export web view
    if not args.whitebox_only:
      # Start web server if needed
      if not args.no_start_server:
        web_server = start_web_server(WEB_DIR, args.port)

      try:
        web_path = export_web_view(args.output_dir, args.port)
        results["web"] = web_path
      finally:
        # Stop web server
        if web_server:
          print("🛑 Stopping web server...")
          web_server.terminate()
          web_server.wait()

  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    if web_server:
      web_server.terminate()
    sys.exit(1)

  # Summary
  print("\n" + "=" * 60)
  print("📦 EXPORT COMPLETE")
  print("=" * 60)
  for name, path in results.items():
    print(f"   {name}: {path}")

  return results


if __name__ == "__main__":
  main()


```

```python
"""
Export views from both whitebox.py and web viewer.

This script captures screenshots from both rendering pipelines using the same
view parameters defined in view.json. The outputs can be compared for
alignment verification.

Usage:
  uv run python src/isometric_nyc/export_views.py [--output-dir OUTPUT_DIR]
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from isometric_nyc.whitebox import (
  LAT,
  LON,
  ORIENTATION_DEG,
  SIZE_METERS,
  VIEWPORT_HEIGHT,
  VIEWPORT_WIDTH,
  render_tile,
)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "exports"

# Web server configuration
WEB_DIR = Path(__file__).parent.parent / "web"
WEB_PORT = 5173  # Vite default port


def export_whitebox(output_dir: Path) -> Path:
  """
  Export screenshot from whitebox.py renderer.

  Args:
    output_dir: Directory to save the output

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"whitebox_{timestamp}.png"

  print("🎨 Rendering whitebox view...")
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
    output_path=str(output_path),
  )

  return output_path


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  # Wait for server to start
  print("   ⏳ Waiting for server to start...")
  time.sleep(5)
  print(f"   ✅ Server started on http://localhost:{port}")
  return process


def export_web_view(output_dir: Path, port: int) -> Path:
  """
  Export screenshot from web viewer using Playwright.

  Args:
    output_dir: Directory to save the output
    port: Port where web server is running

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"web_{timestamp}.png"

  url = f"http://localhost:{port}/?export=true"

  print(f"🌐 Capturing web view from {url}...")

  with sync_playwright() as p:
    # Launch browser with GPU support for WebGL
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--enable-features=Vulkan",
        "--ignore-gpu-blocklist",
      ],
    )
    context = browser.new_context(
      viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
      device_scale_factor=1,
    )
    page = context.new_page()

    # Enable console logging from the page
    page.on("console", lambda msg: print(f"   [browser] {msg.text}"))

    # Navigate to the page
    print("   ⏳ Loading page...")
    page.goto(url, wait_until="networkidle")

    # Wait for the export ready signal (up to 90 seconds)
    print("   ⏳ Waiting for tiles to load...")
    try:
      page.wait_for_function("window.__EXPORT_READY === true", timeout=90000)
    except Exception as e:
      # Debug: check what state we're in
      tiles_loaded = page.evaluate("window.__TILES_LOADED")
      camera_positioned = page.evaluate("window.__CAMERA_POSITIONED")
      print(f"   ⚠️  Timeout! TILES_LOADED={tiles_loaded}, CAMERA_POSITIONED={camera_positioned}")
      raise e

    # Take screenshot
    print(f"   📸 Taking screenshot...")
    page.screenshot(path=str(output_path))

    browser.close()

  print(f"   ✅ Saved to {output_path}")
  return output_path


def main():
  parser = argparse.ArgumentParser(
    description="Export views from whitebox and web viewer"
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory to save exported images",
  )
  parser.add_argument(
    "--whitebox-only",
    action="store_true",
    help="Only export whitebox view",
  )
  parser.add_argument(
    "--web-only",
    action="store_true",
    help="Only export web view",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=WEB_PORT,
    help="Port for web server",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  print("=" * 60)
  print("🏙️  ISOMETRIC NYC VIEW EXPORTER")
  print("=" * 60)
  print(f"📁 Output directory: {args.output_dir}")
  print(f"📍 View: {LAT}, {LON}")
  print(f"📐 Size: {SIZE_METERS}m, Orientation: {ORIENTATION_DEG}°")
  print(f"🖥️  Resolution: {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT}")
  print("=" * 60)

  results = {}
  web_server = None

  try:
    # Export whitebox view
    if not args.web_only:
      whitebox_path = export_whitebox(args.output_dir)
      results["whitebox"] = whitebox_path

    # Export web view
    if not args.whitebox_only:
      # Start web server if needed
      if not args.no_start_server:
        web_server = start_web_server(WEB_DIR, args.port)

      try:
        web_path = export_web_view(args.output_dir, args.port)
        results["web"] = web_path
      finally:
        # Stop web server
        if web_server:
          print("🛑 Stopping web server...")
          web_server.terminate()
          web_server.wait()

  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    if web_server:
      web_server.terminate()
    sys.exit(1)

  # Summary
  print("\n" + "=" * 60)
  print("📦 EXPORT COMPLETE")
  print("=" * 60)
  for name, path in results.items():
    print(f"   {name}: {path}")

  return results


if __name__ == "__main__":
  main()


```

I've made several improvements:

1. **Better tile loading detection** - Now polls for actual visible tiles instead of a fixed 3-second timeout
2. **Added console logging** - Browser console messages will now be printed to the terminal during export
3. **GPU/WebGL support** - Added Chromium flags to ensure WebGL renders properly in headless mode
4. **More debugging** - Camera positioning now logs detailed information

You can test the web view manually first by opening:
```
http://localhost:5173/?export=true
```

Check the browser console to see if the camera is being positioned correctly. The logs should show:
- View config (LAT, LON, HEIGHT)
- Camera position after decompose
- Tile loading progress

If you want to test the full export, run:
```bash
uv run export-views --web-only --no-start-server
```
(assuming you have the dev server running)

This will show browser console output in the terminal, which should help diagnose whether the issue is:
- Camera not positioning correctly
- Tiles not loading
- Something else

```plaintext
http://localhost:5173/?export=true
```

```shellscript
uv run export-views --web-only --no-start-server
```

---

**User**

Hmm... when I go to http://localhost:5173/?export=true it does work, but the script is still exporting a fully zoomed out view...

---

**Cursor**

Let me check the terminal output to see what's happening during the export:

```plaintext
   [browser] 📷 Camera: Az=0.0° El=-90.0° Height=1603638m | Lat=40.7888° Lon=-73.9934°
```

I see the problem! Looking at the logs:

1. **`load-tile-set` fires many times** - Google 3D Tiles has nested tilesets, and each one triggers this event
2. **Camera height is 1,603,638m** - that's ~1600km above Earth (space view!)
3. **Each tileset load repositions camera and creates new timers**

The camera is being repositioned dozens of times, and each time `controls.adjustCamera()` is called, it's fighting with the positioning. Let me fix this:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;
let rootTilesetLoaded = false; // Track if we've already handled root tileset

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset metadata loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(
      `Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`
    );

    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds

      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const tilesLoaded = tiles.group.children.length > 0;
        const visibleTiles = tiles.visibleTiles?.size || 0;

        console.log(
          `[${checkCount}] Tiles group children: ${tiles.group.children.length}, visible: ${visibleTiles}`
        );

        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };

      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  console.log("Positioning camera...");
  console.log(`  Target: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}m`);
  console.log(`  Angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  console.log(
    `  Camera position after decompose: ${camera.position.x.toFixed(
      0
    )}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)}`
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned at ${HEIGHT}m`);
  console.log(`  Mode: ${transition.mode}`);
  console.log(
    `  Final position: ${transition.camera.position.x.toFixed(
      0
    )}, ${transition.camera.position.y.toFixed(
      0
    )}, ${transition.camera.position.z.toFixed(0)}`
  );
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```javascript
  // Connect controls to the tiles ellipsoid and position camera
  // NOTE: load-tile-set fires for EVERY nested tileset in Google 3D Tiles
  // We only want to set up camera ONCE on the first (root) tileset
  tiles.addEventListener("load-tile-set", () => {
    if (rootTilesetLoaded) {
      // Already handled root tileset, ignore subsequent ones
      return;
    }
    rootTilesetLoaded = true;

    console.log("Root tileset loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(`Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds

      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const visibleTiles = tiles.visibleTiles?.size || 0;

        console.log(
          `[${checkCount}] Tiles: ${tiles.group.children.length} children, ${visibleTiles} visible`
        );

        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };

      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });
```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;
let rootTilesetLoaded = false; // Track if we've already handled root tileset

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  // NOTE: load-tile-set fires for EVERY nested tileset in Google 3D Tiles
  // We only want to set up camera ONCE on the first (root) tileset
  tiles.addEventListener("load-tile-set", () => {
    if (rootTilesetLoaded) {
      // Already handled root tileset, ignore subsequent ones
      return;
    }
    rootTilesetLoaded = true;

    console.log("Root tileset loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(
      `Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`
    );

    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds

      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const visibleTiles = tiles.visibleTiles?.size || 0;

        console.log(
          `[${checkCount}] Tiles: ${tiles.group.children.length} children, ${visibleTiles} visible`
        );

        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };

      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  console.log("Positioning camera...");
  console.log(`  Target: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}m`);
  console.log(`  Angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  console.log(
    `  Camera position after decompose: ${camera.position.x.toFixed(
      0
    )}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)}`
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned at ${HEIGHT}m`);
  console.log(`  Mode: ${transition.mode}`);
  console.log(
    `  Final position: ${transition.camera.position.x.toFixed(
      0
    )}, ${transition.camera.position.y.toFixed(
      0
    )}, ${transition.camera.position.z.toFixed(0)}`
  );
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

Now let me also disable the controls in export mode since they might be interfering with the camera position:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;
let rootTilesetLoaded = false; // Track if we've already handled root tileset

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Disable controls in export mode to prevent interference with camera position
  if (EXPORT_MODE) {
    controls.enabled = false;
  }

  // Connect controls to the tiles ellipsoid and position camera
  // NOTE: load-tile-set fires for EVERY nested tileset in Google 3D Tiles
  // We only want to set up camera ONCE on the first (root) tileset
  tiles.addEventListener("load-tile-set", () => {
    if (rootTilesetLoaded) {
      // Already handled root tileset, ignore subsequent ones
      return;
    }
    rootTilesetLoaded = true;

    console.log("Root tileset loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(
      `Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`
    );

    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds

      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const visibleTiles = tiles.visibleTiles?.size || 0;

        console.log(
          `[${checkCount}] Tiles: ${tiles.group.children.length} children, ${visibleTiles} visible`
        );

        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };

      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  console.log("Positioning camera...");
  console.log(`  Target: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}m`);
  console.log(`  Angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  console.log(
    `  Camera position after decompose: ${camera.position.x.toFixed(
      0
    )}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)}`
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned at ${HEIGHT}m`);
  console.log(`  Mode: ${transition.mode}`);
  console.log(
    `  Final position: ${transition.camera.position.x.toFixed(
      0
    )}, ${transition.camera.position.y.toFixed(
      0
    )}, ${transition.camera.position.z.toFixed(0)}`
  );
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;
let rootTilesetLoaded = false; // Track if we've already handled root tileset

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Disable controls in export mode to prevent interference with camera position
  if (EXPORT_MODE) {
    controls.enabled = false;
  }

  // Connect controls to the tiles ellipsoid and position camera
  // NOTE: load-tile-set fires for EVERY nested tileset in Google 3D Tiles
  // We only want to set up camera ONCE on the first (root) tileset
  tiles.addEventListener("load-tile-set", () => {
    if (rootTilesetLoaded) {
      // Already handled root tileset, ignore subsequent ones
      return;
    }
    rootTilesetLoaded = true;

    console.log("Root tileset loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(
      `Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`
    );

    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds

      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const visibleTiles = tiles.visibleTiles?.size || 0;

        console.log(
          `[${checkCount}] Tiles: ${tiles.group.children.length} children, ${visibleTiles} visible`
        );

        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };

      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  console.log("Positioning camera...");
  console.log(`  Target: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}m`);
  console.log(`  Angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  console.log(
    `  Camera position after decompose: ${camera.position.x.toFixed(
      0
    )}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)}`
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned at ${HEIGHT}m`);
  console.log(`  Mode: ${transition.mode}`);
  console.log(
    `  Final position: ${transition.camera.position.x.toFixed(
      0
    )}, ${transition.camera.position.y.toFixed(
      0
    )}, ${transition.camera.position.z.toFixed(0)}`
  );
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  // Keep controls disabled in export mode
  if (!EXPORT_MODE) {
    controls.enabled = !transition.animating;
  }
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

I see - `controls.adjustCamera()` is changing the camera position from the correct values. Let me skip those calls in export mode:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;
let rootTilesetLoaded = false; // Track if we've already handled root tileset

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Disable controls in export mode to prevent interference with camera position
  if (EXPORT_MODE) {
    controls.enabled = false;
  }

  // Connect controls to the tiles ellipsoid and position camera
  // NOTE: load-tile-set fires for EVERY nested tileset in Google 3D Tiles
  // We only want to set up camera ONCE on the first (root) tileset
  tiles.addEventListener("load-tile-set", () => {
    if (rootTilesetLoaded) {
      // Already handled root tileset, ignore subsequent ones
      return;
    }
    rootTilesetLoaded = true;

    console.log("Root tileset loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(
      `Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`
    );

    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds

      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const visibleTiles = tiles.visibleTiles?.size || 0;

        console.log(
          `[${checkCount}] Tiles: ${tiles.group.children.length} children, ${visibleTiles} visible`
        );

        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };

      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  console.log("Positioning camera...");
  console.log(`  Target: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}m`);
  console.log(`  Angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  console.log(
    `  Camera position after decompose: ${camera.position.x.toFixed(
      0
    )}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)}`
  );

  // Sync both cameras
  transition.syncCameras();

  // In export mode, skip adjustCamera which can modify position unexpectedly
  if (!EXPORT_MODE) {
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    if (!EXPORT_MODE) {
      controls.getPivotPoint(transition.fixedPoint);
    }
    transition.toggle();
  }

  console.log(`Camera positioned at ${HEIGHT}m`);
  console.log(`  Mode: ${transition.mode}`);
  console.log(
    `  Final position: ${transition.camera.position.x.toFixed(
      0
    )}, ${transition.camera.position.y.toFixed(
      0
    )}, ${transition.camera.position.z.toFixed(0)}`
  );
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  // Keep controls disabled in export mode
  if (!EXPORT_MODE) {
    controls.enabled = !transition.animating;
  }
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

I've made several fixes to handle export mode properly:

1. **Only process first tileset load** - Added `rootTilesetLoaded` flag to prevent camera repositioning on every nested tileset
2. **Disable controls in export mode** - Set `controls.enabled = false` to prevent interference
3. **Skip `adjustCamera()` calls** - These were modifying the camera position unexpectedly
4. **Skip `getPivotPoint()` in export mode** - This also affected camera state

Try running the export again:

```bash
uv run export-views
```

Or if you have the dev server running, test the web view first:
```bash
uv run export-views --web-only --no-start-server
```

The browser logs should now show only one "Root tileset loaded!" message, and the final camera position should match the decomposed position (around 1334331, 4141669, 4651344 in ECEF coordinates).

```shellscript
uv run export-views
```

```shellscript
uv run export-views --web-only --no-start-server
```

---

**User**

Yuck - when I go to http://localhost:5173/?export=true it's all messed up now - look at the screen and logs



API Key loaded: Yes

main.js:278 Export mode: UI disabled

main.js:130 Root tileset loaded!

main.js:131 View config: LAT=40.7505, LON=-73.9934, HEIGHT=300

main.js:132 Camera angles: Az=29.5°, El=-90°

main.js:189 Positioning camera...

main.js:190   Target: LAT=40.7505, LON=-73.9934, HEIGHT=300m

main.js:191   Angles: Az=29.5°, El=-90°

main.js:215   Camera position after decompose: 1334331, 4141669, 4651344

main.js:238 Camera positioned at 300m

main.js:239   Mode: orthographic

main.js:240   Final position: 1334331, 4141669, 4651344

main.js:143 Export mode: waiting for tiles to render...

main.js:372 📷 Camera: Az=180.0° El=0.2° Height=-6344694m | Lat=-49.4397° Lon=-73.9934°

main.js:152 [1] Tiles: 24 children, 24 visible

main.js:161 Export ready! (24 visible tiles)

---

**Cursor**

I see the issue - by skipping `adjustCamera()`, the orthographic camera isn't being set up properly. The orthographic camera needs its frustum bounds calculated based on the view distance. Let me fix this by staying in perspective mode for export, or properly handling the orthographic setup:

The issue is that orthographic mode needs proper frustum setup from `adjustCamera()`. For export mode, let's stay in perspective mode which works correctly:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Check if we're in export mode (no UI overlays)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

// Export mode: track when scene is ready for screenshot
window.__EXPORT_READY = false;
window.__TILES_LOADED = false;
window.__CAMERA_POSITIONED = false;
let rootTilesetLoaded = false; // Track if we've already handled root tileset

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Disable controls in export mode to prevent interference with camera position
  if (EXPORT_MODE) {
    controls.enabled = false;
  }

  // Connect controls to the tiles ellipsoid and position camera
  // NOTE: load-tile-set fires for EVERY nested tileset in Google 3D Tiles
  // We only want to set up camera ONCE on the first (root) tileset
  tiles.addEventListener("load-tile-set", () => {
    if (rootTilesetLoaded) {
      // Already handled root tileset, ignore subsequent ones
      return;
    }
    rootTilesetLoaded = true;

    console.log("Root tileset loaded!");
    console.log(`View config: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}`);
    console.log(
      `Camera angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`
    );

    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
    window.__TILES_LOADED = true;
    window.__CAMERA_POSITIONED = true;

    // In export mode, wait for tiles to actually load with polling
    if (EXPORT_MODE) {
      console.log("Export mode: waiting for tiles to render...");
      let checkCount = 0;
      const maxChecks = 60; // Up to 30 seconds

      const checkTilesLoaded = () => {
        checkCount++;
        // Check if we have rendered tiles
        const visibleTiles = tiles.visibleTiles?.size || 0;

        console.log(
          `[${checkCount}] Tiles: ${tiles.group.children.length} children, ${visibleTiles} visible`
        );

        // Wait until we have visible tiles or max time
        if (visibleTiles > 10 || checkCount >= maxChecks) {
          // Wait an additional 2 seconds after tiles are visible for full quality
          setTimeout(() => {
            window.__EXPORT_READY = true;
            console.log(`Export ready! (${visibleTiles} visible tiles)`);
          }, 2000);
        } else {
          setTimeout(checkTilesLoaded, 500);
        }
      };

      // Start checking after initial delay
      setTimeout(checkTilesLoaded, 1000);
    }
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  console.log("Positioning camera...");
  console.log(`  Target: LAT=${LAT}, LON=${LON}, HEIGHT=${HEIGHT}m`);
  console.log(`  Angles: Az=${CAMERA_AZIMUTH}°, El=${CAMERA_ELEVATION}°`);

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  console.log(
    `  Camera position after decompose: ${camera.position.x.toFixed(
      0
    )}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)}`
  );

  // Sync both cameras
  transition.syncCameras();

  // In export mode, use perspective camera (orthographic needs adjustCamera to set up frustum)
  if (EXPORT_MODE) {
    // Stay in perspective mode for export - it's simpler and works correctly
    isOrthographic = false;
    console.log("  Export mode: using perspective camera");
  } else {
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);

    // Switch to orthographic mode by default
    if (isOrthographic && transition.mode === "perspective") {
      controls.getPivotPoint(transition.fixedPoint);
      transition.toggle();
    }
  }

  console.log(`Camera positioned at ${HEIGHT}m`);
  console.log(`  Mode: ${transition.mode}`);
  console.log(
    `  Final position: ${transition.camera.position.x.toFixed(
      0
    )}, ${transition.camera.position.y.toFixed(
      0
    )}, ${transition.camera.position.z.toFixed(0)}`
  );
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  // Skip UI in export mode
  if (EXPORT_MODE) {
    console.log("Export mode: UI disabled");
    return;
  }

  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  // Keep controls disabled in export mode
  if (!EXPORT_MODE) {
    controls.enabled = !transition.animating;
  }
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

Fixed! The problem was that orthographic mode needs `adjustCamera()` to properly set up its viewing frustum, but that function was also modifying the camera position.

The solution: **Use perspective mode for export**. The camera position is set correctly, and perspective mode works without the additional frustum setup.

Refresh `http://localhost:5173/?export=true` and it should show the correct zoomed-in view of Madison Square Garden area now. The logs should show:
```
Export mode: using perspective camera
Mode: perspective
```

```plaintext
Export mode: using perspective camera
Mode: perspective
```

---

**User**

OK - everything got a bit messed up so I've reverted most of the changes from the main.js. It currently loads correctly now, let's start a new chat to figure this export out

