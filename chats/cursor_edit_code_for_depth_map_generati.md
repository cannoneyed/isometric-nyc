# Edit code for depth map generation
_Exported on 12/2/2025 at 17:21:36 EST from Cursor (2.1.39)_

---

**User**

OK next experiment - I want you to edit the code in @src/isometric_nyc/whitebox.py to generate a depth map instead of a whitebox

---

**Cursor**

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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
  
  if render_style == "depth":
      plotter.set_background("black")
      use_satellite = False  # Disable satellite texture in depth mode
  else:
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
  
  if render_style == "depth":
      plotter.set_background("black")
      use_satellite = False  # Disable satellite texture in depth mode
  else:
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

  # Helper for depth calculation
  def get_depth_scalar(points, el_deg):
      rad = np.radians(el_deg)
      c = np.cos(rad)
      s = np.sin(rad)
      # Depth along view vector (assuming Azimuth aligned to Y)
      # Vector from cam to target is (0, cos, sin)
      return points[:, 1] * c + points[:, 2] * s

  if render_style == "depth":
      # Calculate global min/max depth for consistent normalization
      all_depths = []
      
      # Collect all active meshes
      active_meshes = []
      if batches.get(710): active_meshes.append(batches[710])
      if batches.get("other"): 
          batches["other"].translate((0, 0, 0.1), inplace=True)
          active_meshes.append(batches["other"])
      if batches.get(709): 
          batches[709].translate((0, 0, 0.2), inplace=True)
          active_meshes.append(batches[709])
      if batches.get(712): 
          batches[712].translate((0, 0, 0.3), inplace=True)
          active_meshes.append(batches[712])
          
      # Also add base plane for depth continuity
      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 1.0
      base_plane = pv.Plane(center=(0, 0, ground_z), i_size=ground_size, j_size=ground_size, direction=(0,0,1))
      # PyVista Plane creates a mesh. We need to make sure it covers the area.
      # Note: pv.Plane defaults to centered at 'center' with normal 'direction'.
      # i_size is X, j_size is Y.
      # But we used custom vertices before. Let's recreate that simple mesh to be sure.
      base_points = np.array([
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z]
      ])
      base_faces = [4, 0, 1, 2, 3]
      base_mesh = pv.PolyData(base_points, base_faces)
      active_meshes.insert(0, base_mesh)

      # Calculate bounds
      for mesh in active_meshes:
          d = get_depth_scalar(mesh.points, camera_elevation_deg)
          all_depths.append(d)
          # Store for later use to avoid recomputing
          mesh.point_data["depth"] = d
      
      if all_depths:
          full_concat = np.concatenate(all_depths)
          min_d, max_d = np.min(full_concat), np.max(full_concat)
          print(f"   📏 Depth Range: {min_d:.2f} to {max_d:.2f}")
          
          for mesh in active_meshes:
              plotter.add_mesh(
                  mesh, 
                  scalars="depth", 
                  cmap="gray_r",  # Close (Min) = White, Far (Max) = Black
                  clim=[min_d, max_d],
                  show_scalar_bar=False,
                  show_edges=False
              )
      
  # 6.1. Add satellite image as ground plane texture (Whitebox Mode)
  elif use_satellite and satellite_texture is not None:
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
  
  if render_style == "depth":
      plotter.set_background("black")
      use_satellite = False  # Disable satellite texture in depth mode
  else:
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

  # Helper for depth calculation
  def get_depth_scalar(points, el_deg):
      rad = np.radians(el_deg)
      c = np.cos(rad)
      s = np.sin(rad)
      # Depth along view vector (assuming Azimuth aligned to Y)
      # Vector from cam to target is (0, cos, sin)
      return points[:, 1] * c + points[:, 2] * s

  if render_style == "depth":
      # Calculate global min/max depth for consistent normalization
      all_depths = []
      
      # Collect all active meshes
      active_meshes = []
      if batches.get(710): active_meshes.append(batches[710])
      if batches.get("other"): 
          batches["other"].translate((0, 0, 0.1), inplace=True)
          active_meshes.append(batches["other"])
      if batches.get(709): 
          batches[709].translate((0, 0, 0.2), inplace=True)
          active_meshes.append(batches[709])
      if batches.get(712): 
          batches[712].translate((0, 0, 0.3), inplace=True)
          active_meshes.append(batches[712])
          
      # Also add base plane for depth continuity
      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 1.0
      base_plane = pv.Plane(center=(0, 0, ground_z), i_size=ground_size, j_size=ground_size, direction=(0,0,1))
      # PyVista Plane creates a mesh. We need to make sure it covers the area.
      # Note: pv.Plane defaults to centered at 'center' with normal 'direction'.
      # i_size is X, j_size is Y.
      # But we used custom vertices before. Let's recreate that simple mesh to be sure.
      base_points = np.array([
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z]
      ])
      base_faces = [4, 0, 1, 2, 3]
      base_mesh = pv.PolyData(base_points, base_faces)
      active_meshes.insert(0, base_mesh)

      # Calculate bounds
      for mesh in active_meshes:
          d = get_depth_scalar(mesh.points, camera_elevation_deg)
          all_depths.append(d)
          # Store for later use to avoid recomputing
          mesh.point_data["depth"] = d
      
      if all_depths:
          full_concat = np.concatenate(all_depths)
          min_d, max_d = np.min(full_concat), np.max(full_concat)
          print(f"   📏 Depth Range: {min_d:.2f} to {max_d:.2f}")
          
          for mesh in active_meshes:
              plotter.add_mesh(
                  mesh, 
                  scalars="depth", 
                  cmap="gray_r",  # Close (Min) = White, Far (Max) = Black
                  clim=[min_d, max_d],
                  show_scalar_bar=False,
                  show_edges=False
              )
      
  # 6.1. Add satellite image as ground plane texture (Whitebox Mode)
  elif use_satellite and satellite_texture is not None:
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

  if render_style == "whitebox":
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
  
  if render_style == "depth":
      plotter.set_background("black")
      use_satellite = False  # Disable satellite texture in depth mode
  else:
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

  # Helper for depth calculation
  def get_depth_scalar(points, el_deg):
      rad = np.radians(el_deg)
      c = np.cos(rad)
      s = np.sin(rad)
      # Depth along view vector (assuming Azimuth aligned to Y)
      # Vector from cam to target is (0, cos, sin)
      return points[:, 1] * c + points[:, 2] * s

  if render_style == "depth":
      # Calculate global min/max depth for consistent normalization
      all_depths = []
      
      # Collect all active meshes
      active_meshes = []
      if batches.get(710): active_meshes.append(batches[710])
      if batches.get("other"): 
          batches["other"].translate((0, 0, 0.1), inplace=True)
          active_meshes.append(batches["other"])
      if batches.get(709): 
          batches[709].translate((0, 0, 0.2), inplace=True)
          active_meshes.append(batches[709])
      if batches.get(712): 
          batches[712].translate((0, 0, 0.3), inplace=True)
          active_meshes.append(batches[712])
          
      # Also add base plane for depth continuity
      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 1.0
      base_plane = pv.Plane(center=(0, 0, ground_z), i_size=ground_size, j_size=ground_size, direction=(0,0,1))
      # PyVista Plane creates a mesh. We need to make sure it covers the area.
      # Note: pv.Plane defaults to centered at 'center' with normal 'direction'.
      # i_size is X, j_size is Y.
      # But we used custom vertices before. Let's recreate that simple mesh to be sure.
      base_points = np.array([
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z]
      ])
      base_faces = [4, 0, 1, 2, 3]
      base_mesh = pv.PolyData(base_points, base_faces)
      active_meshes.insert(0, base_mesh)

      # Calculate bounds
      for mesh in active_meshes:
          d = get_depth_scalar(mesh.points, camera_elevation_deg)
          all_depths.append(d)
          # Store for later use to avoid recomputing
          mesh.point_data["depth"] = d
      
      if all_depths:
          full_concat = np.concatenate(all_depths)
          min_d, max_d = np.min(full_concat), np.max(full_concat)
          print(f"   📏 Depth Range: {min_d:.2f} to {max_d:.2f}")
          
          for mesh in active_meshes:
              plotter.add_mesh(
                  mesh, 
                  scalars="depth", 
                  cmap="gray_r",  # Close (Min) = White, Far (Max) = Black
                  clim=[min_d, max_d],
                  show_scalar_bar=False,
                  show_edges=False
              )
      
  # 6.1. Add satellite image as ground plane texture (Whitebox Mode)
  elif use_satellite and satellite_texture is not None:
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

  if render_style == "whitebox":
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
    render_style="depth",
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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
  
  if render_style == "depth":
      plotter.set_background("black")
      use_satellite = False  # Disable satellite texture in depth mode
  else:
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

  # Helper for depth calculation
  def get_depth_scalar(points, el_deg):
      rad = np.radians(el_deg)
      c = np.cos(rad)
      s = np.sin(rad)
      # Depth along view vector (assuming Azimuth aligned to Y)
      # Vector from cam to target is (0, cos, sin)
      return points[:, 1] * c + points[:, 2] * s

  if render_style == "depth":
      # Calculate global min/max depth for consistent normalization
      all_depths = []
      
      # Collect all active meshes
      active_meshes = []
      if batches.get(710): active_meshes.append(batches[710])
      if batches.get("other"): 
          batches["other"].translate((0, 0, 0.1), inplace=True)
          active_meshes.append(batches["other"])
      if batches.get(709): 
          batches[709].translate((0, 0, 0.2), inplace=True)
          active_meshes.append(batches[709])
      if batches.get(712): 
          batches[712].translate((0, 0, 0.3), inplace=True)
          active_meshes.append(batches[712])
          
      # Also add base plane for depth continuity
      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 1.0
      base_plane = pv.Plane(center=(0, 0, ground_z), i_size=ground_size, j_size=ground_size, direction=(0,0,1))
      # PyVista Plane creates a mesh. We need to make sure it covers the area.
      # Note: pv.Plane defaults to centered at 'center' with normal 'direction'.
      # i_size is X, j_size is Y.
      # But we used custom vertices before. Let's recreate that simple mesh to be sure.
      base_points = np.array([
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z]
      ])
      base_faces = [4, 0, 1, 2, 3]
      base_mesh = pv.PolyData(base_points, base_faces)
      active_meshes.insert(0, base_mesh)

      # Calculate bounds
      for mesh in active_meshes:
          d = get_depth_scalar(mesh.points, camera_elevation_deg)
          all_depths.append(d)
          # Store for later use to avoid recomputing
          mesh.point_data["depth"] = d
      
      if all_depths:
          full_concat = np.concatenate(all_depths)
          min_d, max_d = np.min(full_concat), np.max(full_concat)
          print(f"   📏 Depth Range: {min_d:.2f} to {max_d:.2f}")
          
          for mesh in active_meshes:
              plotter.add_mesh(
                  mesh, 
                  scalars="depth", 
                  cmap="gray_r",  # Close (Min) = White, Far (Max) = Black
                  clim=[min_d, max_d],
                  show_scalar_bar=False,
                  show_edges=False
              )
      
  # 6.1. Add satellite image as ground plane texture (Whitebox Mode)
  elif use_satellite and satellite_texture is not None:
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

  if render_style == "whitebox":
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
    render_style="depth",
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
    output_path="depth_map_test.png"
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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
  
  if render_style == "depth":
      plotter.set_background("black")
      use_satellite = False  # Disable satellite texture in depth mode
  else:
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

  # Helper for depth calculation
  def get_depth_scalar(points, el_deg):
      rad = np.radians(el_deg)
      c = np.cos(rad)
      s = np.sin(rad)
      # Depth along view vector (assuming Azimuth aligned to Y)
      # Vector from cam to target is (0, cos, sin)
      return points[:, 1] * c + points[:, 2] * s

  if render_style == "depth":
      # Calculate global min/max depth for consistent normalization
      all_depths = []
      
      # Collect all active meshes
      active_meshes = []
      if batches.get(710): active_meshes.append(batches[710])
      if batches.get("other"): 
          batches["other"].translate((0, 0, 0.1), inplace=True)
          active_meshes.append(batches["other"])
      if batches.get(709): 
          batches[709].translate((0, 0, 0.2), inplace=True)
          active_meshes.append(batches[709])
      if batches.get(712): 
          batches[712].translate((0, 0, 0.3), inplace=True)
          active_meshes.append(batches[712])
          
      # Also add base plane for depth continuity
      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 1.0
      base_plane = pv.Plane(center=(0, 0, ground_z), i_size=ground_size, j_size=ground_size, direction=(0,0,1))
      # PyVista Plane creates a mesh. We need to make sure it covers the area.
      # Note: pv.Plane defaults to centered at 'center' with normal 'direction'.
      # i_size is X, j_size is Y.
      # But we used custom vertices before. Let's recreate that simple mesh to be sure.
      base_points = np.array([
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z]
      ])
      base_faces = [4, 0, 1, 2, 3]
      base_mesh = pv.PolyData(base_points, base_faces)
      active_meshes.insert(0, base_mesh)

      # Calculate bounds
      for mesh in active_meshes:
          d = get_depth_scalar(mesh.points, camera_elevation_deg)
          all_depths.append(d)
          # Store for later use to avoid recomputing
          mesh.point_data["depth"] = d
      
      if all_depths:
          full_concat = np.concatenate(all_depths)
          min_d, max_d = np.min(full_concat), np.max(full_concat)
          print(f"   📏 Depth Range: {min_d:.2f} to {max_d:.2f}")
          
          for mesh in active_meshes:
              plotter.add_mesh(
                  mesh, 
                  scalars="depth", 
                  cmap="gray_r",  # Close (Min) = White, Far (Max) = Black
                  clim=[min_d, max_d],
                  show_scalar_bar=False,
                  show_edges=False
              )
      
  # 6.1. Add satellite image as ground plane texture (Whitebox Mode)
  elif use_satellite and satellite_texture is not None:
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

  if render_style == "whitebox":
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
  if footprint_lines and render_style != "depth":
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
    render_style="depth",
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
    output_path="depth_map_test.png"
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
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = (
  VIEW_HEIGHT_METERS / 2
)  # Parallel scale - lower = more zoomed in, higher = more zoomed out
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
  render_style="depth",  # "whitebox" or "depth"
  viewport_width=None,
  viewport_height=None,
  output_path=None,
  camera_elevation_deg=None,
  view_height_meters=None,
):
  """
  Render a 3D tile of NYC buildings.

  Args:
    lat, lon: Center coordinates
    size_meters: Size of the area to render
    orientation_deg: Rotation angle in degrees
    use_satellite: Whether to use satellite imagery
    render_style: "whitebox" for semantic colors, "depth" for depth map
    viewport_width, viewport_height: Render resolution
    output_path: If provided, save screenshot to this path instead of showing interactive viewer
    camera_elevation_deg: Camera elevation angle in degrees
    view_height_meters: Height of the view in world meters (affects zoom)
  """
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT
  camera_elevation_deg = (
    camera_elevation_deg if camera_elevation_deg is not None else CAMERA_ELEVATION_DEG
  )
  view_height_meters = (
    view_height_meters if view_height_meters is not None else VIEW_HEIGHT_METERS
  )

  # Calculate zoom based on view height
  camera_zoom = view_height_meters / 2

  # Auto-calculate required size based on view frustum to ensure coverage
  # Calculate view diagonal in world units
  if viewport_width and viewport_height:
    aspect_ratio = viewport_width / viewport_height
    view_width_meters = view_height_meters * aspect_ratio
    # Diagonal of the visible rectangular area
    diagonal_meters = (view_width_meters**2 + view_height_meters**2) ** 0.5

    # We need to fetch geometry that covers this diagonal (plus rotation)
    # The fetch logic uses a box of size = size_meters * 1.5
    # So we need size_meters * 1.5 >= diagonal_meters
    # Let's be conservative and ensure size_meters >= diagonal_meters
    if size_meters < diagonal_meters:
      print(
        f"⚠️  Provided size_meters ({size_meters}m) is too small for view ({view_height_meters}m height)."
      )
      print(f"    Auto-adjusting fetch size to {diagonal_meters:.0f}m.")
      size_meters = diagonal_meters

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
  plotter = pv.Plotter(
    window_size=(viewport_width, viewport_height), off_screen=off_screen
  )
  
  if render_style == "depth":
      plotter.set_background("black")
      use_satellite = False  # Disable satellite texture in depth mode
  else:
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

  # Helper for depth calculation
  def get_depth_scalar(points, el_deg):
      rad = np.radians(el_deg)
      c = np.cos(rad)
      s = np.sin(rad)
      # Depth along view vector (assuming Azimuth aligned to Y)
      # Vector from cam to target is (0, cos, sin)
      return points[:, 1] * c + points[:, 2] * s

  if render_style == "depth":
      # Calculate global min/max depth for consistent normalization
      all_depths = []
      
      # Collect all active meshes
      active_meshes = []
      if batches.get(710): active_meshes.append(batches[710])
      if batches.get("other"): 
          batches["other"].translate((0, 0, 0.1), inplace=True)
          active_meshes.append(batches["other"])
      if batches.get(709): 
          batches[709].translate((0, 0, 0.2), inplace=True)
          active_meshes.append(batches[709])
      if batches.get(712): 
          batches[712].translate((0, 0, 0.3), inplace=True)
          active_meshes.append(batches[712])
          
      # Also add base plane for depth continuity
      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 1.0
      base_plane = pv.Plane(center=(0, 0, ground_z), i_size=ground_size, j_size=ground_size, direction=(0,0,1))
      # PyVista Plane creates a mesh. We need to make sure it covers the area.
      # Note: pv.Plane defaults to centered at 'center' with normal 'direction'.
      # i_size is X, j_size is Y.
      # But we used custom vertices before. Let's recreate that simple mesh to be sure.
      base_points = np.array([
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z]
      ])
      base_faces = [4, 0, 1, 2, 3]
      base_mesh = pv.PolyData(base_points, base_faces)
      active_meshes.insert(0, base_mesh)

      # Calculate bounds
      for mesh in active_meshes:
          d = get_depth_scalar(mesh.points, camera_elevation_deg)
          all_depths.append(d)
          # Store for later use to avoid recomputing
          mesh.point_data["depth"] = d
      
      if all_depths:
          full_concat = np.concatenate(all_depths)
          min_d, max_d = np.min(full_concat), np.max(full_concat)
          print(f"   📏 Depth Range: {min_d:.2f} to {max_d:.2f}")
          
          for mesh in active_meshes:
              plotter.add_mesh(
                  mesh, 
                  scalars="depth", 
                  cmap="gray_r",  # Close (Min) = White, Far (Max) = Black
                  clim=[min_d, max_d],
                  show_scalar_bar=False,
                  show_edges=False
              )
      
  # 6.1. Add satellite image as ground plane texture (Whitebox Mode)
  elif use_satellite and satellite_texture is not None:
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

  if render_style == "whitebox":
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
  if footprint_lines and render_style != "depth":
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
  elevation_rad = np.radians(camera_elevation_deg)
  dist = 2000

  if abs(camera_elevation_deg) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    # Geometry is ALREADY rotated by ORIENTATION_DEG to align azimuth with +Y
    # So we just need to look from -Y direction (front) to see it correctly
    # Or more simply: position camera at (0, -dist, dist) relative to rotation?

    # Actually: The geometry is rotated around Z axis.
    # If we want to see the "front" of the rotated geometry, we should look from a fixed direction.
    # But current code calculates camera position based on azimuth AGAIN.
    # Let's simplify:
    # 1. Geometry is rotated so "North" (or azimuth dir) points to +Y
    # 2. We want to look at it from the "South" (or opposite azimuth)

    # Fixed camera position relative to the ROTATED geometry
    # Looking from "South" (-Y) towards "North" (+Y)
    # Elevation determines height (Z) and distance (Y)

    # Convert elevation to radians (0=horizontal, -90=down)
    elev_rad = np.radians(camera_elevation_deg)

    # Calculate Y and Z components based on elevation
    # Z is height: sin(elevation)
    # Y is distance back: cos(elevation)

    # Since elevation is negative (e.g. -45), sin is negative.
    # We want camera ABOVE ground (+Z), so negate sin(elev) * dist
    cz = -dist * np.sin(elev_rad)

    # We want camera "BACK" from the object.
    # If geometry is aligned to Y axis, we look from -Y.
    cy = -dist * np.cos(elev_rad)

    # X is 0 (centered)
    cx = 0

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = camera_zoom

  print(f"📷 Camera: Az={orientation_deg}° El={camera_elevation_deg}°")
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
    render_style="depth",
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

The code in `src/isometric_nyc/whitebox.py` has been updated to generate a depth map.

### Changes Made
1.  **Updated `render_tile` signature:** Added a `render_style` argument (defaulting to `"depth"` in `main()`).
2.  **Implemented Depth Calculation:** Added logic to calculate the distance of each vertex along the camera's view vector. This ensures accurate depth values even in isometric projection.
3.  **Depth Rendering Logic:**
    -   When `render_style="depth"`, the background is set to black.
    -   Meshes are colored by their calculated depth using an inverted grayscale colormap (`gray_r`), where **closer objects are brighter** and **further objects are darker** (standard for ControlNet depth inputs).
    -   Satellite textures are disabled in depth mode.
    -   A base ground plane is included to ensure no gaps in the depth map.
    -   Debug footprint outlines are disabled in depth mode to avoid artifacts.
4.  **Updated `main()`:** Adjusted the default call to use `render_style="depth"`.

You can now run the script to generate the depth map:
```bash
uv run python src/isometric_nyc/whitebox.py
```

```shellscript
uv run python src/isometric_nyc/whitebox.py
```

---

**User**

OK - now instead of the template being white for the unused pixels, use the pixels from @whitebox.py 

---

**Cursor**

```python
import argparse
import json
from pathlib import Path

from PIL import Image


def create_template(tile_dir_path: Path) -> None:
  """
  Creates a template.png for the given tile directory based on its neighbors' generation.png files.
  """
  # Validate tile directory
  if not tile_dir_path.exists():
    raise FileNotFoundError(f"Tile directory not found: {tile_dir_path}")

  # Load view.json
  view_json_path = tile_dir_path / "view.json"
  if not view_json_path.exists():
    raise FileNotFoundError(f"view.json not found in {tile_dir_path}")

  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  # Extract grid info
  row = view_json.get("row")
  col = view_json.get("col")
  width_px = view_json.get("width_px", 1024)
  height_px = view_json.get("height_px", 1024)

  if row is None or col is None:
    print("Error: view.json missing 'row' or 'col' fields.")
    return

  # Parent directory (plan directory)
  plan_dir = tile_dir_path.parent

  # Define neighbors: (row_offset, col_offset, name)
  # We assume neighbors are stored in directories named "{row:03d}_{col:03d}"
  neighbors = [
    (0, -1, "left"),
    (0, 1, "right"),
    (-1, 0, "top"),
    (1, 0, "bottom"),
  ]

  # Create canvas
  # Try to load whitebox.png to use as background
  whitebox_path = tile_dir_path / "whitebox.png"
  if whitebox_path.exists():
    try:
      with Image.open(whitebox_path) as wb:
        # Ensure it matches dimensions
        if wb.size != (width_px, height_px):
          print(f"Resizing whitebox from {wb.size} to {(width_px, height_px)}")
          wb = wb.resize((width_px, height_px), Image.Resampling.LANCZOS)
        canvas = wb.convert("RGB")
        print(f"Using whitebox.png as template background")
    except Exception as e:
      print(f"Error loading whitebox.png: {e}. Using white background.")
      canvas = Image.new("RGB", (width_px, height_px), "white")
  else:
    print("whitebox.png not found. Using white background.")
    canvas = Image.new("RGB", (width_px, height_px), "white")

  found_neighbor = False

  for r_off, c_off, name in neighbors:
    n_row = row + r_off
    n_col = col + c_off

    # Construct neighbor directory name
    n_dir_name = f"{n_row:03d}_{n_col:03d}"
    n_dir_path = plan_dir / n_dir_name

    n_gen_path = n_dir_path / "generation.png"

    if n_gen_path.exists():
      print(f"Found {name} neighbor at {n_dir_name}")
      try:
        with Image.open(n_gen_path) as n_img:
          # Ensure neighbor image matches expected size (or resize?)
          # Assuming neighbors are same size for now
          if n_img.size != (width_px, height_px):
            print(
              f"Warning: Neighbor {name} size {n_img.size} does not match expected {(width_px, height_px)}. Skipping."
            )
            continue

          found_neighbor = True

          # Logic for pasting neighbor parts
          # Overlap is 50% (half width/height)
          half_w = width_px // 2
          half_h = height_px // 2

          if name == "left":
            # Take Right half of neighbor -> Paste to Left half of canvas
            # Crop box: (left, top, right, bottom)
            region = n_img.crop((half_w, 0, width_px, height_px))
            canvas.paste(region, (0, 0))

          elif name == "right":
            # Take Left half of neighbor -> Paste to Right half of canvas
            region = n_img.crop((0, 0, half_w, height_px))
            canvas.paste(region, (half_w, 0))

          elif name == "top":
            # Take Bottom half of neighbor -> Paste to Top half of canvas
            region = n_img.crop((0, half_h, width_px, height_px))
            canvas.paste(region, (0, 0))

          elif name == "bottom":
            # Take Top half of neighbor -> Paste to Bottom half of canvas
            region = n_img.crop((0, 0, width_px, half_h))
            canvas.paste(region, (0, half_h))

      except Exception as e:
        print(f"Error processing neighbor {name}: {e}")

  if found_neighbor:
    output_path = tile_dir_path / "template.png"
    canvas.save(output_path)
    print(f"Created template at {output_path}")
  else:
    print("No neighbors with 'generation.png' found. No template created.")


def main():
  parser = argparse.ArgumentParser(
    description="Create a template image from neighbor tiles."
  )
  parser.add_argument(
    "--tile_dir",
    help="Directory containing the tile assets (view.json)",
  )

  args = parser.parse_args()
  tile_dir = Path(args.tile_dir)

  create_template(tile_dir)


if __name__ == "__main__":
  main()

```

```python
    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated, superimposed on the <whitebox> geometry.
    
    You must continue the generation by filling in the unfinished areas (where the whitebox geometry is visible), ensuring that the colors and textures seamlessly integrate with the existing neighbor parts.
    
    The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template. Your generation for those areas should match them, but the final output will composite them back in.
    """
```

```python
    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated, superimposed on the <whitebox> geometry.
    
    You must continue the generation by filling in the unfinished areas (where the whitebox geometry is visible), ensuring that the colors and textures seamlessly integrate with the existing neighbor parts.
    
    The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template. Your generation for those areas should match them, but the final output will composite them back in.
    """
```

```python
import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from isometric_nyc.create_template import create_template

MODEL_NAME = "gemini-2.5-flash"
IMAGE_MODEL_NAME = "gemini-3-pro-image-preview"
REFERENCE_IMAGE_NAME = "style_a.png"


def generate_tile(
  tile_dir_path: Path,
  references_dir_path: Path,
  downscale_factor: float = 4.0,
  skip_description: bool = False,
) -> None:
  """
  Generates an isometric pixel art image for the given tile directory using Gemini.
  """
  # Load environment variables
  load_dotenv()
  gemini_api_key = os.getenv("GEMINI_API_KEY")
  if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

  client = genai.Client(api_key=gemini_api_key)

  # Validate directories
  if not tile_dir_path.exists():
    raise FileNotFoundError(f"Tile directory not found: {tile_dir_path}")
  if not references_dir_path.exists():
    raise FileNotFoundError(f"References directory not found: {references_dir_path}")

  # Load view.json
  view_json_path = tile_dir_path / "view.json"
  if not view_json_path.exists():
    raise FileNotFoundError(f"view.json not found in {tile_dir_path}")

  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  latitude = view_json.get("lat")
  longitude = view_json.get("lon")

  print(f"Processing tile at {latitude}, {longitude}...")

  # Full-size render path (used for feature checklist)
  render_path_full = tile_dir_path / "render.png"
  if not render_path_full.exists():
    raise FileNotFoundError(f"render.png not found in {tile_dir_path}")

  # Downscaled render path (used for image generation)
  render_path_downscaled: Path | None = None
  if downscale_factor > 1.0:
    print(f"Downscaling render.png by factor of {downscale_factor}...")
    with Image.open(render_path_full) as img:
      new_width = int(img.width / downscale_factor)
      new_height = int(img.height / downscale_factor)
      resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

      # Save to temp file
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        resized_img.save(tmp_file.name)
        render_path_downscaled = Path(tmp_file.name)
        print(f"Saved downscaled image to {render_path_downscaled}")

  # Define prompts (copied from nano-banana.py)
  pixel_art_techniques = """
    PIXEL ART TECHNIQUES (Apply Aggressively): Translate the textures and details from <render> into the following pixel art conventions:

    Heavy Dithering: All gradients, shadows, and complex textures (like the arena roof, asphalt, and building facades) must be rendered using visible cross-hatch or Bayer pattern dithering. There should be NO smooth color transitions.

    Indexed Color Palette: Use a strictly limited palette (e.g., 256 colors). Colors should be flat and distinct, typical of late 90s gaming hardware.

    Aliased Edges: Every object, building, and car must have a sharp, jagged, non-anti-aliased pixel outline.

    Tiled Textures: Building windows and brickwork should look like repeating grids of pixel tiles, not realistic materials.

    Sprites: The cars in the parking lot and on the streets must be rendered as tiny, distinct pixel art sprites, not blurry dots.
    """.strip()

  description_prompt = f"""
    You are an advanced image analysis agent. Your task is to generate a checklist of no more than ten features from the attached overhead isometric render of a section of New York City. These features will be used to populate a prompt for an image generation model that will transform the input image into a stylized isometric pixel art style image in the style of SimCity 3000 based on the constraints. It's critical that the model adhere to the colors and textures of the guide image, and that's what the checklist should aim to ensure.

    The following instructions will also be provided to the model for adhering to the pixel art style - you may emphasize any of these points to ensure that the model most accurately adheres to the colors, styles, and features of the reference image. If you recognize any of the buildings or features, please refer to them by name.

    The image is an overhead isometric render of the following coordinates:
    latitude: {latitude}
    longitude: {longitude}

    {pixel_art_techniques}

    Generate *ONLY* the list of features, nothing more.
    """

  checklist = ""
  if not skip_description:
    # Use full-size render for feature analysis (better detail)
    print("Uploading full-size render.png for analysis...")
    render_ref_full = client.files.upload(file=render_path_full)

    # Generate the checklist of features
    print("Generating feature checklist...")
    checklist_response = client.models.generate_content(
      model=MODEL_NAME,
      contents=[
        render_ref_full,
        description_prompt,
      ],
      config=types.GenerateContentConfig(
        response_modalities=["TEXT"],
      ),
    )
    checklist = checklist_response.text
    print(f"Checklist generated:\n{checklist}")
  else:
    print("Skipping description generation...")
    checklist = "Follow the style of the reference images."

  # Prepare generation prompt
  generation_prompt = """
    (((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

    <render> is the 3D render of the city - use this image as a reference for the details, textures, colors, and lighting of the buildings, but DO NOT  downsample the pixels - we want to use the style of <reference>.

    <reference> is a reference image for the style of SimCity 3000 pixel art - you MUST use this style for the pixel art generation.

    Use the white masses in <whitebox> as the blueprint for all building shapes and locations. Check carefully to make sure every building in <whitebox> and <render> is present in the generation, and ensure that the colors and textures of the buildings are correct.
    """

  # Upload assets for generation
  whitebox_path = tile_dir_path / "whitebox.png"
  if not whitebox_path.exists():
    raise FileNotFoundError(f"whitebox.png not found in {tile_dir_path}")

  reference_path = references_dir_path / REFERENCE_IMAGE_NAME
  if not reference_path.exists():
    raise FileNotFoundError(
      f"{REFERENCE_IMAGE_NAME} not found in {references_dir_path}"
    )

  print("Uploading assets for generation...")
  whitebox_ref = client.files.upload(file=whitebox_path)

  # Upload render for generation (use downscaled version if available)
  render_path_for_generation = render_path_downscaled or render_path_full
  print(
    f"Uploading render.png for generation (downscaled: {render_path_downscaled is not None})..."
  )
  render_ref = client.files.upload(file=render_path_for_generation)

  reference_ref = client.files.upload(file=reference_path)
  whitebox_prefix = "This is a whitebox geometry of isometric render of a section of New York City. We'll refer to this as <whitebox>."
  render_prefix = "This is a rendered view of the 3D building data using Google 3D tiles API. We'll refer to this as <render>."
  reference_prefix = "This is a reference image for the style of SimCity 3000 pixel art. We'll refer to this as <reference>."

  contents = [
    whitebox_prefix,
    whitebox_ref,
    render_prefix,
    render_ref,
    reference_prefix,
    reference_ref,
    generation_prompt,
  ]

  # Create template from neighbors if they exist
  print("Checking for neighbors to create template...")
  create_template(tile_dir_path)

  # Check for template.png
  template_path = tile_dir_path / "template.png"
  if template_path.exists():
    print("Found template.png, uploading and updating prompt...")
    template_prefix = "This is a template image that contains parts of neighboring tiles that have already been generated. We'll refer to this as <template>."
    template_ref = client.files.upload(file=template_path)
    contents.append(template_prefix)
    contents.append(template_ref)

    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. You must continue the generation by filling in the unrendered whitebox pixels in the template, ensuring that the colors and textures of the buildings are correct.
    
    The non-whitebox pixels MUST be preserved exactly as they appear in the template, and you MUST adhere to the style of the non-whitebox pixels in the template.
    """

  print("Generating pixel art image...")

  response = client.models.generate_content(
    model=IMAGE_MODEL_NAME,
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  output_path = tile_dir_path / "generation.png"
  if output_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = tile_dir_path / f"generation_{timestamp}.png"

  image_saved = False
  for part in response.parts:
    if part.text is not None:
      print(f"Response text: {part.text}")

    # Check if part has image (the structure might differ slightly in library versions)
    # The notebook used: elif image:= part.as_image():
    try:
      image = part.as_image()
      if image:
        print(f"Saving image to {output_path}...")
        image.save(output_path)

        # If template exists, composite the generation with the template
        if template_path.exists():
          # Save raw generation first
          raw_output_path = tile_dir_path / "raw_generation.png"
          image.save(raw_output_path)
          print(f"Saved raw generation to {raw_output_path}")

          print("Compositing generation with template...")
          try:
            # Open the saved image
            with Image.open(output_path) as img_file:
              generated_img = img_file.convert("RGBA")

            with Image.open(template_path) as tmpl:
              # Ensure size matches
              if generated_img.size != tmpl.size:
                print(
                  f"Warning: Generated image size {generated_img.size} differs from template size {tmpl.size}. Resizing generation."
                )
                generated_img = generated_img.resize(
                  tmpl.size, Image.Resampling.LANCZOS
                )

              # Convert template to RGBA
              tmpl = tmpl.convert("RGBA")

              # Prepare mask logic
              whitebox_data = None
              if whitebox_path.exists():
                try:
                    with Image.open(whitebox_path) as wb:
                        if wb.size != tmpl.size:
                             wb = wb.resize(tmpl.size, Image.Resampling.LANCZOS)
                        whitebox_data = list(wb.convert("RGBA").getdata())
                except Exception as e:
                    print(f"Warning: Could not load whitebox for compositing mask: {e}")

              # Make background pixels in template transparent
              # Background is defined as matching whitebox.png or being white (fallback)
              datas = tmpl.getdata()
              new_data = []
              
              for i, item in enumerate(datas):
                is_background = False
                
                if whitebox_data:
                    # Check if matches whitebox pixel (ignoring alpha of whitebox if present)
                    # item is (R, G, B, A)
                    wb_pixel = whitebox_data[i]
                    # Allow small tolerance for compression artifacts
                    if (abs(item[0] - wb_pixel[0]) < 5 and 
                        abs(item[1] - wb_pixel[1]) < 5 and 
                        abs(item[2] - wb_pixel[2]) < 5):
                        is_background = True
                else:
                    # Fallback: Check for white
                    if item[0] > 250 and item[1] > 250 and item[2] > 250:
                        is_background = True

                if is_background:
                  new_data.append((255, 255, 255, 0))  # Transparent
                else:
                  new_data.append(item)  # Keep original pixel

              tmpl.putdata(new_data)

              # Composite: Paste template over generation
              generated_img.paste(tmpl, (0, 0), tmpl)

              # Save back to output path
              generated_img.save(output_path)
          except Exception as e:
            print(f"Error during compositing: {e}")

        image_saved = True
    except Exception as e:
      print(f"Could not save image part: {e}")

  if not image_saved:
    # Sometimes the image is in a different property or needs handling
    # If no image part, maybe it failed to generate image
    print("No image generated in response.")
  else:
    print("Generation complete.")

  # Cleanup temp file if it was created
  if render_path_downscaled and str(render_path_downscaled).startswith(
    tempfile.gettempdir()
  ):
    try:
      os.unlink(render_path_downscaled)
      print(f"Cleaned up temp file {render_path_downscaled}")
    except Exception as e:
      print(f"Error cleaning up temp file: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art for a tile."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )
  parser.add_argument(
    "--references_dir",
    default="references",
    help="Directory containing reference images (simcity.jpg)",
  )
  parser.add_argument(
    "--downscale",
    type=float,
    default=2.0,
    help="Factor to downscale the render image by (e.g. 2.0 for half size)",
  )
  parser.add_argument(
    "--skip-description",
    action="store_true",
    help="Skip the description generation step",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  references_dir = Path(args.references_dir)

  if not references_dir.is_absolute():
    # Assume relative to current working directory
    references_dir = Path.cwd() / references_dir

  generate_tile(
    tile_dir, references_dir, args.downscale, skip_description=args.skip_description
  )


if __name__ == "__main__":
  main()

```

```python
    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated, superimposed on the <whitebox> geometry.
    
    You must continue the generation by filling in the unfinished areas (where the whitebox geometry is visible), ensuring that the colors and textures seamlessly integrate with the existing neighbor parts.
    
    The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template. Your generation for those areas should match them, but the final output will composite them back in.
    """
```

```python
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated, superimposed on the <whitebox> geometry.
    
    You must continue the generation by filling in the unfinished areas (where the whitebox geometry is visible), ensuring that the colors and textures seamlessly integrate with the existing neighbor parts.
    
    The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template. Your generation for those areas should match them, but the final output will composite them back in.
    """
```

```python
import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from isometric_nyc.create_template import create_template

MODEL_NAME = "gemini-2.5-flash"
IMAGE_MODEL_NAME = "gemini-3-pro-image-preview"
REFERENCE_IMAGE_NAME = "style_a.png"


def generate_tile(
  tile_dir_path: Path,
  references_dir_path: Path,
  downscale_factor: float = 4.0,
  skip_description: bool = False,
) -> None:
  """
  Generates an isometric pixel art image for the given tile directory using Gemini.
  """
  # Load environment variables
  load_dotenv()
  gemini_api_key = os.getenv("GEMINI_API_KEY")
  if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

  client = genai.Client(api_key=gemini_api_key)

  # Validate directories
  if not tile_dir_path.exists():
    raise FileNotFoundError(f"Tile directory not found: {tile_dir_path}")
  if not references_dir_path.exists():
    raise FileNotFoundError(f"References directory not found: {references_dir_path}")

  # Load view.json
  view_json_path = tile_dir_path / "view.json"
  if not view_json_path.exists():
    raise FileNotFoundError(f"view.json not found in {tile_dir_path}")

  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  latitude = view_json.get("lat")
  longitude = view_json.get("lon")

  print(f"Processing tile at {latitude}, {longitude}...")

  # Full-size render path (used for feature checklist)
  render_path_full = tile_dir_path / "render.png"
  if not render_path_full.exists():
    raise FileNotFoundError(f"render.png not found in {tile_dir_path}")

  # Downscaled render path (used for image generation)
  render_path_downscaled: Path | None = None
  if downscale_factor > 1.0:
    print(f"Downscaling render.png by factor of {downscale_factor}...")
    with Image.open(render_path_full) as img:
      new_width = int(img.width / downscale_factor)
      new_height = int(img.height / downscale_factor)
      resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

      # Save to temp file
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        resized_img.save(tmp_file.name)
        render_path_downscaled = Path(tmp_file.name)
        print(f"Saved downscaled image to {render_path_downscaled}")

  # Define prompts (copied from nano-banana.py)
  pixel_art_techniques = """
    PIXEL ART TECHNIQUES (Apply Aggressively): Translate the textures and details from <render> into the following pixel art conventions:

    Heavy Dithering: All gradients, shadows, and complex textures (like the arena roof, asphalt, and building facades) must be rendered using visible cross-hatch or Bayer pattern dithering. There should be NO smooth color transitions.

    Indexed Color Palette: Use a strictly limited palette (e.g., 256 colors). Colors should be flat and distinct, typical of late 90s gaming hardware.

    Aliased Edges: Every object, building, and car must have a sharp, jagged, non-anti-aliased pixel outline.

    Tiled Textures: Building windows and brickwork should look like repeating grids of pixel tiles, not realistic materials.

    Sprites: The cars in the parking lot and on the streets must be rendered as tiny, distinct pixel art sprites, not blurry dots.
    """.strip()

  description_prompt = f"""
    You are an advanced image analysis agent. Your task is to generate a checklist of no more than ten features from the attached overhead isometric render of a section of New York City. These features will be used to populate a prompt for an image generation model that will transform the input image into a stylized isometric pixel art style image in the style of SimCity 3000 based on the constraints. It's critical that the model adhere to the colors and textures of the guide image, and that's what the checklist should aim to ensure.

    The following instructions will also be provided to the model for adhering to the pixel art style - you may emphasize any of these points to ensure that the model most accurately adheres to the colors, styles, and features of the reference image. If you recognize any of the buildings or features, please refer to them by name.

    The image is an overhead isometric render of the following coordinates:
    latitude: {latitude}
    longitude: {longitude}

    {pixel_art_techniques}

    Generate *ONLY* the list of features, nothing more.
    """

  checklist = ""
  if not skip_description:
    # Use full-size render for feature analysis (better detail)
    print("Uploading full-size render.png for analysis...")
    render_ref_full = client.files.upload(file=render_path_full)

    # Generate the checklist of features
    print("Generating feature checklist...")
    checklist_response = client.models.generate_content(
      model=MODEL_NAME,
      contents=[
        render_ref_full,
        description_prompt,
      ],
      config=types.GenerateContentConfig(
        response_modalities=["TEXT"],
      ),
    )
    checklist = checklist_response.text
    print(f"Checklist generated:\n{checklist}")
  else:
    print("Skipping description generation...")
    checklist = "Follow the style of the reference images."

  # Prepare generation prompt
  generation_prompt = """
    (((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

    <render> is the 3D render of the city - use this image as a reference for the details, textures, colors, and lighting of the buildings, but DO NOT  downsample the pixels - we want to use the style of <reference>.

    <reference> is a reference image for the style of SimCity 3000 pixel art - you MUST use this style for the pixel art generation.

    Use the white masses in <whitebox> as the blueprint for all building shapes and locations. Check carefully to make sure every building in <whitebox> and <render> is present in the generation, and ensure that the colors and textures of the buildings are correct.
    """

  # Upload assets for generation
  whitebox_path = tile_dir_path / "whitebox.png"
  if not whitebox_path.exists():
    raise FileNotFoundError(f"whitebox.png not found in {tile_dir_path}")

  reference_path = references_dir_path / REFERENCE_IMAGE_NAME
  if not reference_path.exists():
    raise FileNotFoundError(
      f"{REFERENCE_IMAGE_NAME} not found in {references_dir_path}"
    )

  print("Uploading assets for generation...")
  whitebox_ref = client.files.upload(file=whitebox_path)

  # Upload render for generation (use downscaled version if available)
  render_path_for_generation = render_path_downscaled or render_path_full
  print(
    f"Uploading render.png for generation (downscaled: {render_path_downscaled is not None})..."
  )
  render_ref = client.files.upload(file=render_path_for_generation)

  reference_ref = client.files.upload(file=reference_path)
  whitebox_prefix = "This is a whitebox geometry of isometric render of a section of New York City. We'll refer to this as <whitebox>."
  render_prefix = "This is a rendered view of the 3D building data using Google 3D tiles API. We'll refer to this as <render>."
  reference_prefix = "This is a reference image for the style of SimCity 3000 pixel art. We'll refer to this as <reference>."

  contents = [
    whitebox_prefix,
    whitebox_ref,
    render_prefix,
    render_ref,
    reference_prefix,
    reference_ref,
    generation_prompt,
  ]

  # Create template from neighbors if they exist
  print("Checking for neighbors to create template...")
  create_template(tile_dir_path)

  # Check for template.png
  template_path = tile_dir_path / "template.png"
  if template_path.exists():
    print("Found template.png, uploading and updating prompt...")
    template_prefix = "This is a template image that contains parts of neighboring tiles that have already been generated. We'll refer to this as <template>."
    template_ref = client.files.upload(file=template_path)
    contents.append(template_prefix)
    contents.append(template_ref)

    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated, superimposed on the <whitebox> geometry.
    
    You must continue the generation by filling in the unfinished areas (where the whitebox geometry is visible), ensuring that the colors and textures seamlessly integrate with the existing neighbor parts.
    
    The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template. Your generation for those areas should match them, but the final output will composite them back in.
    """

  print("Generating pixel art image...")

  response = client.models.generate_content(
    model=IMAGE_MODEL_NAME,
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  output_path = tile_dir_path / "generation.png"
  if output_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = tile_dir_path / f"generation_{timestamp}.png"

  image_saved = False
  for part in response.parts:
    if part.text is not None:
      print(f"Response text: {part.text}")

    # Check if part has image (the structure might differ slightly in library versions)
    # The notebook used: elif image:= part.as_image():
    try:
      image = part.as_image()
      if image:
        print(f"Saving image to {output_path}...")
        image.save(output_path)

        # If template exists, composite the generation with the template
        if template_path.exists():
          # Save raw generation first
          raw_output_path = tile_dir_path / "raw_generation.png"
          image.save(raw_output_path)
          print(f"Saved raw generation to {raw_output_path}")

          print("Compositing generation with template...")
          try:
            # Open the saved image
            with Image.open(output_path) as img_file:
              generated_img = img_file.convert("RGBA")

            with Image.open(template_path) as tmpl:
              # Ensure size matches
              if generated_img.size != tmpl.size:
                print(
                  f"Warning: Generated image size {generated_img.size} differs from template size {tmpl.size}. Resizing generation."
                )
                generated_img = generated_img.resize(
                  tmpl.size, Image.Resampling.LANCZOS
                )

              # Convert template to RGBA
              tmpl = tmpl.convert("RGBA")

              # Prepare mask logic
              whitebox_data = None
              if whitebox_path.exists():
                try:
                    with Image.open(whitebox_path) as wb:
                        if wb.size != tmpl.size:
                             wb = wb.resize(tmpl.size, Image.Resampling.LANCZOS)
                        whitebox_data = list(wb.convert("RGBA").getdata())
                except Exception as e:
                    print(f"Warning: Could not load whitebox for compositing mask: {e}")

              # Make background pixels in template transparent
              # Background is defined as matching whitebox.png or being white (fallback)
              datas = tmpl.getdata()
              new_data = []
              
              for i, item in enumerate(datas):
                is_background = False
                
                if whitebox_data:
                    # Check if matches whitebox pixel (ignoring alpha of whitebox if present)
                    # item is (R, G, B, A)
                    wb_pixel = whitebox_data[i]
                    # Allow small tolerance for compression artifacts
                    if (abs(item[0] - wb_pixel[0]) < 5 and 
                        abs(item[1] - wb_pixel[1]) < 5 and 
                        abs(item[2] - wb_pixel[2]) < 5):
                        is_background = True
                else:
                    # Fallback: Check for white
                    if item[0] > 250 and item[1] > 250 and item[2] > 250:
                        is_background = True

                if is_background:
                  new_data.append((255, 255, 255, 0))  # Transparent
                else:
                  new_data.append(item)  # Keep original pixel

              tmpl.putdata(new_data)

              # Composite: Paste template over generation
              generated_img.paste(tmpl, (0, 0), tmpl)

              # Save back to output path
              generated_img.save(output_path)
          except Exception as e:
            print(f"Error during compositing: {e}")

        image_saved = True
    except Exception as e:
      print(f"Could not save image part: {e}")

  if not image_saved:
    # Sometimes the image is in a different property or needs handling
    # If no image part, maybe it failed to generate image
    print("No image generated in response.")
  else:
    print("Generation complete.")

  # Cleanup temp file if it was created
  if render_path_downscaled and str(render_path_downscaled).startswith(
    tempfile.gettempdir()
  ):
    try:
      os.unlink(render_path_downscaled)
      print(f"Cleaned up temp file {render_path_downscaled}")
    except Exception as e:
      print(f"Error cleaning up temp file: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art for a tile."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )
  parser.add_argument(
    "--references_dir",
    default="references",
    help="Directory containing reference images (simcity.jpg)",
  )
  parser.add_argument(
    "--downscale",
    type=float,
    default=2.0,
    help="Factor to downscale the render image by (e.g. 2.0 for half size)",
  )
  parser.add_argument(
    "--skip-description",
    action="store_true",
    help="Skip the description generation step",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  references_dir = Path(args.references_dir)

  if not references_dir.is_absolute():
    # Assume relative to current working directory
    references_dir = Path.cwd() / references_dir

  generate_tile(
    tile_dir, references_dir, args.downscale, skip_description=args.skip_description
  )


if __name__ == "__main__":
  main()

```

The changes have been applied:

1.  **`src/isometric_nyc/create_template.py`**: Now initializes the canvas using `whitebox.png` (the depth map or massing) as the background instead of plain white, before pasting neighbor tiles on top.
2.  **`src/isometric_nyc/generate_tile.py`**:
    *   **Prompt:** Updated to instruct the model that the template contains "unfinished areas (where the whitebox geometry is visible)" instead of "white pixels".
    *   **Compositing:** Updated logic to treat pixels matching `whitebox.png` as transparent during the final composite, ensuring the newly generated buildings fill those gaps while preserving the exact pixels of neighboring tiles.

You can now proceed with tile generation, and the "unused" areas of the template will show the whitebox guide to the model.

---

**User**

Can you fix the @src/isometric_nyc/generate_tile.py script to properly splice in the generated pixels into the template? It's leaving them white

---

**Cursor**

```python
import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from isometric_nyc.create_template import create_template

MODEL_NAME = "gemini-2.5-flash"
IMAGE_MODEL_NAME = "gemini-3-pro-image-preview"
REFERENCE_IMAGE_NAME = "style_c.png"


def generate_tile(
  tile_dir_path: Path,
  references_dir_path: Path,
  downscale_factor: float = 4.0,
  skip_description: bool = False,
) -> None:
  """
  Generates an isometric pixel art image for the given tile directory using Gemini.
  """
  # Load environment variables
  load_dotenv()
  gemini_api_key = os.getenv("GEMINI_API_KEY")
  if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

  client = genai.Client(api_key=gemini_api_key)

  # Validate directories
  if not tile_dir_path.exists():
    raise FileNotFoundError(f"Tile directory not found: {tile_dir_path}")
  if not references_dir_path.exists():
    raise FileNotFoundError(f"References directory not found: {references_dir_path}")

  # Load view.json
  view_json_path = tile_dir_path / "view.json"
  if not view_json_path.exists():
    raise FileNotFoundError(f"view.json not found in {tile_dir_path}")

  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  latitude = view_json.get("lat")
  longitude = view_json.get("lon")

  print(f"Processing tile at {latitude}, {longitude}...")

  # Full-size render path (used for feature checklist)
  render_path_full = tile_dir_path / "render.png"
  if not render_path_full.exists():
    raise FileNotFoundError(f"render.png not found in {tile_dir_path}")

  # Downscaled render path (used for image generation)
  render_path_downscaled: Path | None = None
  if downscale_factor > 1.0:
    print(f"Downscaling render.png by factor of {downscale_factor}...")
    with Image.open(render_path_full) as img:
      new_width = int(img.width / downscale_factor)
      new_height = int(img.height / downscale_factor)
      resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

      # Save to temp file
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        resized_img.save(tmp_file.name)
        render_path_downscaled = Path(tmp_file.name)
        print(f"Saved downscaled image to {render_path_downscaled}")

  # Define prompts (copied from nano-banana.py)
  pixel_art_techniques = """
    PIXEL ART TECHNIQUES (Apply Aggressively): Translate the textures and details from <render> into the following pixel art conventions:

    Heavy Dithering: All gradients, shadows, and complex textures (like the arena roof, asphalt, and building facades) must be rendered using visible cross-hatch or Bayer pattern dithering. There should be NO smooth color transitions.

    Indexed Color Palette: Use a strictly limited palette (e.g., 256 colors). Colors should be flat and distinct, typical of late 90s gaming hardware.

    Aliased Edges: Every object, building, and car must have a sharp, jagged, non-anti-aliased pixel outline.

    Tiled Textures: Building windows and brickwork should look like repeating grids of pixel tiles, not realistic materials.

    Sprites: The cars in the parking lot and on the streets must be rendered as tiny, distinct pixel art sprites, not blurry dots.
    """.strip()

  description_prompt = f"""
    You are an advanced image analysis agent. Your task is to generate a checklist of no more than ten features from the attached overhead isometric render of a section of New York City. These features will be used to populate a prompt for an image generation model that will transform the input image into a stylized isometric pixel art style image in the style of SimCity 3000 based on the constraints. It's critical that the model adhere to the colors and textures of the guide image, and that's what the checklist should aim to ensure.

    The following instructions will also be provided to the model for adhering to the pixel art style - you may emphasize any of these points to ensure that the model most accurately adheres to the colors, styles, and features of the reference image. If you recognize any of the buildings or features, please refer to them by name.

    The image is an overhead isometric render of the following coordinates:
    latitude: {latitude}
    longitude: {longitude}

    {pixel_art_techniques}

    Generate *ONLY* the list of features, nothing more.
    """

  checklist = ""
  if not skip_description:
    # Use full-size render for feature analysis (better detail)
    print("Uploading full-size render.png for analysis...")
    render_ref_full = client.files.upload(file=render_path_full)

    # Generate the checklist of features
    print("Generating feature checklist...")
    checklist_response = client.models.generate_content(
      model=MODEL_NAME,
      contents=[
        render_ref_full,
        description_prompt,
      ],
      config=types.GenerateContentConfig(
        response_modalities=["TEXT"],
      ),
    )
    checklist = checklist_response.text
    print(f"Checklist generated:\n{checklist}")
  else:
    print("Skipping description generation...")
    checklist = "Follow the style of the reference images."

  # Prepare generation prompt
  generation_prompt = """
    (((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

    <render> is the 3D render of the city - use this image as a reference for the details, textures, colors, and lighting of the buildings, but DO NOT  downsample the pixels - we want to use the style of <reference>.

    <reference> is a reference image for the style of SimCity 3000 pixel art - you MUST use this style for the pixel art generation.

    Use the white masses in <whitebox> as the blueprint for all building shapes and locations. Check carefully to make sure every building in <whitebox> and <render> is present in the generation, and ensure that the colors and textures of the buildings are correct.
    """

  # Upload assets for generation
  whitebox_path = tile_dir_path / "whitebox.png"
  if not whitebox_path.exists():
    raise FileNotFoundError(f"whitebox.png not found in {tile_dir_path}")

  reference_path = references_dir_path / REFERENCE_IMAGE_NAME
  if not reference_path.exists():
    raise FileNotFoundError(
      f"{REFERENCE_IMAGE_NAME} not found in {references_dir_path}"
    )

  print("Uploading assets for generation...")
  whitebox_ref = client.files.upload(file=whitebox_path)

  # Upload render for generation (use downscaled version if available)
  render_path_for_generation = render_path_downscaled or render_path_full
  print(
    f"Uploading render.png for generation (downscaled: {render_path_downscaled is not None})..."
  )
  render_ref = client.files.upload(file=render_path_for_generation)

  reference_ref = client.files.upload(file=reference_path)
  whitebox_prefix = "This is a whitebox geometry of isometric render of a section of New York City. We'll refer to this as <whitebox>."
  render_prefix = "This is a rendered view of the 3D building data using Google 3D tiles API. We'll refer to this as <render>."
  reference_prefix = "This is a reference image for the style of SimCity 3000 pixel art. We'll refer to this as <reference>."

  contents = [
    whitebox_prefix,
    whitebox_ref,
    render_prefix,
    render_ref,
    reference_prefix,
    reference_ref,
    generation_prompt,
  ]

  # Create template from neighbors if they exist
  print("Checking for neighbors to create template...")
  create_template(tile_dir_path)

  # Check for template.png
  template_path = tile_dir_path / "template.png"
  if template_path.exists():
    print("Found template.png, uploading and updating prompt...")
    template_prefix = "This is a template image that contains parts of neighboring tiles that have already been generated. We'll refer to this as <template>."
    template_ref = client.files.upload(file=template_path)
    contents.append(template_prefix)
    contents.append(template_ref)

    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated. The white areas are empty and need to be filled.
    
    You must continue the generation by filling in the white areas, using the <whitebox> geometry and <render> as guides.
    
    The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template.
    """

  print("Generating pixel art image...")

  response = client.models.generate_content(
    model=IMAGE_MODEL_NAME,
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  output_path = tile_dir_path / "generation.png"
  if output_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = tile_dir_path / f"generation_{timestamp}.png"

  image_saved = False
  for part in response.parts:
    if part.text is not None:
      print(f"Response text: {part.text}")

    # Check if part has image (the structure might differ slightly in library versions)
    # The notebook used: elif image:= part.as_image():
    try:
      image = part.as_image()
      if image:
        print(f"Saving image to {output_path}...")
        image.save(output_path)

        # If template exists, composite the generation with the template
        if template_path.exists():
          # Save raw generation first
          raw_output_path = tile_dir_path / "raw_generation.png"
          image.save(raw_output_path)
          print(f"Saved raw generation to {raw_output_path}")

          print("Compositing generation with template...")
          try:
            # Open the saved image
            with Image.open(output_path) as img_file:
              generated_img = img_file.convert("RGBA")

            with Image.open(template_path) as tmpl:
              # Ensure size matches
              if generated_img.size != tmpl.size:
                print(
                  f"Warning: Generated image size {generated_img.size} differs from template size {tmpl.size}. Resizing generation."
                )
                generated_img = generated_img.resize(
                  tmpl.size, Image.Resampling.LANCZOS
                )

              # Convert template to RGBA
              tmpl = tmpl.convert("RGBA")

              # Prepare mask logic
              whitebox_data = None
              if whitebox_path.exists():
                try:
                  with Image.open(whitebox_path) as wb:
                    if wb.size != tmpl.size:
                      wb = wb.resize(tmpl.size, Image.Resampling.LANCZOS)
                    whitebox_data = list(wb.convert("RGBA").getdata())
                except Exception as e:
                  print(f"Warning: Could not load whitebox for compositing mask: {e}")

              # Make background pixels in template transparent
              # Background is defined as matching whitebox.png or being white (fallback)
              datas = tmpl.getdata()
              new_data = []

              for i, item in enumerate(datas):
                is_background = False

                if whitebox_data:
                  # Check if matches whitebox pixel (ignoring alpha of whitebox if present)
                  # item is (R, G, B, A)
                  wb_pixel = whitebox_data[i]
                  # Allow small tolerance for compression artifacts
                  if (
                    abs(item[0] - wb_pixel[0]) < 5
                    and abs(item[1] - wb_pixel[1]) < 5
                    and abs(item[2] - wb_pixel[2]) < 5
                  ):
                    is_background = True
                else:
                  # Fallback: Check for white
                  if item[0] > 250 and item[1] > 250 and item[2] > 250:
                    is_background = True

                if is_background:
                  new_data.append((255, 255, 255, 0))  # Transparent
                else:
                  new_data.append(item)  # Keep original pixel

              tmpl.putdata(new_data)

              # Composite: Paste template over generation
              generated_img.paste(tmpl, (0, 0), tmpl)

              # Save back to output path
              generated_img.save(output_path)
          except Exception as e:
            print(f"Error during compositing: {e}")

        image_saved = True
    except Exception as e:
      print(f"Could not save image part: {e}")

  if not image_saved:
    # Sometimes the image is in a different property or needs handling
    # If no image part, maybe it failed to generate image
    print("No image generated in response.")
  else:
    print("Generation complete.")

  # Cleanup temp file if it was created
  if render_path_downscaled and str(render_path_downscaled).startswith(
    tempfile.gettempdir()
  ):
    try:
      os.unlink(render_path_downscaled)
      print(f"Cleaned up temp file {render_path_downscaled}")
    except Exception as e:
      print(f"Error cleaning up temp file: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art for a tile."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )
  parser.add_argument(
    "--references_dir",
    default="references",
    help="Directory containing reference images (simcity.jpg)",
  )
  parser.add_argument(
    "--downscale",
    type=float,
    default=2.0,
    help="Factor to downscale the render image by (e.g. 2.0 for half size)",
  )
  parser.add_argument(
    "--skip-description",
    action="store_true",
    help="Skip the description generation step",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  references_dir = Path(args.references_dir)

  if not references_dir.is_absolute():
    # Assume relative to current working directory
    references_dir = Path.cwd() / references_dir

  generate_tile(
    tile_dir, references_dir, args.downscale, skip_description=args.skip_description
  )


if __name__ == "__main__":
  main()

```

```python
import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from isometric_nyc.create_template import create_template

MODEL_NAME = "gemini-2.5-flash"
IMAGE_MODEL_NAME = "gemini-3-pro-image-preview"
REFERENCE_IMAGE_NAME = "style_c.png"


def generate_tile(
  tile_dir_path: Path,
  references_dir_path: Path,
  downscale_factor: float = 4.0,
  skip_description: bool = False,
) -> None:
  """
  Generates an isometric pixel art image for the given tile directory using Gemini.
  """
  # Load environment variables
  load_dotenv()
  gemini_api_key = os.getenv("GEMINI_API_KEY")
  if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

  client = genai.Client(api_key=gemini_api_key)

  # Validate directories
  if not tile_dir_path.exists():
    raise FileNotFoundError(f"Tile directory not found: {tile_dir_path}")
  if not references_dir_path.exists():
    raise FileNotFoundError(f"References directory not found: {references_dir_path}")

  # Load view.json
  view_json_path = tile_dir_path / "view.json"
  if not view_json_path.exists():
    raise FileNotFoundError(f"view.json not found in {tile_dir_path}")

  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  latitude = view_json.get("lat")
  longitude = view_json.get("lon")

  print(f"Processing tile at {latitude}, {longitude}...")

  # Full-size render path (used for feature checklist)
  render_path_full = tile_dir_path / "render.png"
  if not render_path_full.exists():
    raise FileNotFoundError(f"render.png not found in {tile_dir_path}")

  # Downscaled render path (used for image generation)
  render_path_downscaled: Path | None = None
  if downscale_factor > 1.0:
    print(f"Downscaling render.png by factor of {downscale_factor}...")
    with Image.open(render_path_full) as img:
      new_width = int(img.width / downscale_factor)
      new_height = int(img.height / downscale_factor)
      resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

      # Save to temp file
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        resized_img.save(tmp_file.name)
        render_path_downscaled = Path(tmp_file.name)
        print(f"Saved downscaled image to {render_path_downscaled}")

  # Define prompts (copied from nano-banana.py)
  pixel_art_techniques = """
    PIXEL ART TECHNIQUES (Apply Aggressively): Translate the textures and details from <render> into the following pixel art conventions:

    Heavy Dithering: All gradients, shadows, and complex textures (like the arena roof, asphalt, and building facades) must be rendered using visible cross-hatch or Bayer pattern dithering. There should be NO smooth color transitions.

    Indexed Color Palette: Use a strictly limited palette (e.g., 256 colors). Colors should be flat and distinct, typical of late 90s gaming hardware.

    Aliased Edges: Every object, building, and car must have a sharp, jagged, non-anti-aliased pixel outline.

    Tiled Textures: Building windows and brickwork should look like repeating grids of pixel tiles, not realistic materials.

    Sprites: The cars in the parking lot and on the streets must be rendered as tiny, distinct pixel art sprites, not blurry dots.
    """.strip()

  description_prompt = f"""
    You are an advanced image analysis agent. Your task is to generate a checklist of no more than ten features from the attached overhead isometric render of a section of New York City. These features will be used to populate a prompt for an image generation model that will transform the input image into a stylized isometric pixel art style image in the style of SimCity 3000 based on the constraints. It's critical that the model adhere to the colors and textures of the guide image, and that's what the checklist should aim to ensure.

    The following instructions will also be provided to the model for adhering to the pixel art style - you may emphasize any of these points to ensure that the model most accurately adheres to the colors, styles, and features of the reference image. If you recognize any of the buildings or features, please refer to them by name.

    The image is an overhead isometric render of the following coordinates:
    latitude: {latitude}
    longitude: {longitude}

    {pixel_art_techniques}

    Generate *ONLY* the list of features, nothing more.
    """

  checklist = ""
  if not skip_description:
    # Use full-size render for feature analysis (better detail)
    print("Uploading full-size render.png for analysis...")
    render_ref_full = client.files.upload(file=render_path_full)

    # Generate the checklist of features
    print("Generating feature checklist...")
    checklist_response = client.models.generate_content(
      model=MODEL_NAME,
      contents=[
        render_ref_full,
        description_prompt,
      ],
      config=types.GenerateContentConfig(
        response_modalities=["TEXT"],
      ),
    )
    checklist = checklist_response.text
    print(f"Checklist generated:\n{checklist}")
  else:
    print("Skipping description generation...")
    checklist = "Follow the style of the reference images."

  # Prepare generation prompt
  generation_prompt = """
    (((Isometric pixel art:1.6))), (classic city builder game aesthetic:1.5), (orthographic projection:1.5), (detailed 32-bit graphics:1.4), (sharp crisp edges:1.3), (dense urban cityscape:1.3), (complex architectural geometry:1.2), (directional hard shadows:1.2), neutral color palette, bird's-eye view.

    <render> is the 3D render of the city - use this image as a reference for the details, textures, colors, and lighting of the buildings, but DO NOT  downsample the pixels - we want to use the style of <reference>.

    <reference> is a reference image for the style of SimCity 3000 pixel art - you MUST use this style for the pixel art generation.

    Use the white masses in <whitebox> as the blueprint for all building shapes and locations. Check carefully to make sure every building in <whitebox> and <render> is present in the generation, and ensure that the colors and textures of the buildings are correct.
    """

  # Upload assets for generation
  whitebox_path = tile_dir_path / "whitebox.png"
  if not whitebox_path.exists():
    raise FileNotFoundError(f"whitebox.png not found in {tile_dir_path}")

  reference_path = references_dir_path / REFERENCE_IMAGE_NAME
  if not reference_path.exists():
    raise FileNotFoundError(
      f"{REFERENCE_IMAGE_NAME} not found in {references_dir_path}"
    )

  print("Uploading assets for generation...")
  whitebox_ref = client.files.upload(file=whitebox_path)

  # Upload render for generation (use downscaled version if available)
  render_path_for_generation = render_path_downscaled or render_path_full
  print(
    f"Uploading render.png for generation (downscaled: {render_path_downscaled is not None})..."
  )
  render_ref = client.files.upload(file=render_path_for_generation)

  reference_ref = client.files.upload(file=reference_path)
  whitebox_prefix = "This is a whitebox geometry of isometric render of a section of New York City. We'll refer to this as <whitebox>."
  render_prefix = "This is a rendered view of the 3D building data using Google 3D tiles API. We'll refer to this as <render>."
  reference_prefix = "This is a reference image for the style of SimCity 3000 pixel art. We'll refer to this as <reference>."

  contents = [
    whitebox_prefix,
    whitebox_ref,
    render_prefix,
    render_ref,
    reference_prefix,
    reference_ref,
    generation_prompt,
  ]

  # Create template from neighbors if they exist
  print("Checking for neighbors to create template...")
  create_template(tile_dir_path)

  # Check for template.png
  template_path = tile_dir_path / "template.png"
  if template_path.exists():
    print("Found template.png, uploading and updating prompt...")
    template_prefix = "This is a template image that contains parts of neighboring tiles that have already been generated. We'll refer to this as <template>."
    template_ref = client.files.upload(file=template_path)
    contents.append(template_prefix)
    contents.append(template_ref)

    # Update prompt to include template instructions
    # Assuming template is the 4th image (index 3)
    contents[0] += """

    TEMPLATE INSTRUCTIONS:
    The last image provided is a template image <template>. It contains parts of neighboring tiles that have already been generated. The white areas are empty and need to be filled.
    
    You must continue the generation by filling in the white areas, using the <whitebox> geometry and <render> as guides.
    
    The existing neighbor parts (the colorful pixel art sections) MUST be preserved exactly as they appear in the template.
    """

  print("Generating pixel art image...")

  response = client.models.generate_content(
    model=IMAGE_MODEL_NAME,
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  output_path = tile_dir_path / "generation.png"
  if output_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = tile_dir_path / f"generation_{timestamp}.png"

  image_saved = False
  for part in response.parts:
    if part.text is not None:
      print(f"Response text: {part.text}")

    # Check if part has image (the structure might differ slightly in library versions)
    # The notebook used: elif image:= part.as_image():
    try:
      image = part.as_image()
      if image:
        print(f"Saving image to {output_path}...")
        image.save(output_path)

        # If template exists, composite the generation with the template
        if template_path.exists():
          # Save raw generation first
          raw_output_path = tile_dir_path / "raw_generation.png"
          image.save(raw_output_path)
          print(f"Saved raw generation to {raw_output_path}")

          print("Compositing generation with template...")
          try:
            # Open the saved image
            with Image.open(output_path) as img_file:
              generated_img = img_file.convert("RGBA")

            with Image.open(template_path) as tmpl:
              # Ensure size matches
              if generated_img.size != tmpl.size:
                print(
                  f"Warning: Generated image size {generated_img.size} differs from template size {tmpl.size}. Resizing generation."
                )
                generated_img = generated_img.resize(
                  tmpl.size, Image.Resampling.LANCZOS
                )

              # Convert template to RGBA
              tmpl = tmpl.convert("RGBA")

              # Make white pixels in template transparent
              datas = tmpl.getdata()
              new_data = []

              for item in datas:
                # Check for white (allowing slight variance)
                # Pure white (255, 255, 255) is the background color from create_template.py
                if item[0] > 250 and item[1] > 250 and item[2] > 250:
                  new_data.append((255, 255, 255, 0))  # Transparent
                else:
                  new_data.append(item)  # Keep original pixel

              tmpl.putdata(new_data)

              # Composite: Paste template over generation
              generated_img.paste(tmpl, (0, 0), tmpl)

              # Save back to output path
              generated_img.save(output_path)
          except Exception as e:
            print(f"Error during compositing: {e}")

        image_saved = True
    except Exception as e:
      print(f"Could not save image part: {e}")

  if not image_saved:
    # Sometimes the image is in a different property or needs handling
    # If no image part, maybe it failed to generate image
    print("No image generated in response.")
  else:
    print("Generation complete.")

  # Cleanup temp file if it was created
  if render_path_downscaled and str(render_path_downscaled).startswith(
    tempfile.gettempdir()
  ):
    try:
      os.unlink(render_path_downscaled)
      print(f"Cleaned up temp file {render_path_downscaled}")
    except Exception as e:
      print(f"Error cleaning up temp file: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art for a tile."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, whitebox.png, render.png)",
  )
  parser.add_argument(
    "--references_dir",
    default="references",
    help="Directory containing reference images (simcity.jpg)",
  )
  parser.add_argument(
    "--downscale",
    type=float,
    default=2.0,
    help="Factor to downscale the render image by (e.g. 2.0 for half size)",
  )
  parser.add_argument(
    "--skip-description",
    action="store_true",
    help="Skip the description generation step",
  )

  args = parser.parse_args()

  tile_dir = Path(args.tile_dir)
  references_dir = Path(args.references_dir)

  if not references_dir.is_absolute():
    # Assume relative to current working directory
    references_dir = Path.cwd() / references_dir

  generate_tile(
    tile_dir, references_dir, args.downscale, skip_description=args.skip_description
  )


if __name__ == "__main__":
  main()

```

I've updated `src/isometric_nyc/generate_tile.py` to fix the compositing issue.

### Changes Made
1.  **Reverted Compositing Logic:** The script no longer tries to use `whitebox.png` as a mask (which was causing the template's white background to override the generation because it didn't match the depth map values). Instead, it now treats any near-white pixels (`> 250, 250, 250`) in the template as transparent, allowing the newly generated pixels underneath to show through.
2.  **Updated Prompt:** The prompt instructions were updated to match the reverted template format (white background instead of whitebox background), instructing the model to fill in the "white areas".

This should correctly splice the generated pixels into the empty (white) areas of the template.

