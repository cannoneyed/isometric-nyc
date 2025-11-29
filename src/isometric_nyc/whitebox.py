import io
import os

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

# Constants (Times Square)
LAT = 40.7580
LON = -73.9855
SIZE_METERS = 300
ORIENTATION_DEG = 29

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail but potentially angled)
SATELLITE_TILE_SIZE = (
  "640x640"  # Size per tile (max 640x640 for free, 2048x2048 for premium)
)
SATELLITE_GRID = 3  # Fetch NxN grid of tiles (e.g., 3x3 = 9 tiles)
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 0.5  # Opacity for buildings (1.0=solid, 0.0=transparent)

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


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API."""
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))
  return image


def fetch_satellite_tiles(center_lat, center_lon, zoom, tile_size, grid_size):
  """
  Fetch a grid of satellite tiles and stitch them together.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size: Size of each tile (e.g., "640x640")
    grid_size: NxN grid (e.g., 3 for 3x3 = 9 tiles)

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  tile_px = int(tile_size.split("x")[0])

  # Calculate the offset in degrees for each tile
  # At zoom level z, the number of pixels per degree of longitude at equator is:
  # pixels_per_degree = (256 * 2^z) / 360
  # We need to adjust for latitude using Mercator projection
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  # 1 degree latitude ≈ 111,111 meters
  # 1 degree longitude at this lat ≈ 111,111 * cos(lat) meters
  meters_per_tile = tile_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Fetching {grid_size}x{grid_size} tiles, "
    f"offset: {lat_offset_per_tile:.5f}°, {lon_offset_per_tile:.5f}°"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m")

  # Create canvas for stitching
  canvas_size = tile_px * grid_size
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

      print(f"   📥 Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_px
      y = row * tile_px
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


def render_tile(lat, lon, size_meters=300, orientation_deg=29, use_satellite=True):
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
      # Get high-res satellite image by stitching tiles
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, SATELLITE_GRID
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      ground_size = size_meters * 1.5

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
      # Negative because we're rotating the texture, not the geometry
      satellite_image = satellite_image.rotate(
        -orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(1280, 720))
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
  angle_rad = np.radians(-orientation_deg)
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

  # 7. SimCity 3000 Camera Setup
  plotter.camera.enable_parallel_projection()
  alpha = np.arctan(0.7)
  beta = np.radians(30)
  dist = 2000

  cx = dist * np.cos(alpha) * np.sin(beta)
  cy = dist * np.cos(alpha) * np.cos(beta)
  cz = dist * np.sin(alpha)

  plotter.camera.position = (cx, cy, cz)
  plotter.camera.focal_point = (0, 0, 0)

  # Default values
  plotter.camera.up = (0, 0, 1)
  plotter.camera.parallel_scale = size_meters / 4

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
  )


if __name__ == "__main__":
  main()
