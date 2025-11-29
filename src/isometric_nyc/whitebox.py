import io
import os

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
SATELLITE_ZOOM = 19  # Google Maps zoom level
SATELLITE_SIZE = "2048x2048"  # Image resolution
GROUND_Z = -0.5  # Height of ground plane (adjust if not aligned)

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
      # Get high-res satellite image
      # Fetch a larger image to cover the expanded bounding box (1.5x size)
      satellite_image = fetch_satellite_image(
        lat, lon, zoom=SATELLITE_ZOOM, size=SATELLITE_SIZE
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

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
    else:
      batches[class_id] = None

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Create a ground plane mesh that matches our view area
    # The satellite image from Google Maps is centered on our lat/lon
    # and rotated to align with the orientation

    # Create a simple quad for the ground plane
    # Coordinates in our transformed space (centered at 0,0)
    # Position at GROUND_Z to be just below ground surfaces but visible
    ground_size = size_meters * 1.5
    half_size = ground_size / 2

    # Define the four corners of the ground plane
    ground_points = np.array(
      [
        [-half_size, -half_size, GROUND_Z],  # Bottom-left
        [half_size, -half_size, GROUND_Z],  # Bottom-right
        [half_size, half_size, GROUND_Z],  # Top-right
        [-half_size, half_size, GROUND_Z],  # Top-left
      ]
    )

    # Define the face (a quad with 4 vertices)
    ground_faces = [4, 0, 1, 2, 3]

    # Create the mesh
    ground_mesh = pv.PolyData(ground_points, ground_faces)

    # Add texture coordinates (UV mapping) using texture_map_to_plane
    # This automatically generates texture coordinates for a planar surface
    ground_mesh = ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, GROUND_Z),
      point_u=(half_size, -half_size, GROUND_Z),
      point_v=(-half_size, half_size, GROUND_Z),
    )

    # Convert RGB to texture format PyVista expects
    texture = pv.Texture(satellite_texture)

    plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Satellite ground plane added")

  if batches.get(710):  # GroundSurface (original geometry)
    # Only render if not using satellite, to avoid overlap
    if not use_satellite:
      plotter.add_mesh(batches[710], color=COLORS[710], show_edges=False)

  if batches.get("other"):  # Roads / Misc
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(batches["other"], color=COLORS["road"], show_edges=False)

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(batches[709], color=COLORS[709], show_edges=False)

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(batches[712], color=COLORS[712], show_edges=False)

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
