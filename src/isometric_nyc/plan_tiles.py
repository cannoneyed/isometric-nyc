import argparse
import json
import math
from pathlib import Path


def calculate_offset(
  lat_center,
  lon_center,
  shift_x_px,
  shift_y_px,
  view_height_meters,
  viewport_height_px,
  azimuth_deg,
  elevation_deg,
):
  """
  Calculate the new lat/lon center after shifting the view by shift_x_px and shift_y_px.

  Logic derived from whitebox.py camera setup:
  - Isometric view with parallel projection.
  - view_height_meters corresponds to viewport_height_px.
  - Camera is rotated by azimuth_deg.
  - Elevation determines the projection angle.
  """

  # 1. Convert pixel shift to meters in screen space
  # Note: We assume square pixels and aspect ratio handling logic from whitebox.py
  # In whitebox.py: parallel_scale = view_height_meters / 2
  # parallel_scale represents half the height.
  # So viewport_height_px corresponds to view_height_meters.

  meters_per_pixel = view_height_meters / viewport_height_px

  shift_right_meters = shift_x_px * meters_per_pixel
  shift_up_meters = shift_y_px * meters_per_pixel

  # 2. Convert screen movement to rotated world movement
  # Based on derivation:
  # Move_X (Right) = shift_right_meters
  # Move_Y (Projected Up) = -shift_up_meters / sin(elev)

  elev_rad = math.radians(elevation_deg)
  sin_elev = math.sin(elev_rad)

  # Avoid division by zero (though elevation should be negative, e.g. -45)
  if abs(sin_elev) < 1e-6:
    raise ValueError(f"Elevation {elevation_deg} is too close to 0/180.")

  delta_rot_x = shift_right_meters
  delta_rot_y = -shift_up_meters / sin_elev

  # 3. Rotate back to global coordinates (East, North)
  # Rotation was:
  # x_rot = x_global * cos(a) - y_global * sin(a)
  # y_rot = x_global * sin(a) + y_global * cos(a)
  # Inverse:
  # x_global = x_rot * cos(a) + y_rot * sin(a)
  # y_global = -x_rot * sin(a) + y_rot * cos(a)

  # Note: In whitebox.py, orientation_deg (azimuth) is used.
  # "Positive to align azimuth direction with +Y (top of view)"
  # This implies standard rotation matrix?
  # Let's check whitebox.py line 605:
  # x * cos - y * sin
  # x * sin + y * cos
  # This is standard CCW rotation if a is angle from X to X'.
  # But usually Azimuth 0 is North (+Y).
  # If Azimuth is 45 (NE), we want North to rotate to top-left?
  # Let's trust the math derived earlier assuming standard rotation matrix usage in whitebox.py.

  azimuth_rad = math.radians(azimuth_deg)
  cos_a = math.cos(azimuth_rad)
  sin_a = math.sin(azimuth_rad)

  delta_east_meters = delta_rot_x * cos_a + delta_rot_y * sin_a
  delta_north_meters = -delta_rot_x * sin_a + delta_rot_y * cos_a

  # 4. Convert meters to lat/lon
  # 1 deg lat approx 111,111 meters
  # 1 deg lon approx 111,111 * cos(lat) meters

  delta_lat = delta_north_meters / 111111.0
  delta_lon = delta_east_meters / (111111.0 * math.cos(math.radians(lat_center)))

  return lat_center + delta_lat, lon_center + delta_lon


def generate_plan(
  name,
  start_lat,
  start_lon,
  rows,
  cols,
  view_height_meters,
  azimuth,
  elevation,
  width_px=1024,
  height_px=1024,
  tile_step=0.5,
):
  base_dir = Path("tile_plans") / name

  print(f"Generating plan '{name}' with {rows}x{cols} tiles...")
  print(f"Base Center: {start_lat}, {start_lon}")
  print(f"Output Directory: {base_dir}")
  print(f"Tile Step: {tile_step}")

  # Ensure base directory exists
  if not base_dir.exists():
    base_dir.mkdir(parents=True)

  # Calculate offsets for each tile
  # Tile (0,0) is centered at start_lat, start_lon.
  # Tile (r, c) is offset.

  for r in range(rows):
    for c in range(cols):
      # Calculate pixel shift relative to (0,0)
      # Columns shift RIGHT (+)
      # Rows shift DOWN (Screen Y -)

      shift_x_px = c * (width_px * tile_step)
      shift_y_px = -r * (height_px * tile_step)

      lat, lon = calculate_offset(
        start_lat,
        start_lon,
        shift_x_px,
        shift_y_px,
        view_height_meters,
        height_px,
        azimuth,
        elevation,
      )

      # Generate ID
      # Use row/col format: "rrr_ccc"
      view_id = f"{r:03d}_{c:03d}"

      # Create directory
      tile_dir = base_dir / view_id
      tile_dir.mkdir(exist_ok=True)

      # Create view.json
      view_data = {
        "name": f"{name} - {r}x{c}",
        "lat": lat,
        "lon": lon,
        "camera_azimuth_degrees": azimuth,
        "camera_elevation_degrees": elevation,
        "width_px": width_px,
        "height_px": height_px,
        "view_height_meters": view_height_meters,
        "tile_step": tile_step,
        "grid_pos": {"row": r, "col": c},
        "row": r,
        "col": c,
      }

      with open(tile_dir / "view.json", "w") as f:
        json.dump(view_data, f, indent=2)

      print(f"  Tile [{r},{c}] -> {view_id}: {lat:.6f}, {lon:.6f}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate a tile plan for Isometric NYC."
  )
  parser.add_argument("--name", required=True, help="Name of the plan")
  parser.add_argument(
    "--lat", type=float, required=True, help="Latitude of the top-left tile center"
  )
  parser.add_argument(
    "--lon", type=float, required=True, help="Longitude of the top-left tile center"
  )
  parser.add_argument("--rows", type=int, default=1, help="Number of rows")
  parser.add_argument("--cols", type=int, default=1, help="Number of columns")
  parser.add_argument(
    "--view-height", type=float, default=300, help="View height in meters (coverage)"
  )
  parser.add_argument(
    "--azimuth", type=float, default=-15, help="Camera azimuth in degrees"
  )
  parser.add_argument(
    "--elevation", type=float, default=-45, help="Camera elevation in degrees"
  )
  parser.add_argument("--width", type=int, default=1024, help="Tile width in pixels")
  parser.add_argument("--height", type=int, default=1024, help="Tile height in pixels")
  parser.add_argument(
    "--tile-step",
    type=float,
    default=0.5,
    help="Fraction of tile size to step for next tile (default: 0.5)",
  )

  args = parser.parse_args()

  generate_plan(
    args.name,
    args.lat,
    args.lon,
    args.rows,
    args.cols,
    args.view_height,
    args.azimuth,
    args.elevation,
    args.width,
    args.height,
    args.tile_step,
  )


if __name__ == "__main__":
  main()
