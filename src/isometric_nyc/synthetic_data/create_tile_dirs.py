import argparse
import json
import os
from pathlib import Path


def generate_plan(
  *,
  name: str,
  view_json: dict,
):
  locations = view_json["locations"]
  base_dir = Path("synthetic_data") / "tiles" / name

  # Ensure base directory exists
  if not base_dir.exists():
    base_dir.mkdir(parents=True)

  # Calculate offsets for each tile
  # Tile (0,0) is centered at start_lat, start_lon.
  # Tile (r, c) is offset.

  for i, location in enumerate(locations):
    # Generate ID
    tile_dir = base_dir / f"{i:03d}"
    tile_dir.mkdir(exist_ok=True)

    # Create view.json
    view_data = {
      "name": location["name"],
      "lat": location["lat"],
      "lon": location["lon"],
      "camera_azimuth_degrees": view_json["camera_azimuth_degrees"],
      "camera_elevation_degrees": view_json["camera_elevation_degrees"],
      "width_px": view_json["width_px"],
      "height_px": view_json["height_px"],
      "view_height_meters": view_json["view_height_meters"],
    }

    with open(tile_dir / "view.json", "w") as f:
      json.dump(view_data, f, indent=2)


def main():
  parser = argparse.ArgumentParser(
    description="Generate synthetic data tiles from a view.json file"
  )
  parser.add_argument("--name", required=True, help="Name of the plan")
  args = parser.parse_args()

  view_json_path = os.path.join(os.path.dirname(__file__), "view.json")
  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  generate_plan(
    name=args.name,
    view_json=view_json,
  )


if __name__ == "__main__":
  main()
