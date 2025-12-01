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

  # Create blank canvas (white)
  # "white pixels on the right" implies white background
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
