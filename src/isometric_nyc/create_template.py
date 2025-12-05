import argparse
import json
from pathlib import Path

from PIL import Image


def create_template(tile_dir_path: Path, use_render: bool = False) -> None:
  """
  Creates a template.png for the given tile directory based on its neighbors' generation.png files.

  Args:
    tile_dir_path: Path to the tile directory containing view.json
    use_render: If True, use the right half of render.png instead of white pixels
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
  tile_step = view_json.get("tile_step", 0.5)

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

  # If use_render is True, paste the right half of render.png onto the canvas
  if use_render:
    render_path = tile_dir_path / "render.png"
    if render_path.exists():
      try:
        with Image.open(render_path) as render_img:
          if render_img.size != (width_px, height_px):
            print(
              f"Warning: render.png size {render_img.size} does not match expected {(width_px, height_px)}. Skipping render."
            )
          else:
            # Crop the right half of render.png and paste it
            half_width = width_px // 2
            right_half = render_img.crop((half_width, 0, width_px, height_px))
            canvas.paste(right_half, (half_width, 0))
            print("Added right half of render.png to template")
      except Exception as e:
        print(f"Error loading render.png: {e}")
    else:
      print(
        f"Warning: --use-render specified but render.png not found at {render_path}"
      )

  found_neighbor = False

  # Calculate step sizes in pixels
  step_w = int(width_px * tile_step)
  step_h = int(height_px * tile_step)

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
          if name == "left":
            # Neighbor is shifted LEFT by step_w
            # We want the part of neighbor that overlaps with us
            # Neighbor's right-most part [step_w, width] overlaps our [0, width-step_w]
            region = n_img.crop((step_w, 0, width_px, height_px))
            canvas.paste(region, (0, 0))

          elif name == "right":
            # Neighbor is shifted RIGHT by step_w
            # We want the part of neighbor that overlaps with us
            # Neighbor's left-most part [0, width-step_w] overlaps our [step_w, width]
            region = n_img.crop((0, 0, width_px - step_w, height_px))
            canvas.paste(region, (step_w, 0))

          elif name == "top":
            # Neighbor is shifted UP by step_h
            # We want the part of neighbor that overlaps with us
            # Neighbor's bottom-most part [step_h, height] overlaps our [0, height-step_h]
            region = n_img.crop((0, step_h, width_px, height_px))
            canvas.paste(region, (0, 0))

          elif name == "bottom":
            # Neighbor is shifted DOWN by step_h
            # We want the part of neighbor that overlaps with us
            # Neighbor's top-most part [0, height-step_h] overlaps our [step_h, height]
            region = n_img.crop((0, 0, width_px, height_px - step_h))
            canvas.paste(region, (0, step_h))

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
  parser.add_argument(
    "--use-render",
    action="store_true",
    help="Use the right half of render.png instead of white pixels",
  )

  args = parser.parse_args()
  tile_dir = Path(args.tile_dir)

  create_template(tile_dir, use_render=args.use_render)


if __name__ == "__main__":
  main()
