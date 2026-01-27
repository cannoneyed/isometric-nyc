import argparse
import json
import re
from pathlib import Path

from PIL import Image


def load_view_config(path: Path):
  with open(path, "r") as f:
    return json.load(f)


def stitch_images(tile_dir: Path, filename: str, output_name: str):
  """
  Stitch images from tile subdirectories into a single large image.

  Args:
      tile_dir: Directory containing tile subdirectories (000_000, 000_001, etc.)
      filename: Name of the image file to look for in each subdirectory (e.g., "render.png", "whitebox.png")
      output_name: Name of the final stitched image file
  """

  # 1. Scan directory for tiles and load metadata
  tiles = []

  # Look for directories matching rrr_ccc pattern
  pattern = re.compile(r"(\d{3})_(\d{3})")

  for path in tile_dir.iterdir():
    if path.is_dir():
      match = pattern.match(path.name)
      if match:
        row, col = int(match.group(1)), int(match.group(2))

        # Check if image exists
        img_path = path / filename
        if not img_path.exists():
          print(f"⚠️  Missing {filename} in {path.name}, skipping...")
          continue

        # Load view config for dimensions
        view_json_path = path / "view.json"
        if not view_json_path.exists():
          print(f"⚠️  Missing view.json in {path.name}, skipping...")
          continue

        view_config = load_view_config(view_json_path)

        tiles.append(
          {
            "row": row,
            "col": col,
            "path": img_path,
            "width": view_config["width_px"],
            "height": view_config["height_px"],
          }
        )

  if not tiles:
    print(f"❌ No tiles found with {filename} in {tile_dir}")
    return

  # 2. Determine grid size
  max_row = max(t["row"] for t in tiles)
  max_col = max(t["col"] for t in tiles)

  # Assume all tiles have the same dimensions (take first one)
  tile_w = tiles[0]["width"]
  tile_h = tiles[0]["height"]

  # Overlap is 50% of tile size
  # Stride is 50% of tile size
  stride_x = tile_w // 2
  stride_y = tile_h // 2

  # Calculate canvas size
  # Width = (cols * stride_x) + remaining_half_tile
  # Actually:
  # Col 0 starts at 0
  # Col 1 starts at stride_x
  # Col C starts at C * stride_x
  # Width needed = (max_col * stride_x) + tile_w

  canvas_w = (max_col * stride_x) + tile_w
  canvas_h = (max_row * stride_y) + tile_h

  print(f"🧩 Stitching {len(tiles)} tiles into {canvas_w}x{canvas_h} canvas...")
  print(f"   Grid: {max_row + 1} rows x {max_col + 1} cols")
  print(f"   Tile size: {tile_w}x{tile_h}")

  # 3. Create canvas
  canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

  # 4. Paste tiles
  # Order matters? Maybe draw top-left first.
  # Actually, we want to blend them if there's alpha, but these are opaque renders usually.
  # For opaque renders with 50% overlap:
  # The overlap area should theoretically be identical.
  # We can just paste them in order.

  for tile in tiles:
    r, c = tile["row"], tile["col"]
    img = Image.open(tile["path"])

    x = c * stride_x
    y = r * stride_y

    # If image is not same size as config (e.g. high DPI), resize?
    if img.size != (tile_w, tile_h):
      img = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)

    canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)

  # 5. Save output
  output_path = tile_dir / output_name
  canvas.save(output_path)
  print(f"✅ Saved stitched image to {output_path}")


def main():
  parser = argparse.ArgumentParser(
    description="Validate tile plan by stitching images."
  )
  parser.add_argument(
    "tile_dir", type=Path, help="Directory containing tile subdirectories"
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    return

  print(f"🔍 Validating plan in {args.tile_dir}...")

  # Stitch renders
  stitch_images(args.tile_dir, "render.png", "full_render.png")

  # Stitch whiteboxes
  stitch_images(args.tile_dir, "whitebox.png", "full_whitebox.png")


if __name__ == "__main__":
  main()
