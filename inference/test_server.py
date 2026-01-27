"""
Test script for the Modal inference server.

Builds a template image from quadrants in the database and calls the Modal endpoint.

Usage:
  uv run python inference/test_server.py \
    --generation-dir generations/tiny-nyc \
    --quadrants "(0,0),(1,0)" \
    --source-layer generations \
    --endpoint https://your-workspace--qwen-image-edit-server-imageeditor-edit-b64.modal.run
"""

import argparse
import base64
import json
import sqlite3
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

# Constants
QUADRANT_SIZE = 512
TEMPLATE_SIZE = 1024


def load_generation_config(generation_dir: Path) -> dict:
  """Load the generation configuration."""
  config_path = generation_dir / "generation_config.json"
  if not config_path.exists():
    raise FileNotFoundError(f"generation_config.json not found in {generation_dir}")
  with open(config_path) as f:
    return json.load(f)


def parse_quadrants(quadrant_str: str) -> list[tuple[int, int]]:
  """Parse quadrant string like '(0,0),(1,0)' into list of tuples."""
  import re

  pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
  matches = re.findall(pattern, quadrant_str)
  if not matches:
    raise ValueError(f"No valid quadrant tuples found in: {quadrant_str}")
  return [(int(x), int(y)) for x, y in matches]


def get_quadrant_image(
  conn: sqlite3.Connection,
  x: int,
  y: int,
  source_layer: str,
) -> Image.Image | None:
  """Get a quadrant image from the specified source layer."""
  cursor = conn.cursor()

  if source_layer == "generations":
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
  elif source_layer == "renders":
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
  else:
    raise ValueError(f"Unknown source layer: {source_layer}")

  row = cursor.fetchone()
  if row is None or row[0] is None:
    return None

  return Image.open(BytesIO(row[0]))


def build_template(
  conn: sqlite3.Connection,
  quadrants: list[tuple[int, int]],
  source_layer: str,
  border_width: int = 2,
) -> Image.Image:
  """
  Build a template image from the specified quadrants.

  Builds a 1024x1024 template with all quadrants from the source layer,
  and a red border around the infill region.
  """
  # Find bounding box of quadrants
  min_qx = min(q[0] for q in quadrants)
  max_qx = max(q[0] for q in quadrants)
  min_qy = min(q[1] for q in quadrants)
  max_qy = max(q[1] for q in quadrants)

  infill_width = (max_qx - min_qx + 1) * QUADRANT_SIZE
  infill_height = (max_qy - min_qy + 1) * QUADRANT_SIZE

  # Calculate template placement (center the infill region)
  margin_x = TEMPLATE_SIZE - infill_width
  margin_y = TEMPLATE_SIZE - infill_height
  infill_x = margin_x // 2
  infill_y = margin_y // 2

  # World coordinates of template top-left
  world_offset_x = min_qx * QUADRANT_SIZE - infill_x
  world_offset_y = min_qy * QUADRANT_SIZE - infill_y

  # Create template image
  template = Image.new("RGB", (TEMPLATE_SIZE, TEMPLATE_SIZE), (255, 255, 255))

  # Determine which quadrants are covered by the template
  start_qx = world_offset_x // QUADRANT_SIZE
  end_qx = (world_offset_x + TEMPLATE_SIZE - 1) // QUADRANT_SIZE
  start_qy = world_offset_y // QUADRANT_SIZE
  end_qy = (world_offset_y + TEMPLATE_SIZE - 1) // QUADRANT_SIZE

  # Fill in all quadrants from source layer
  for qx in range(start_qx, end_qx + 1):
    for qy in range(start_qy, end_qy + 1):
      quad_img = get_quadrant_image(conn, qx, qy, source_layer)

      if quad_img is not None:
        # Calculate position in template
        paste_x = qx * QUADRANT_SIZE - world_offset_x
        paste_y = qy * QUADRANT_SIZE - world_offset_y
        template.paste(quad_img, (paste_x, paste_y))

  # Draw red border around infill region
  from PIL import ImageDraw

  draw = ImageDraw.Draw(template)
  for i in range(border_width):
    draw.rectangle(
      [
        infill_x + i,
        infill_y + i,
        infill_x + infill_width - 1 - i,
        infill_y + infill_height - 1 - i,
      ],
      outline=(255, 0, 0),
    )

  return template


def call_endpoint(
  endpoint: str,
  image: Image.Image,
  prompt: str,
  steps: int = 25,
  guidance_scale: float = 3.0,
  true_cfg_scale: float = 2.0,
) -> Image.Image:
  """Call the Modal inference endpoint."""
  # Encode image as base64
  buffer = BytesIO()
  image.save(buffer, format="PNG")
  image_b64 = base64.b64encode(buffer.getvalue()).decode()

  print(f"Calling endpoint: {endpoint}")
  print(f"Image size: {len(image_b64):,} bytes (base64)")
  print(f"Params: steps={steps}, guidance_scale={guidance_scale}, true_cfg_scale={true_cfg_scale}")

  response = httpx.post(
    endpoint,
    json={
      "image_b64": image_b64,
      "prompt": prompt,
      "steps": steps,
      "guidance_scale": guidance_scale,
      "true_cfg_scale": true_cfg_scale,
    },
    timeout=600,  # 10 min to account for cold starts
    follow_redirects=True,
  )
  response.raise_for_status()

  result = response.json()
  result_b64 = result["image_b64"]
  print(f"Received: {len(result_b64):,} bytes (base64)")

  return Image.open(BytesIO(base64.b64decode(result_b64)))


def main():
  parser = argparse.ArgumentParser(
    description="Test the Modal inference server",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "--generation-dir",
    type=Path,
    required=True,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--quadrants",
    type=str,
    required=True,
    help='Quadrant coordinates, e.g., "(0,0),(1,0)"',
  )
  parser.add_argument(
    "--source-layer",
    type=str,
    choices=["renders", "generations"],
    default="generations",
    help="Source layer for context (default: generations)",
  )
  parser.add_argument(
    "--endpoint",
    type=str,
    required=True,
    help="Modal endpoint URL for the edit_b64 endpoint",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=25,
    help="Number of inference steps (default: 25)",
  )
  parser.add_argument(
    "--prompt",
    type=str,
    default="Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present).",
    help="Prompt for the model",
  )
  parser.add_argument(
    "--guidance-scale",
    type=float,
    default=3.0,
    help="Guidance scale (default: 3.0, try 5-7 for stronger effect)",
  )
  parser.add_argument(
    "--true-cfg-scale",
    type=float,
    default=2.0,
    help="True CFG scale (default: 2.0, try 3-5 for stronger effect)",
  )

  args = parser.parse_args()

  # Validate paths
  db_path = args.generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Parse quadrants
  quadrants = parse_quadrants(args.quadrants)
  print(f"Quadrants: {quadrants}")
  print(f"Source layer: {args.source_layer}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  # Build template
  print("Building template...")
  template = build_template(conn, quadrants, args.source_layer)

  # Save input
  output_dir = Path(__file__).parent
  input_path = output_dir / "input.png"
  template.save(input_path)
  print(f"Saved input: {input_path}")

  # Call endpoint
  print("Calling inference endpoint...")
  print(f"Prompt: {args.prompt}")
  result = call_endpoint(
    args.endpoint,
    template,
    prompt=args.prompt,
    steps=args.steps,
    guidance_scale=args.guidance_scale,
    true_cfg_scale=args.true_cfg_scale,
  )

  # Save output
  output_path = output_dir / "output.png"
  result.save(output_path)
  print(f"Saved output: {output_path}")

  conn.close()
  print("Done!")


if __name__ == "__main__":
  main()
