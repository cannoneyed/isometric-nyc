"""
Generate tiles using the Oxen.ai fine-tuned model with infill strategy.

This script creates an infill template by compositing neighboring tile generations
with the current tile's render, draws red outlines around rendered regions,
uploads to GCS, and uses the Oxen API to generate the pixel art infill.

The template follows the quadrant-based infill strategy:
- Quadrants covered by neighbor generations are marked as "generated"
- Quadrants not covered by neighbors use the render and are outlined in red
- The prompt describes which region(s) to convert

Usage:
  uv run python src/isometric_nyc/generate_tile_oxen_infill.py <tile_dir>
"""

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image, ImageDraw

# Red outline settings
OUTLINE_COLOR = (255, 0, 0)
OUTLINE_WIDTH = 1


def draw_outline(
  img: Image.Image,
  box: Tuple[int, int, int, int],
  color: Tuple[int, int, int] = OUTLINE_COLOR,
  width: int = OUTLINE_WIDTH,
) -> None:
  """
  Draw an outline around a rectangular region.

  The outline is drawn inside the box. When the box extends to the
  image edge, the outline is inset slightly to ensure visibility.
  """
  draw = ImageDraw.Draw(img)
  img_width, img_height = img.size
  x1, y1, x2, y2 = box

  # Inset outline from image edges to ensure visibility
  edge_inset = 1
  draw_x1 = edge_inset if x1 <= 0 else x1
  draw_y1 = edge_inset if y1 <= 0 else y1
  draw_x2 = (img_width - 1 - edge_inset) if x2 >= img_width else (x2 - 1)
  draw_y2 = (img_height - 1 - edge_inset) if y2 >= img_height else (y2 - 1)

  for i in range(width):
    draw.rectangle(
      [draw_x1 + i, draw_y1 + i, draw_x2 - i, draw_y2 - i],
      outline=color,
      fill=None,
    )


def merge_adjacent_rendered_boxes(
  rendered_indices: List[int],
  quadrant_boxes: List[Tuple[int, int, int, int]],
  width: int,
  height: int,
) -> List[Tuple[int, int, int, int]]:
  """
  Merge adjacent rendered quadrants into larger bounding boxes.
  """
  half_w = width // 2
  half_h = height // 2

  rendered_set = set(rendered_indices)
  merged_boxes: List[Tuple[int, int, int, int]] = []
  used: set[int] = set()

  # Check horizontal adjacencies (same row)
  if 0 in rendered_set and 1 in rendered_set:
    merged_boxes.append((0, 0, width, half_h))
    used.update([0, 1])

  if 2 in rendered_set and 3 in rendered_set:
    merged_boxes.append((0, half_h, width, height))
    used.update([2, 3])

  # Check vertical adjacencies (same column)
  if 0 in rendered_set and 2 in rendered_set and 0 not in used and 2 not in used:
    merged_boxes.append((0, 0, half_w, height))
    used.update([0, 2])

  if 1 in rendered_set and 3 in rendered_set and 1 not in used and 3 not in used:
    merged_boxes.append((half_w, 0, width, height))
    used.update([1, 3])

  # Add remaining single quadrants
  for idx in rendered_indices:
    if idx not in used:
      merged_boxes.append(quadrant_boxes[idx])

  return merged_boxes


def get_region_description(rendered_indices: List[int]) -> str:
  """
  Get a human-readable description of the rendered region(s).

  Args:
      rendered_indices: List of quadrant indices (0=TL, 1=TR, 2=BL, 3=BR).

  Returns:
      Description like "right half", "bottom left quadrant", etc.
  """
  rendered_set = set(rendered_indices)

  # Check for half patterns
  if rendered_set == {0, 2}:
    return "left half"
  if rendered_set == {1, 3}:
    return "right half"
  if rendered_set == {0, 1}:
    return "top half"
  if rendered_set == {2, 3}:
    return "bottom half"

  # Check for single quadrants
  if rendered_set == {0}:
    return "top left quadrant"
  if rendered_set == {1}:
    return "top right quadrant"
  if rendered_set == {2}:
    return "bottom left quadrant"
  if rendered_set == {3}:
    return "bottom right quadrant"

  # Multiple non-adjacent quadrants (rare case)
  quadrant_names = {
    0: "top left",
    1: "top right",
    2: "bottom left",
    3: "bottom right",
  }
  names = [quadrant_names[i] for i in sorted(rendered_indices)]
  return " and ".join(names) + " quadrants"


def create_infill_template(tile_dir_path: Path) -> Tuple[Path | None, str | None]:
  """
  Create a template.png for the given tile directory based on neighbors.

  Returns:
      Tuple of (path to template.png, prompt string) or (None, None) if failed.
  """
  if not tile_dir_path.exists():
    raise FileNotFoundError(f"Tile directory not found: {tile_dir_path}")

  # Load view.json
  view_json_path = tile_dir_path / "view.json"
  if not view_json_path.exists():
    raise FileNotFoundError(f"view.json not found in {tile_dir_path}")

  with open(view_json_path, "r") as f:
    view_json = json.load(f)

  row = view_json.get("row")
  col = view_json.get("col")
  width_px = view_json.get("width_px", 1024)
  height_px = view_json.get("height_px", 1024)

  if row is None or col is None:
    print("Error: view.json missing 'row' or 'col' fields.")
    return None, None

  plan_dir = tile_dir_path.parent
  half_w = width_px // 2
  half_h = height_px // 2

  # Load the render image for this tile
  render_path = tile_dir_path / "render.png"
  if not render_path.exists():
    print(f"Error: render.png not found in {tile_dir_path}")
    return None, None

  render_img = Image.open(render_path).convert("RGB")
  if render_img.size != (width_px, height_px):
    render_img = render_img.resize((width_px, height_px), Image.Resampling.LANCZOS)

  # Create canvas starting with render
  canvas = render_img.copy()

  # Track which quadrants are covered by neighbor generations
  # Quadrant indices: 0=TL, 1=TR, 2=BL, 3=BR
  quadrant_has_generation: Dict[int, bool] = {0: False, 1: False, 2: False, 3: False}

  # Define quadrant boxes
  quadrant_boxes = [
    (0, 0, half_w, half_h),  # TL (0)
    (half_w, 0, width_px, half_h),  # TR (1)
    (0, half_h, half_w, height_px),  # BL (2)
    (half_w, half_h, width_px, height_px),  # BR (3)
  ]

  # Check neighbors and paste their overlapping regions
  # Neighbor format: (row_offset, col_offset, quadrants_covered, crop_region)
  # crop_region is (x1, y1, x2, y2) of the neighbor's generation to use
  neighbors = [
    # Left neighbor: its right half covers our left half (TL, BL)
    (-1, 0, [0, 2], (half_w, 0, width_px, height_px), (0, 0)),
    # Right neighbor: its left half covers our right half (TR, BR)
    (1, 0, [1, 3], (0, 0, half_w, height_px), (half_w, 0)),
    # Top neighbor: its bottom half covers our top half (TL, TR)
    (0, -1, [0, 1], (0, half_h, width_px, height_px), (0, 0)),
    # Bottom neighbor: its top half covers our bottom half (BL, BR)
    (0, 1, [2, 3], (0, 0, width_px, half_h), (0, half_h)),
  ]

  for r_off, c_off, quadrants, crop_box, paste_pos in neighbors:
    n_row = row + r_off
    n_col = col + c_off

    # Try both naming conventions
    n_dir_names = [f"{n_row:03d}_{n_col:03d}", f"{n_row:03d}"]
    n_gen_path = None

    for n_dir_name in n_dir_names:
      candidate = plan_dir / n_dir_name / "generation.png"
      if candidate.exists():
        n_gen_path = candidate
        break

    if n_gen_path and n_gen_path.exists():
      print(f"Found neighbor at row={n_row}, col={n_col}")
      try:
        n_img = Image.open(n_gen_path).convert("RGB")
        if n_img.size != (width_px, height_px):
          n_img = n_img.resize((width_px, height_px), Image.Resampling.LANCZOS)

        # Crop the relevant region from neighbor
        region = n_img.crop(crop_box)
        canvas.paste(region, paste_pos)

        # Mark quadrants as having generation
        for q in quadrants:
          quadrant_has_generation[q] = True

      except Exception as e:
        print(f"Error processing neighbor: {e}")

  # Determine which quadrants are rendered (not covered by neighbors)
  rendered_indices = [
    q for q, has_gen in quadrant_has_generation.items() if not has_gen
  ]

  if not rendered_indices:
    print("All quadrants covered by neighbors - no generation needed.")
    return None, None

  print(f"Rendered quadrants: {rendered_indices}")

  # Merge adjacent rendered regions and draw outlines
  outline_boxes = merge_adjacent_rendered_boxes(
    rendered_indices, quadrant_boxes, width_px, height_px
  )

  for box in outline_boxes:
    draw_outline(canvas, box)

  # Save template
  output_path = tile_dir_path / "template.png"
  canvas.save(output_path)
  print(f"Created template at {output_path}")

  # Generate prompt
  region_desc = get_region_description(rendered_indices)
  prompt = (
    f"Convert the {region_desc} of the image to isometric nyc pixel art "
    f"in precisely the style of the other part of the image."
  )

  return output_path, prompt


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """Upload a file to Google Cloud Storage and return its public URL."""
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"templates/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"Uploading {local_path} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  public_url = blob.public_url
  print(f"File uploaded. Public URL: {public_url}")

  return public_url


def call_oxen_api(image_url: str, prompt: str, api_key: str) -> str:
  """Call the Oxen API to generate an infill image."""
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": "cannoneyed-modern-salmon-unicorn",
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": 28,
  }

  print(f"Calling Oxen API with image: {image_url}")
  print(f"Prompt: {prompt}")

  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()
  print(f"Oxen API response: {result}")

  if "images" in result and len(result["images"]) > 0:
    return result["images"][0]["url"]
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  else:
    raise ValueError(f"Unexpected API response format: {result}")


def download_image(url: str, output_path: Path) -> None:
  """Download an image from a URL and save it."""
  print(f"Downloading generated image from {url}...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)

  print(f"Image saved to {output_path}")


def generate_tile(tile_dir: str, bucket_name: str) -> None:
  """Generate a tile using the Oxen API with infill strategy."""
  load_dotenv()

  api_key = os.getenv("OXEN_INFILL_V04_API_KEY")
  if not api_key:
    raise ValueError(
      "OXEN_INFILL_V04_API_KEY environment variable not set. "
      "Please add it to your .env file."
    )

  tile_dir_path = Path(tile_dir)

  # Step 1: Create the template image
  print("\n" + "=" * 60)
  print("STEP 1: Creating infill template")
  print("=" * 60)
  template_path, prompt = create_infill_template(tile_dir_path)

  if template_path is None or prompt is None:
    print("Cannot proceed without template. Exiting.")
    return

  # Step 2: Upload to Google Cloud Storage
  print("\n" + "=" * 60)
  print("STEP 2: Uploading to Google Cloud Storage")
  print("=" * 60)
  image_url = upload_to_gcs(template_path, bucket_name)

  # Step 3: Call Oxen API
  print("\n" + "=" * 60)
  print("STEP 3: Calling Oxen API")
  print("=" * 60)
  generated_url = call_oxen_api(image_url, prompt, api_key)

  # Step 4: Download the result
  print("\n" + "=" * 60)
  print("STEP 4: Downloading generated image")
  print("=" * 60)
  output_path = tile_dir_path / "generation.png"
  download_image(generated_url, output_path)

  print("\n" + "=" * 60)
  print("GENERATION COMPLETE!")
  print(f"Output saved to: {output_path}")
  print("=" * 60)


def main():
  parser = argparse.ArgumentParser(
    description="Generate isometric pixel art tiles using Oxen API with infill strategy."
  )
  parser.add_argument(
    "tile_dir",
    help="Directory containing the tile assets (view.json, render.png)",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="Google Cloud Storage bucket name for uploading images",
  )
  parser.add_argument(
    "--template-only",
    action="store_true",
    help="Only create the template, don't call the API",
  )

  args = parser.parse_args()

  if args.template_only:
    tile_dir_path = Path(args.tile_dir)
    template_path, prompt = create_infill_template(tile_dir_path)
    if template_path and prompt:
      print(f"\nTemplate created: {template_path}")
      print(f"Prompt: {prompt}")
  else:
    generate_tile(args.tile_dir, args.bucket)


if __name__ == "__main__":
  main()
