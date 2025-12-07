"""
Generate pixel art for a tile using the Oxen.ai model.

This script generates pixel art for a tile at a specific (x, y) coordinate.
It handles:
1. Rendering quadrants if they don't exist
2. Stitching renders and neighbor generations for infill
3. Calling the Oxen API to generate pixel art
4. Saving generations to the database and filesystem

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  check_all_quadrants_generated,
  check_all_quadrants_rendered,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# Oxen API Model Constants
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# GCS and Oxen API Integration (from generate_tile_oxen.py)
# =============================================================================


def upload_to_gcs(
  local_path: Path, bucket_name: str, blob_name: str | None = None
) -> str:
  """
  Upload a file to Google Cloud Storage and return its public URL.

  Args:
    local_path: Path to the local file to upload
    bucket_name: Name of the GCS bucket
    blob_name: Name for the blob in GCS (defaults to unique name based on filename)

  Returns:
    Public URL of the uploaded file
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)

  if blob_name is None:
    unique_id = uuid.uuid4().hex[:8]
    blob_name = f"infills/{local_path.stem}_{unique_id}{local_path.suffix}"

  blob = bucket.blob(blob_name)

  print(f"   📤 Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
  blob.upload_from_filename(str(local_path))
  blob.make_public()

  return blob.public_url


def get_region_description(render_positions: list[tuple[int, int]]) -> str:
  """
  Get a human-readable description of the rendered region(s).

  Args:
    render_positions: List of (dx, dy) positions that need generation.
      (0, 0) = TL, (1, 0) = TR, (0, 1) = BL, (1, 1) = BR

  Returns:
    Description like "right half", "bottom left quadrant", etc.
  """
  # Convert (dx, dy) to quadrant indices: 0=TL, 1=TR, 2=BL, 3=BR
  rendered_indices = {dy * 2 + dx for dx, dy in render_positions}

  # Check for half patterns
  if rendered_indices == {0, 2}:
    return "left half"
  if rendered_indices == {1, 3}:
    return "right half"
  if rendered_indices == {0, 1}:
    return "top half"
  if rendered_indices == {2, 3}:
    return "bottom half"

  # Check for single quadrants
  if rendered_indices == {0}:
    return "top left quadrant"
  if rendered_indices == {1}:
    return "top right quadrant"
  if rendered_indices == {2}:
    return "bottom left quadrant"
  if rendered_indices == {3}:
    return "bottom right quadrant"

  # Three quadrants (L-shapes)
  if rendered_indices == {0, 1, 2}:
    return "top half and bottom left quadrant"
  if rendered_indices == {0, 1, 3}:
    return "top half and bottom right quadrant"
  if rendered_indices == {0, 2, 3}:
    return "left half and bottom right quadrant"
  if rendered_indices == {1, 2, 3}:
    return "right half and bottom left quadrant"

  # Multiple non-adjacent quadrants (diagonal)
  if rendered_indices == {0, 3}:
    return "top left and bottom right quadrants"
  if rendered_indices == {1, 2}:
    return "top right and bottom left quadrants"

  # Fallback for any other combination
  quadrant_names = {
    0: "top left",
    1: "top right",
    2: "bottom left",
    3: "bottom right",
  }
  names = [quadrant_names[i] for i in sorted(rendered_indices)]
  return " and ".join(names) + " quadrants"


def call_oxen_api(
  image_url: str,
  api_key: str,
  *,
  use_infill_model: bool = False,
  region_description: str | None = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
    image_url: Public URL of the input image
    api_key: Oxen API key
    use_infill_model: If True, use the infill model instead of generation model
    region_description: For infill model, describes which region to convert

  Returns:
    URL of the generated image
  """
  endpoint = "https://hub.oxen.ai/api/images/edit"

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  # We're experimenting with using *only* the inpainting model
  model = INPAINTING_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image."

  # if use_infill_model:
  #   model = "cannoneyed-satisfactory-harlequin-minnow"  # V04 infill model
  #   if region_description:
  #     prompt = (
  #       f"Convert the {region_description} of the image to isometric nyc pixel art "
  #       f"in precisely the style of the other part of the image."
  #     )
  #   else:
  #     prompt = "Convert the input image to <isometric nyc pixel art>"
  # else:
  #   model = "cannoneyed-odd-blue-marmot"  # V04 generation model
  #   prompt = "Convert the input image to <isometric nyc pixel art>"

  payload = {
    "model": model,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": 28,
  }

  print("   🤖 Calling Oxen API...")
  print(f"      Model: {payload['model']}")
  print(f"      Prompt: {payload['prompt']}")
  print(f"      Image: {image_url}")

  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()

  result = response.json()

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
  print("   📥 Downloading generated image...")

  response = requests.get(url, timeout=120)
  response.raise_for_status()

  with open(output_path, "wb") as f:
    f.write(response.content)


# =============================================================================
# Rendering Logic
# =============================================================================


def render_tile_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  renders_dir: Path,
  port: int,
) -> bool:
  """
  Render a tile and save its 4 quadrants to the database.

  Args:
    conn: Database connection
    config: Generation config
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)
    renders_dir: Directory to save renders
    port: Web server port

  Returns:
    True if successful
  """
  # Ensure the top-left quadrant exists
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile at ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

  # Build URL for rendering
  params = {
    "export": "true",
    "lat": quadrant["lat"],
    "lon": quadrant["lng"],
    "width": config["width_px"],
    "height": config["height_px"],
    "azimuth": config["camera_azimuth_degrees"],
    "elevation": config["camera_elevation_degrees"],
    "view_height": config.get("view_height_meters", 200),
  }
  query_string = urlencode(params)
  url = f"http://localhost:{port}/?{query_string}"

  # Render using Playwright
  output_path = renders_dir / f"render_{x}_{y}.png"

  with sync_playwright() as p:
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

    context = browser.new_context(
      viewport={"width": config["width_px"], "height": config["height_px"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    page.goto(url, wait_until="networkidle")

    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
    except Exception:
      print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

    page.screenshot(path=str(output_path))

    page.close()
    context.close()
    browser.close()

  print(f"      ✓ Saved full tile to {output_path.name}")

  # Split into quadrants and save to database
  tile_image = Image.open(output_path)
  quadrant_images = split_tile_into_quadrants(tile_image)

  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved quadrant ({qx}, {qy}) to DB")
    else:
      print(f"      ⚠️  Failed to save quadrant ({qx}, {qy})")

  return True


# =============================================================================
# Infill Image Creation
# =============================================================================


def get_quadrant_generation_local(
  conn: sqlite3.Connection, x: int, y: int
) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row and row[0] else None


def create_infill_image(
  conn: sqlite3.Connection, x: int, y: int
) -> tuple[Image.Image, list[tuple[int, int]]]:
  """
  Create an infill image for a tile at (x, y).

  The infill image is a 2x2 tile (same size as output) where:
  - Quadrants that already have generations use those generations
  - Quadrants that need generation use renders
  - A 1px red border is drawn around quadrants using renders

  Due to 50% tile overlap, some quadrants in the current tile may already
  have generations from previously generated tiles.

  Args:
    conn: Database connection
    x: Tile x coordinate (top-left quadrant x)
    y: Tile y coordinate (top-left quadrant y)

  Returns:
    Tuple of (infill_image, render_positions) where render_positions is a list
    of (dx, dy) positions that are using renders (need generation).
  """
  # Get dimensions from one of the renders
  sample_render = get_quadrant_render(conn, x, y)
  if sample_render is None:
    raise ValueError(f"Missing render for quadrant ({x}, {y})")
  sample_img = png_bytes_to_image(sample_render)
  quad_w, quad_h = sample_img.size

  # Create tile-sized infill image
  infill = Image.new("RGBA", (quad_w * 2, quad_h * 2))

  # Track which positions are using renders (need generation)
  render_positions = []

  # For each quadrant in the tile
  for dx in range(2):
    for dy in range(2):
      qx, qy = x + dx, y + dy
      pos_x = dx * quad_w
      pos_y = dy * quad_h

      # Check if this quadrant already has a generation
      gen_bytes = get_quadrant_generation_local(conn, qx, qy)

      if gen_bytes is not None:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        render_positions.append((dx, dy))
        print(f"      ({qx}, {qy}): using render (needs generation)")

  return infill, render_positions


def draw_red_border(
  image: Image.Image,
  render_positions: list[tuple[int, int]],
  quad_w: int,
  quad_h: int,
  border_width: int = 1,
) -> Image.Image:
  """
  Draw a red border around the quadrants that need generation.

  Args:
    image: The infill image to draw on
    render_positions: List of (dx, dy) positions that need generation
    quad_w: Width of each quadrant
    quad_h: Height of each quadrant
    border_width: Width of the border in pixels

  Returns:
    Image with red border drawn
  """
  if not render_positions:
    return image

  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  # Create a copy to draw on
  result = image.copy()

  # Find the bounding box of all render positions
  min_dx = min(p[0] for p in render_positions)
  max_dx = max(p[0] for p in render_positions)
  min_dy = min(p[1] for p in render_positions)
  max_dy = max(p[1] for p in render_positions)

  # Calculate pixel coordinates of the bounding box
  left = min_dx * quad_w
  right = (max_dx + 1) * quad_w
  top = min_dy * quad_h
  bottom = (max_dy + 1) * quad_h

  # Draw red border (1px lines)
  red = (255, 0, 0, 255)

  # Draw each border line
  for i in range(border_width):
    # Top edge
    for px in range(left, right):
      if 0 <= top + i < result.height:
        result.putpixel((px, top + i), red)
    # Bottom edge
    for px in range(left, right):
      if 0 <= bottom - 1 - i < result.height:
        result.putpixel((px, bottom - 1 - i), red)
    # Left edge
    for py in range(top, bottom):
      if 0 <= left + i < result.width:
        result.putpixel((left + i, py), red)
    # Right edge
    for py in range(top, bottom):
      if 0 <= right - 1 - i < result.width:
        result.putpixel((right - 1 - i, py), red)

  return result


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile at position (x, y).

  Steps:
  1. Check if renders exist for all 4 quadrants; render if needed
  2. Check if generations already exist; skip if not overwriting
  3. Create infill image (generated quadrants + renders with red border)
  4. Upload to GCS and call Oxen API (unless debug mode)
  5. Save generation to database and filesystem

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate
    y: Tile y coordinate
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(
        "OXEN_INFILL_V02_API_KEY environment variable not set. "
        "Please add it to your .env file."
      )
  else:
    api_key = None

  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create directories
    renders_dir = generation_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    infills_dir = generation_dir / "infills"
    infills_dir.mkdir(exist_ok=True)
    generations_dir = generation_dir / "generations"
    generations_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
    print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, x, y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, x, y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if not debug and not overwrite and check_all_quadrants_generated(conn, x, y):
      print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
      return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(conn, x, y)

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, x, y)
    sample_img = png_bytes_to_image(sample_render)
    quad_w, quad_h = sample_img.size

    # Draw red border around the render (non-generated) portion
    if render_positions:
      print(
        f"\n   Drawing red border around {len(render_positions)} render quadrant(s)"
      )
      infill_image = draw_red_border(
        infill_image, render_positions, quad_w, quad_h, border_width=1
      )

    # Save infill image to disk
    infill_path = infills_dir / f"infill_{x}_{y}.png"
    infill_image.save(infill_path)
    print(f"   ✓ Saved infill to {infill_path.name}")

    # If debug mode, stop here
    if debug:
      print(f"\n{'=' * 60}")
      print("🔍 DEBUG MODE - Skipping model inference")
      print(f"   Infill saved to: {infill_path}")
      print(f"{'=' * 60}")
      return infill_path

    # Step 4: Upload and generate
    print("\n📋 Step 3: Generating pixel art...")
    image_url = upload_to_gcs(infill_path, bucket_name)

    # We use the generation model if *all* quadrants must be generated (i.e. len(render_positions) == 4)
    use_infill_model = len(render_positions) != 4

    # Get region description for infill model prompt
    region_description = None
    if use_infill_model:
      region_description = get_region_description(render_positions)
      print(f"   Region to generate: {region_description}")

    generated_url = call_oxen_api(
      image_url=image_url,
      api_key=api_key,
      use_infill_model=use_infill_model,
      region_description=region_description,
    )

    # Download result
    generation_path = generations_dir / f"{x}_{y}.png"
    download_image(generated_url, generation_path)
    print(f"   ✓ Downloaded generation to {generation_path.name}")

    # Step 5: Extract quadrants and save to database
    print("\n📋 Step 4: Saving to database...")
    generation_image = Image.open(generation_path)

    # Split into quadrants and save (only the ones that needed generation)
    quadrant_images = split_tile_into_quadrants(generation_image)

    for (dx, dy), quad_img in quadrant_images.items():
      # Only save quadrants that were using renders (needed generation)
      if (dx, dy) in render_positions:
        qx, qy = x + dx, y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = x + dx, y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - already had generation")

    print(f"\n{'=' * 60}")
    print("✅ Generation complete!")
    print(f"   Infill: {infill_path}")
    print(f"   Generation: {generation_path}")
    print(f"{'=' * 60}")

    return generation_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Tile x coordinate",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Tile y coordinate",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate even if generations already exist",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode: skip model inference and just save the infill image",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    result = generate_tile(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
      args.debug,
    )
    return 0 if result else 1

  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
  finally:
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())
