# Refactor omni tile generation script
_Exported on 12/10/2025 at 09:09:25 EST from Cursor (2.1.39)_

---

**User**

Please refactor @src/isometric_nyc/e2e_generation/generate_tile_omni.py to use the logic in @src/isometric_nyc/e2e_generation/generate_tile.py

---

**Cursor**

```python
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
  upload_to_gcs,
)
```

Looking at both files, I can see they share a lot of duplicated code. The main differences are:
1. Different API key environment variable names (`OXEN_OMNI_v04_API_KEY` vs `OXEN_INFILL_V02_API_KEY`)
2. `generate_tile.py` has extra parameters in `call_oxen_api` that are now unused (commented out code)

Let me first check how these are used in the app to understand the best refactoring approach:

Let me also check `generate_omni.py` since that's what the app imports from:

Now I have a clear picture. The key observations:
1. `generate_tile.py` and `generate_tile_omni.py` share ~95% identical code
2. Main difference is the API key environment variable name
3. `generate_tile.py` has a duplicate `upload_to_gcs` function that's already in `shared.py`
4. Both use the same `OMNI_MODEL_ID` model and same prompt

Let me refactor `generate_tile_omni.py` to reuse the logic from `generate_tile.py`. First, I'll also fix `generate_tile.py` to use the shared `upload_to_gcs`:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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

```

Now let me remove the duplicate `upload_to_gcs` function from `generate_tile.py`:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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

```

Now let me update the `generate_tile` function in `generate_tile.py` to accept an optional `api_key` parameter, making it reusable:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
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
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

```

Let me check and fix the code after the API key section:

```python
  if not debug:
    api_key = os.getenv("OXEN_INFILL_V02_API_KEY")
    if not api_key:
      raise ValueError(...)
  else:
    api_key = None
```

```python
  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(...)
```

Let me also check if there was an `else` clause that needs to be removed:

Now let me completely refactor `generate_tile_omni.py` to use the logic from `generate_tile.py`:

```python
"""
Generate pixel art for a tile using the Oxen.ai omni model.

This is a thin wrapper around generate_tile.py that uses the omni model
configuration (OXEN_OMNI_v04_API_KEY environment variable).

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
from pathlib import Path

from isometric_nyc.e2e_generation.generate_tile import generate_tile
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  start_web_server,
)

# Default API key environment variable for the omni model
OMNI_API_KEY_ENV_VAR = "OXEN_OMNI_v04_API_KEY"


def generate_tile_omni(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile using the omni model.

  This is a convenience wrapper around generate_tile() that uses
  OXEN_OMNI_v04_API_KEY for authentication.

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
  return generate_tile(
    generation_dir=generation_dir,
    x=x,
    y=y,
    port=port,
    bucket_name=bucket_name,
    overwrite=overwrite,
    debug=debug,
    api_key_env_var=OMNI_API_KEY_ENV_VAR,
  )


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API (omni model)."
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

    result = generate_tile_omni(
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

```

Let me check for any linter errors in both files:

Let me verify both files are correct by viewing the final versions:

Let me run a quick test to make sure the modules import correctly:

The refactoring is complete. Here's a summary of the changes:

## Changes to `generate_tile.py`:
1. **Removed duplicate `upload_to_gcs`** - Now imports from `shared.py` instead of defining its own copy
2. **Removed unused imports** - `uuid` and `google.cloud.storage` are no longer needed
3. **Added configurable API key parameters** to `generate_tile()`:
   - `api_key: str | None = None` - Allows passing the key directly
   - `api_key_env_var: str = "OXEN_INFILL_V02_API_KEY"` - Configurable env var name

## Refactored `generate_tile_omni.py`:
The file went from **613 lines** to **~150 lines** by:
1. Importing and reusing `generate_tile` from `generate_tile.py`
2. Providing a thin wrapper `generate_tile_omni()` that calls `generate_tile()` with `api_key_env_var="OXEN_OMNI_v04_API_KEY"`
3. Keeping the same CLI interface for backwards compatibility

The new `generate_tile_omni.py` is now just a thin configuration wrapper:

```17:63:src/isometric_nyc/e2e_generation/generate_tile_omni.py
from isometric_nyc.e2e_generation.generate_tile import generate_tile
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  start_web_server,
)

# Default API key environment variable for the omni model
OMNI_API_KEY_ENV_VAR = "OXEN_OMNI_v04_API_KEY"


def generate_tile_omni(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile using the omni model.
  ...
  """
  return generate_tile(
    generation_dir=generation_dir,
    x=x,
    y=y,
    port=port,
    bucket_name=bucket_name,
    overwrite=overwrite,
    debug=debug,
    api_key_env_var=OMNI_API_KEY_ENV_VAR,
  )
```

```python
from isometric_nyc.e2e_generation.generate_tile import generate_tile
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  start_web_server,
)

# Default API key environment variable for the omni model
OMNI_API_KEY_ENV_VAR = "OXEN_OMNI_v04_API_KEY"


def generate_tile_omni(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
) -> Path | None:
  """
  Generate pixel art for a tile using the omni model.
  ...
  """
  return generate_tile(
    generation_dir=generation_dir,
    x=x,
    y=y,
    port=port,
    bucket_name=bucket_name,
    overwrite=overwrite,
    debug=debug,
    api_key_env_var=OMNI_API_KEY_ENV_VAR,
  )
```

---

**User**

OK great - now lets add a new parameter to the generation script that allows you to specify which quadrant to generate in a 2x2 tile - for example, I'd like to be able to say something like



x=0

y=0



and context = [(-1,-1),(-1,0),(0,-1),(0,0)]



That way I can define the four qudrants of the tile I want to use for surrounding context, with the generated target in the bottom right (instead of the default top left)

---

**Cursor**

I understand - you want to specify which quadrant position within the 2x2 tile should be the target for generation, so you can use surrounding context quadrants. Let me implement this.

Now I understand the structure. Let me implement the `target_position` parameter. I'll add:

1. A helper to parse target position strings like "tl", "tr", "bl", "br"
2. Modify `create_infill_image` to accept optional target offsets (only mark those as needing generation)
3. Modify `generate_tile` to calculate tile anchor based on target position
4. Update CLI

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
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
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

```

Now let me modify `create_infill_image` to accept an optional `target_offsets` parameter:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


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
  conn: sqlite3.Connection,
  x: int,
  y: int,
  target_offsets: list[tuple[int, int]] | None = None,
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
    target_offsets: Optional list of (dx, dy) offsets specifying which quadrants
      to generate. If None, generates all quadrants that don't have generations.
      If specified, only these quadrants will be marked for generation (even if
      they already have generations when overwrite is True).

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

      # Determine if this quadrant should be generated
      if target_offsets is not None:
        # User specified which quadrants to generate
        should_generate = (dx, dy) in target_offsets
      else:
        # Default behavior: generate if no existing generation
        should_generate = gen_bytes is None

      if gen_bytes is not None and not should_generate:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render (either needs generation or explicitly targeted)
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        if should_generate:
          render_positions.append((dx, dy))
          status = "(TARGET)" if target_offsets else "(needs generation)"
          print(f"      ({qx}, {qy}): using render {status}")
        else:
          print(f"      ({qx}, {qy}): using render (context only)")

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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
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
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

```

Now let me update the `generate_tile` function to accept the `target_position` parameter:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


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
  conn: sqlite3.Connection,
  x: int,
  y: int,
  target_offsets: list[tuple[int, int]] | None = None,
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
    target_offsets: Optional list of (dx, dy) offsets specifying which quadrants
      to generate. If None, generates all quadrants that don't have generations.
      If specified, only these quadrants will be marked for generation (even if
      they already have generations when overwrite is True).

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

      # Determine if this quadrant should be generated
      if target_offsets is not None:
        # User specified which quadrants to generate
        should_generate = (dx, dy) in target_offsets
      else:
        # Default behavior: generate if no existing generation
        should_generate = gen_bytes is None

      if gen_bytes is not None and not should_generate:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render (either needs generation or explicitly targeted)
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        if should_generate:
          render_positions.append((dx, dy))
          status = "(TARGET)" if target_offsets else "(needs generation)"
          print(f"      ({qx}, {qy}): using render {status}")
        else:
          print(f"      ({qx}, {qy}): using render (context only)")

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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
  target_position: str | None = None,
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
    x: Tile x coordinate (or target quadrant x if target_position is specified)
    y: Tile y coordinate (or target quadrant y if target_position is specified)
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), x,y is the tile anchor and all non-generated quadrants are generated.
      If specified, x,y is the target quadrant and the tile anchor is calculated based on
      where the target should be positioned within the tile.

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

```

Now I need to update the function body to use `target_position`:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


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
  conn: sqlite3.Connection,
  x: int,
  y: int,
  target_offsets: list[tuple[int, int]] | None = None,
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
    target_offsets: Optional list of (dx, dy) offsets specifying which quadrants
      to generate. If None, generates all quadrants that don't have generations.
      If specified, only these quadrants will be marked for generation (even if
      they already have generations when overwrite is True).

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

      # Determine if this quadrant should be generated
      if target_offsets is not None:
        # User specified which quadrants to generate
        should_generate = (dx, dy) in target_offsets
      else:
        # Default behavior: generate if no existing generation
        should_generate = gen_bytes is None

      if gen_bytes is not None and not should_generate:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render (either needs generation or explicitly targeted)
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        if should_generate:
          render_positions.append((dx, dy))
          status = "(TARGET)" if target_offsets else "(needs generation)"
          print(f"      ({qx}, {qy}): using render {status}")
        else:
          print(f"      ({qx}, {qy}): using render (context only)")

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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
  target_position: str | None = None,
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
    x: Tile x coordinate (or target quadrant x if target_position is specified)
    y: Tile y coordinate (or target quadrant y if target_position is specified)
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), x,y is the tile anchor and all non-generated quadrants are generated.
      If specified, x,y is the target quadrant and the tile anchor is calculated based on
      where the target should be positioned within the tile.

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

    # Calculate tile anchor and target offsets based on target_position
    if target_position is not None:
      # x, y is the target quadrant; calculate tile anchor
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      target_offset = parse_target_position(target_position)
      target_offsets = [target_offset]
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print(f"   Context quadrants: ", end="")
      for dx in range(2):
        for dy in range(2):
          qx, qy = tile_x + dx, tile_y + dy
          is_target = (dx, dy) == target_offset
          marker = " [TARGET]" if is_target else ""
          print(f"({qx},{qy}){marker}", end=" ")
      print()
      if debug:
        print("   [DEBUG MODE]")
      print(f"{'=' * 60}")
    else:
      # x, y is the tile anchor (default behavior)
      tile_x, tile_y = x, y
      target_offsets = None
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
      print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, tile_x, tile_y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, tile_x, tile_y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if target_position is not None:
      # Only check if the specific target quadrant has a generation
      if not debug and not overwrite:
        gen_bytes = get_quadrant_generation_local(conn, x, y)
        if gen_bytes is not None:
          print(f"\n⏭️  Skipping - quadrant ({x}, {y}) already generated (use --overwrite)")
          return None
    else:
      # Check all quadrants in the tile
      if not debug and not overwrite and check_all_quadrants_generated(conn, tile_x, tile_y):
        print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
        return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(
      conn, tile_x, tile_y, target_offsets
    )

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

```

Now let me update the rest of the function to use `tile_x, tile_y` where appropriate:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


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
  conn: sqlite3.Connection,
  x: int,
  y: int,
  target_offsets: list[tuple[int, int]] | None = None,
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
    target_offsets: Optional list of (dx, dy) offsets specifying which quadrants
      to generate. If None, generates all quadrants that don't have generations.
      If specified, only these quadrants will be marked for generation (even if
      they already have generations when overwrite is True).

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

      # Determine if this quadrant should be generated
      if target_offsets is not None:
        # User specified which quadrants to generate
        should_generate = (dx, dy) in target_offsets
      else:
        # Default behavior: generate if no existing generation
        should_generate = gen_bytes is None

      if gen_bytes is not None and not should_generate:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render (either needs generation or explicitly targeted)
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        if should_generate:
          render_positions.append((dx, dy))
          status = "(TARGET)" if target_offsets else "(needs generation)"
          print(f"      ({qx}, {qy}): using render {status}")
        else:
          print(f"      ({qx}, {qy}): using render (context only)")

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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
  target_position: str | None = None,
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
    x: Tile x coordinate (or target quadrant x if target_position is specified)
    y: Tile y coordinate (or target quadrant y if target_position is specified)
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), x,y is the tile anchor and all non-generated quadrants are generated.
      If specified, x,y is the target quadrant and the tile anchor is calculated based on
      where the target should be positioned within the tile.

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

    # Calculate tile anchor and target offsets based on target_position
    if target_position is not None:
      # x, y is the target quadrant; calculate tile anchor
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      target_offset = parse_target_position(target_position)
      target_offsets = [target_offset]
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print(f"   Context quadrants: ", end="")
      for dx in range(2):
        for dy in range(2):
          qx, qy = tile_x + dx, tile_y + dy
          is_target = (dx, dy) == target_offset
          marker = " [TARGET]" if is_target else ""
          print(f"({qx},{qy}){marker}", end=" ")
      print()
      if debug:
        print("   [DEBUG MODE]")
      print(f"{'=' * 60}")
    else:
      # x, y is the tile anchor (default behavior)
      tile_x, tile_y = x, y
      target_offsets = None
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
      print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, tile_x, tile_y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, tile_x, tile_y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if target_position is not None:
      # Only check if the specific target quadrant has a generation
      if not debug and not overwrite:
        gen_bytes = get_quadrant_generation_local(conn, x, y)
        if gen_bytes is not None:
          print(f"\n⏭️  Skipping - quadrant ({x}, {y}) already generated (use --overwrite)")
          return None
    else:
      # Check all quadrants in the tile
      if not debug and not overwrite and check_all_quadrants_generated(conn, tile_x, tile_y):
        print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
        return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(
      conn, tile_x, tile_y, target_offsets
    )

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, tile_x, tile_y)
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

    # Save infill image to disk (use original x, y in filename for clarity)
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

    # Download result (use original x, y in filename for clarity)
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
        qx, qy = tile_x + dx, tile_y + dy
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

```

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


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
  conn: sqlite3.Connection,
  x: int,
  y: int,
  target_offsets: list[tuple[int, int]] | None = None,
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
    target_offsets: Optional list of (dx, dy) offsets specifying which quadrants
      to generate. If None, generates all quadrants that don't have generations.
      If specified, only these quadrants will be marked for generation (even if
      they already have generations when overwrite is True).

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

      # Determine if this quadrant should be generated
      if target_offsets is not None:
        # User specified which quadrants to generate
        should_generate = (dx, dy) in target_offsets
      else:
        # Default behavior: generate if no existing generation
        should_generate = gen_bytes is None

      if gen_bytes is not None and not should_generate:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render (either needs generation or explicitly targeted)
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        if should_generate:
          render_positions.append((dx, dy))
          status = "(TARGET)" if target_offsets else "(needs generation)"
          print(f"      ({qx}, {qy}): using render {status}")
        else:
          print(f"      ({qx}, {qy}): using render (context only)")

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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
  target_position: str | None = None,
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
    x: Tile x coordinate (or target quadrant x if target_position is specified)
    y: Tile y coordinate (or target quadrant y if target_position is specified)
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), x,y is the tile anchor and all non-generated quadrants are generated.
      If specified, x,y is the target quadrant and the tile anchor is calculated based on
      where the target should be positioned within the tile.

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

    # Calculate tile anchor and target offsets based on target_position
    if target_position is not None:
      # x, y is the target quadrant; calculate tile anchor
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      target_offset = parse_target_position(target_position)
      target_offsets = [target_offset]
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print(f"   Context quadrants: ", end="")
      for dx in range(2):
        for dy in range(2):
          qx, qy = tile_x + dx, tile_y + dy
          is_target = (dx, dy) == target_offset
          marker = " [TARGET]" if is_target else ""
          print(f"({qx},{qy}){marker}", end=" ")
      print()
      if debug:
        print("   [DEBUG MODE]")
      print(f"{'=' * 60}")
    else:
      # x, y is the tile anchor (default behavior)
      tile_x, tile_y = x, y
      target_offsets = None
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
      print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, tile_x, tile_y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, tile_x, tile_y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if target_position is not None:
      # Only check if the specific target quadrant has a generation
      if not debug and not overwrite:
        gen_bytes = get_quadrant_generation_local(conn, x, y)
        if gen_bytes is not None:
          print(f"\n⏭️  Skipping - quadrant ({x}, {y}) already generated (use --overwrite)")
          return None
    else:
      # Check all quadrants in the tile
      if not debug and not overwrite and check_all_quadrants_generated(conn, tile_x, tile_y):
        print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
        return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(
      conn, tile_x, tile_y, target_offsets
    )

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, tile_x, tile_y)
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

    # Save infill image to disk (use original x, y in filename for clarity)
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

    # Download result (use original x, y in filename for clarity)
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
        qx, qy = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = tile_x + dx, tile_y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - context only")

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

```

Now let me update the CLI to add the `--target-position` argument:

```python
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
from pathlib import Path
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
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
  upload_to_gcs,
)

# Oxen API Model Constants
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
INPAINTING_MODEL_ID = "cannoneyed-spectacular-beige-hyena"
GENERATION_MODEL_ID = "cannoneyed-odd-blue-marmot"
INFILL_MODEL_ID = "cannoneyed-satisfactory-harlequin-minnow"

# =============================================================================
# Oxen API Integration
# =============================================================================


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

  # We're experimenting with using *only* the omni model
  model = OMNI_MODEL_ID
  prompt = "Fill in the outlined section with the missing pixels corresponding to the <isometric nyc pixel art> style, removing the border and exactly following the shape/style/structure of the surrounding image (if present)."

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
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


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
  conn: sqlite3.Connection,
  x: int,
  y: int,
  target_offsets: list[tuple[int, int]] | None = None,
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
    target_offsets: Optional list of (dx, dy) offsets specifying which quadrants
      to generate. If None, generates all quadrants that don't have generations.
      If specified, only these quadrants will be marked for generation (even if
      they already have generations when overwrite is True).

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

      # Determine if this quadrant should be generated
      if target_offsets is not None:
        # User specified which quadrants to generate
        should_generate = (dx, dy) in target_offsets
      else:
        # Default behavior: generate if no existing generation
        should_generate = gen_bytes is None

      if gen_bytes is not None and not should_generate:
        # Use the existing generation
        img = png_bytes_to_image(gen_bytes)
        infill.paste(img, (pos_x, pos_y))
        print(f"      ({qx}, {qy}): using existing generation")
      else:
        # Use the render (either needs generation or explicitly targeted)
        render_bytes = get_quadrant_render(conn, qx, qy)
        if render_bytes is None:
          raise ValueError(f"Missing render for quadrant ({qx}, {qy})")
        img = png_bytes_to_image(render_bytes)
        infill.paste(img, (pos_x, pos_y))
        if should_generate:
          render_positions.append((dx, dy))
          status = "(TARGET)" if target_offsets else "(needs generation)"
          print(f"      ({qx}, {qy}): using render {status}")
        else:
          print(f"      ({qx}, {qy}): using render (context only)")

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
  api_key: str | None = None,
  api_key_env_var: str = "OXEN_INFILL_V02_API_KEY",
  target_position: str | None = None,
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
    x: Tile x coordinate (or target quadrant x if target_position is specified)
    y: Tile y coordinate (or target quadrant y if target_position is specified)
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill
    api_key: Oxen API key (if not provided, reads from api_key_env_var)
    api_key_env_var: Environment variable name for API key (default: OXEN_INFILL_V02_API_KEY)
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), x,y is the tile anchor and all non-generated quadrants are generated.
      If specified, x,y is the target quadrant and the tile anchor is calculated based on
      where the target should be positioned within the tile.

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  load_dotenv()

  if not debug:
    if api_key is None:
      api_key = os.getenv(api_key_env_var)
    if not api_key:
      raise ValueError(
        f"{api_key_env_var} environment variable not set. "
        "Please add it to your .env file."
      )

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

    # Calculate tile anchor and target offsets based on target_position
    if target_position is not None:
      # x, y is the target quadrant; calculate tile anchor
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      target_offset = parse_target_position(target_position)
      target_offsets = [target_offset]
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print(f"   Context quadrants: ", end="")
      for dx in range(2):
        for dy in range(2):
          qx, qy = tile_x + dx, tile_y + dy
          is_target = (dx, dy) == target_offset
          marker = " [TARGET]" if is_target else ""
          print(f"({qx},{qy}){marker}", end=" ")
      print()
      if debug:
        print("   [DEBUG MODE]")
      print(f"{'=' * 60}")
    else:
      # x, y is the tile anchor (default behavior)
      tile_x, tile_y = x, y
      target_offsets = None
      print(f"\n{'=' * 60}")
      print(f"🎯 Generating tile at ({x}, {y})" + (" [DEBUG MODE]" if debug else ""))
      print(f"{'=' * 60}")

    # Step 1: Ensure all quadrants are rendered
    print("\n📋 Step 1: Checking renders...")
    if not check_all_quadrants_rendered(conn, tile_x, tile_y):
      print("   Some quadrants need rendering...")
      render_tile_quadrants(conn, config, tile_x, tile_y, renders_dir, port)
    else:
      print("   ✓ All quadrants already rendered")

    # Step 2: Check if we should skip (not in debug mode)
    if target_position is not None:
      # Only check if the specific target quadrant has a generation
      if not debug and not overwrite:
        gen_bytes = get_quadrant_generation_local(conn, x, y)
        if gen_bytes is not None:
          print(f"\n⏭️  Skipping - quadrant ({x}, {y}) already generated (use --overwrite)")
          return None
    else:
      # Check all quadrants in the tile
      if not debug and not overwrite and check_all_quadrants_generated(conn, tile_x, tile_y):
        print("\n⏭️  Skipping - all quadrants already generated (use --overwrite)")
        return None

    # Step 3: Create infill image
    print("\n📋 Step 2: Creating infill image...")
    print("   Checking quadrant status:")
    infill_image, render_positions = create_infill_image(
      conn, tile_x, tile_y, target_offsets
    )

    # Get quadrant dimensions for border drawing
    sample_render = get_quadrant_render(conn, tile_x, tile_y)
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

    # Save infill image to disk (use original x, y in filename for clarity)
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

    # Download result (use original x, y in filename for clarity)
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
        qx, qy = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)

        if save_quadrant_generation(conn, config, qx, qy, png_bytes):
          print(f"   ✓ Saved generation for quadrant ({qx}, {qy})")
        else:
          print(f"   ⚠️  Failed to save generation for quadrant ({qx}, {qy})")
      else:
        qx, qy = tile_x + dx, tile_y + dy
        print(f"   ⏭️  Skipping quadrant ({qx}, {qy}) - context only")

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
    help="Quadrant x coordinate (tile anchor x, or target x if --target-position is used)",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Quadrant y coordinate (tile anchor y, or target y if --target-position is used)",
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
  parser.add_argument(
    "--target-position",
    "-t",
    choices=["tl", "tr", "bl", "br"],
    help=(
      "Position of target quadrant (x,y) within the 2x2 tile context. "
      "tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right. "
      "If specified, (x,y) is the target to generate and surrounding quadrants "
      "provide context. If not specified, (x,y) is the tile anchor and all "
      "non-generated quadrants are generated."
    ),
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
      target_position=args.target_position,
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

```

Now let me update `generate_tile_omni.py` to also support the `target_position` parameter:

```python
"""
Generate pixel art for a tile using the Oxen.ai omni model.

This is a thin wrapper around generate_tile.py that uses the omni model
configuration (OXEN_OMNI_v04_API_KEY environment variable).

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
from pathlib import Path

from isometric_nyc.e2e_generation.generate_tile import generate_tile
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  start_web_server,
)

# Default API key environment variable for the omni model
OMNI_API_KEY_ENV_VAR = "OXEN_OMNI_v04_API_KEY"


def generate_tile_omni(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
  target_position: str | None = None,
) -> Path | None:
  """
  Generate pixel art for a tile using the omni model.

  This is a convenience wrapper around generate_tile() that uses
  OXEN_OMNI_v04_API_KEY for authentication.

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate (or target quadrant x if target_position is specified)
    y: Tile y coordinate (or target quadrant y if target_position is specified)
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), x,y is the tile anchor and all non-generated quadrants are generated.

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  return generate_tile(
    generation_dir=generation_dir,
    x=x,
    y=y,
    port=port,
    bucket_name=bucket_name,
    overwrite=overwrite,
    debug=debug,
    api_key_env_var=OMNI_API_KEY_ENV_VAR,
    target_position=target_position,
  )


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API (omni model)."
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

    result = generate_tile_omni(
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

```

```python
"""
Generate pixel art for a tile using the Oxen.ai omni model.

This is a thin wrapper around generate_tile.py that uses the omni model
configuration (OXEN_OMNI_v04_API_KEY environment variable).

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <generation_dir> <x> <y>

Where x and y are the tile coordinates (tile at position x, y has its top-left
quadrant at quadrant position (x, y)).
"""

import argparse
from pathlib import Path

from isometric_nyc.e2e_generation.generate_tile import generate_tile
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  start_web_server,
)

# Default API key environment variable for the omni model
OMNI_API_KEY_ENV_VAR = "OXEN_OMNI_v04_API_KEY"


def generate_tile_omni(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = "isometric-nyc-infills",
  overwrite: bool = False,
  debug: bool = False,
  target_position: str | None = None,
) -> Path | None:
  """
  Generate pixel art for a tile using the omni model.

  This is a convenience wrapper around generate_tile() that uses
  OXEN_OMNI_v04_API_KEY for authentication.

  Args:
    generation_dir: Path to the generation directory
    x: Tile x coordinate (or target quadrant x if target_position is specified)
    y: Tile y coordinate (or target quadrant y if target_position is specified)
    port: Web server port
    bucket_name: GCS bucket for uploads
    overwrite: If True, regenerate even if generations exist
    debug: If True, skip model inference and just save the infill
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), x,y is the tile anchor and all non-generated quadrants are generated.

  Returns:
    Path to the generated/infill image, or None if skipped
  """
  return generate_tile(
    generation_dir=generation_dir,
    x=x,
    y=y,
    port=port,
    bucket_name=bucket_name,
    overwrite=overwrite,
    debug=debug,
    api_key_env_var=OMNI_API_KEY_ENV_VAR,
    target_position=target_position,
  )


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API (omni model)."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Quadrant x coordinate (tile anchor x, or target x if --target-position is used)",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Quadrant y coordinate (tile anchor y, or target y if --target-position is used)",
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
  parser.add_argument(
    "--target-position",
    "-t",
    choices=["tl", "tr", "bl", "br"],
    help=(
      "Position of target quadrant (x,y) within the 2x2 tile context. "
      "tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right. "
      "If specified, (x,y) is the target to generate and surrounding quadrants "
      "provide context. If not specified, (x,y) is the tile anchor and all "
      "non-generated quadrants are generated."
    ),
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

    result = generate_tile_omni(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.bucket,
      args.overwrite,
      args.debug,
      args.target_position,
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

```

Let me check for any linter errors:

Let me test that the imports work and show the help text:

The implementation is complete. Here's a summary of the new `--target-position` parameter:

## Usage

```bash
# Generate quadrant (0,0) with it positioned in the bottom-right of the 2x2 tile
# This uses quadrants (-1,-1), (0,-1), (-1,0) as context
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0 --target-position br

# Short form with -t
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0 -t br
```

## How it works

When you specify `--target-position`, the meaning of `x` and `y` changes:

| Without `--target-position` | With `--target-position br` |
|---|---|
| `(x, y)` = tile anchor (top-left) | `(x, y)` = target quadrant to generate |
| All 4 quadrants generated if needed | Only target quadrant generated |
| Tile: (x,y), (x+1,y), (x,y+1), (x+1,y+1) | Tile anchor calculated from position |

## Position values

| Position | Meaning | Tile anchor for target (0,0) |
|---|---|---|
| `tl` | Target is top-left | (0, 0) |
| `tr` | Target is top-right | (-1, 0) |
| `bl` | Target is bottom-left | (0, -1) |
| `br` | Target is bottom-right | (-1, -1) |

## Your example

With `x=0, y=0, --target-position br`:
- Tile anchor: `(-1, -1)`
- Context quadrants: `(-1,-1)`, `(0,-1)`, `(-1,0)` 
- Target quadrant: `(0,0)` [bottom-right]

This matches your desired context `[(-1,-1), (-1,0), (0,-1), (0,0)]`.

```shellscript
# Generate quadrant (0,0) with it positioned in the bottom-right of the 2x2 tile
# This uses quadrants (-1,-1), (0,-1), (-1,0) as context
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0 --target-position br

# Short form with -t
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0 -t br
```

---

**User**

OK - for the @src/isometric_nyc/e2e_generation/generate_tile_omni.py DO NOT USE ANY FUNCTIONS FROM @src/isometric_nyc/e2e_generation/generate_tile.py , use the functions from @src/isometric_nyc/e2e_generation/generate_omni.py (just like @src/isometric_nyc/e2e_generation/app.py )

---

**Cursor**

I'll rewrite `generate_tile_omni.py` to use the functions from `generate_omni.py` instead of `generate_tile.py`:

```python
"""
Generate pixel art for a tile using the Oxen.ai omni model.

This script generates pixel art for quadrants at specific coordinates.
It uses the generation library from generate_omni.py, same as the web app.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.
"""

import argparse
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.generate_omni import (
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# =============================================================================
# Target Position Utilities
# =============================================================================

# Mapping from position name to (dx, dy) offset within tile
TARGET_POSITION_OFFSETS = {
  "tl": (0, 0),  # top-left
  "tr": (1, 0),  # top-right
  "bl": (0, 1),  # bottom-left
  "br": (1, 1),  # bottom-right
}


def parse_target_position(position: str) -> tuple[int, int]:
  """
  Parse a target position string into (dx, dy) offset within the tile.

  Args:
    position: One of "tl", "tr", "bl", "br"

  Returns:
    Tuple of (dx, dy) offset where dx=0 is left, dx=1 is right,
    dy=0 is top, dy=1 is bottom.

  Raises:
    ValueError: If position is invalid
  """
  position = position.lower().strip()
  if position not in TARGET_POSITION_OFFSETS:
    valid = ", ".join(TARGET_POSITION_OFFSETS.keys())
    raise ValueError(f"Invalid target position '{position}'. Must be one of: {valid}")
  return TARGET_POSITION_OFFSETS[position]


def calculate_tile_anchor(
  target_x: int, target_y: int, target_position: str
) -> tuple[int, int]:
  """
  Calculate the tile anchor (top-left quadrant) given a target quadrant and its position.

  For example, if target is (0, 0) and target_position is "br" (bottom-right),
  then the tile anchor is (-1, -1) because:
  - Tile contains: (-1, -1), (0, -1), (-1, 0), (0, 0)
  - Target (0, 0) is at offset (1, 1) from anchor (-1, -1)

  Args:
    target_x: X coordinate of the target quadrant
    target_y: Y coordinate of the target quadrant
    target_position: Position of target within tile ("tl", "tr", "bl", "br")

  Returns:
    Tuple of (anchor_x, anchor_y) for the tile's top-left quadrant
  """
  dx, dy = parse_target_position(target_position)
  return (target_x - dx, target_y - dy)


def get_tile_quadrants(anchor_x: int, anchor_y: int) -> list[tuple[int, int]]:
  """
  Get all 4 quadrant coordinates for a tile anchored at (anchor_x, anchor_y).

  Returns:
    List of (x, y) tuples for TL, TR, BL, BR quadrants
  """
  return [
    (anchor_x, anchor_y),  # TL
    (anchor_x + 1, anchor_y),  # TR
    (anchor_x, anchor_y + 1),  # BL
    (anchor_x + 1, anchor_y + 1),  # BR
  ]


# =============================================================================
# Main Generation Logic
# =============================================================================


def generate_tile_omni(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the omni model.

  Args:
    generation_dir: Path to the generation directory
    x: Quadrant x coordinate (or target x if target_position is specified)
    y: Quadrant y coordinate (or target y if target_position is specified)
    port: Web server port for rendering
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      If None (default), generates all 4 quadrants in the tile starting at (x, y).
      If specified, (x, y) is the target quadrant to generate, positioned within
      the 2x2 tile context.

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🎯 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🎯 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation using the shared library
    result = run_generation_for_quadrants(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      port=port,
      status_callback=status_callback,
    )

    if result["success"]:
      print(f"\n{'=' * 60}")
      print(f"✅ Generation complete: {result['message']}")
      print(f"{'=' * 60}")
      return True
    else:
      print(f"\n{'=' * 60}")
      print(f"❌ Generation failed: {result['error']}")
      print(f"{'=' * 60}")
      return False

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for a tile using the Oxen API (omni model)."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    help="Quadrant x coordinate (tile anchor x, or target x if --target-position is used)",
  )
  parser.add_argument(
    "y",
    type=int,
    help="Quadrant y coordinate (tile anchor y, or target y if --target-position is used)",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--target-position",
    "-t",
    choices=["tl", "tr", "bl", "br"],
    help=(
      "Position of target quadrant (x,y) within the 2x2 tile context. "
      "tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right. "
      "If specified, (x,y) is the target to generate and surrounding quadrants "
      "provide context. If not specified, (x,y) is the tile anchor and all "
      "4 quadrants are generated."
    ),
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

    success = generate_tile_omni(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
    )
    return 0 if success else 1

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

```

Done! I've refactored `generate_tile_omni.py` to use functions from `generate_omni.py` instead of `generate_tile.py`. 

Key changes:

1. **Uses `run_generation_for_quadrants`** from `generate_omni.py` - the same function used by `app.py`
2. **No imports from `generate_tile.py`** - completely independent
3. **Local target position utilities** - `parse_target_position()`, `calculate_tile_anchor()`, `get_tile_quadrants()`
4. **Simplified parameters** - removed `--bucket`, `--overwrite`, `--debug` since `run_generation_for_quadrants()` handles its own logic

## Usage

```bash
# Generate all 4 quadrants of tile starting at (0, 0)
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0

# Generate only quadrant (0, 0) positioned in the bottom-right of the 2x2 tile
# Uses (-1,-1), (0,-1), (-1,0) as context
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0 -t br
```

The script now works like the web app - it calls `run_generation_for_quadrants()` with the list of quadrants to generate, and that function handles rendering, template building, API calls, and saving.

```shellscript
# Generate all 4 quadrants of tile starting at (0, 0)
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0

# Generate only quadrant (0, 0) positioned in the bottom-right of the 2x2 tile
# Uses (-1,-1), (0,-1), (-1,0) as context
uv run python src/isometric_nyc/e2e_generation/generate_tile_omni.py <gen_dir> 0 0 -t br
```

