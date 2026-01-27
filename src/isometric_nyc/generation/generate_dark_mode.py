"""
Generate dark mode (nighttime) version of tiles using the Gemini nano banana model.

This script generates nighttime versions of isometric pixel art tiles.
Each tile consists of 4 quadrants: (x,y), (x+1,y), (x,y+1), (x+1,y+1).
The x,y coordinate specifies the top-left quadrant of the 2x2 tile.

It uses pixel art generations from the database as input, then calls Gemini to
generate a nighttime version of the tile.

Usage:
  # Generate dark mode for 2x2 tile with top-left at (0, 0):
  uv run python src/isometric_nyc/generation/generate_dark_mode.py \\
    generations/nyc 0 0 --output-dir synthetic_data/datasets/dark_mode

  # Generate for multiple 2x2 tiles (each coord is the top-left):
  uv run python src/isometric_nyc/generation/generate_dark_mode.py \\
    generations/nyc --quadrants "(0,0),(2,0)" --output-dir synthetic_data/datasets/dark_mode

  # Generate with custom prompt override:
  uv run python src/isometric_nyc/generation/generate_dark_mode.py \\
    generations/nyc 0 0 --output-dir synthetic_data/datasets/dark_mode \\
    --prompt "Custom prompt for dark mode generation"

Output structure:
  <output-dir>/
    inputs/<x>_<y>.png        # The 2x2 tile pixel art used as input (daytime)
    generations/<x>_<y>.png   # The generated dark mode (nighttime) version
"""

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from isometric_nyc.generation.generate_tile_nano_banana import (
  parse_quadrant_list,
  render_quadrant,
)
from isometric_nyc.generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  png_bytes_to_image,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Type alias for quadrant position
QuadrantOffset = tuple[int, int]

# Load environment variables
load_dotenv()

# Default prompt for dark mode generation
# TODO: This prompt will be manually refined later
DARK_MODE_PROMPT = """
DARK MODE / NIGHTTIME GENERATION TASK

Transform this isometric pixel art image from daytime to nighttime. The goal is to create a realistic nighttime version that:

1. LIGHTING:
   - Reduce overall brightness significantly (nighttime ambient light)
   - Add warm yellow/orange lights in building windows (randomly lit, not all windows)
   - Add subtle blue-ish moonlight tint to exposed surfaces
   - Street lights should cast warm pools of light on streets/sidewalks
   - Neon signs and shop fronts should glow with vibrant colors
   - THERE SHOULD BE NO REFLECTION OF CITY LIGHTS OR TEXTURES IN THE WATER - THE WATER SHOULD SIMPLY BE DARKENED

2. COLORS:
   - Shift all colors toward cooler/darker tones
   - Shadows should be deep blue/purple, not just black
   - Sky areas (if visible) should be dark blue/black

3. ATMOSPHERE:
   - Maintain all the structural details of buildings and streets
   - Keep the isometric pixel art style intact
   - The image should feel cozy and urban-nightlife-like

4. PRESERVE:
   - Exact same perspective and composition
   - All building shapes, roads, and structural elements
   - The pixel art aesthetic and style
   - DO NOT ADD ANY NEW WATER OR FEATURES

Output a nighttime version of the exact same scene. DO NOT ADD ANY BUILDINGS IF THEY ARE NOT PRESENT IN THE INPUT IMAGE.
""".strip()

# DARK_MODE_PROMPT = """
# DARK MODE / NIGHTTIME GENERATION TASK

# Transform this isometric pixel art image from daytime to nighttime. The goal is to create a realistic nighttime version that:

# 1. LIGHTING:
#    - Reduce overall brightness significantly (nighttime ambient light)
#    - Add subtle blue-ish moonlight tint to exposed surfaces

# 2. COLORS:
#    - Shift all colors toward cooler/darker tones
#    - Shadows should be deep blue/purple, not just black
#    - Sky areas (if visible) should be dark blue/black

# 4. PRESERVE:
#    - Exact same perspective and composition
#    - The pixel art aesthetic and style
#    - DO NOT ADD ANY NEW WATER OR FEATURES

# Output a nighttime version of the exact same scene. DO NOT ADD ANY BUILDINGS IF THEY ARE NOT PRESENT IN THE INPUT IMAGE.
# """.strip()


def call_gemini_dark_mode(
  input_image: Image.Image,
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate a dark mode version of the input image.

  Args:
    input_image: The input image (daytime pixel art) to transform
    prompt: Optional custom prompt text (overrides the default prompt)
    debug_dir: Optional directory to save debug images

  Returns:
    Generated PIL Image (nighttime version)

  Raises:
    ValueError: If the API key is not found or generation fails
  """
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment")

  client = genai.Client(api_key=api_key)

  # Create debug directory if provided
  if debug_dir:
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"   📁 Saving debug images to: {debug_dir}")

  # Build contents list with images
  contents: list = []

  # Upload input image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    input_path = tmp.name
    input_image.save(input_path)

  # Save input to debug dir
  if debug_dir:
    input_debug_path = debug_dir / "input.png"
    input_image.save(input_debug_path)
    print(f"      ✓ Saved input: {input_debug_path}")

  try:
    input_ref = client.files.upload(file=input_path)
    contents.append(input_ref)
  finally:
    Path(input_path).unlink(missing_ok=True)

  # Use custom prompt or default
  generation_prompt = prompt if prompt else DARK_MODE_PROMPT

  contents.append(generation_prompt)

  # Save prompt to debug dir
  if debug_dir:
    prompt_debug_path = debug_dir / "prompt.txt"
    with open(prompt_debug_path, "w") as f:
      f.write(generation_prompt)
    print(f"      ✓ Saved prompt: {prompt_debug_path}")

  # Log the prompt
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST (DARK MODE)")
  print("=" * 60)
  print(f"   📐 Input image: {input_image.size[0]}x{input_image.size[1]}")
  print("\n   📝 PROMPT (first 200 chars):")
  print("-" * 60)
  print(f"   {generation_prompt[:200]}...")
  print("-" * 60)
  print("=" * 60 + "\n")

  print("   🤖 Calling Gemini API...")
  response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated dark mode image")
      # Convert to PIL Image
      pil_img = image._pil_image

      # Save generated image to debug dir
      if debug_dir:
        generated_debug_path = debug_dir / "generated.png"
        pil_img.save(generated_debug_path)
        print(f"      ✓ Saved generated: {generated_debug_path}")

      return pil_img

  raise ValueError("No image in Gemini response")


def generate_dark_mode_tile(
  generation_dir: Path,
  x: int,
  y: int,
  output_dir: Path,
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  use_render: bool = False,
) -> bool:
  """
  Generate a dark mode version for a 2x2 tile at the specified coordinates.

  The tile consists of 4 quadrants: (x,y), (x+1,y), (x,y+1), (x+1,y+1).

  This function:
  1. Gets pixel art generations from the database (or renders if --use-render)
  2. Stitches the 4 quadrants into a single 2x2 tile image
  3. Calls Gemini to generate a dark mode version
  4. Saves both the input and generation to the output directory

  Args:
    generation_dir: Path to the generation directory containing quadrants.db
    x: X coordinate of the top-left quadrant of the 2x2 tile
    y: Y coordinate of the top-left quadrant of the 2x2 tile
    output_dir: Output directory for saving inputs and generations
    port: Web server port for rendering (only used with use_render=True)
    prompt: Optional custom prompt text
    use_render: If True, use 3D renders instead of pixel art generations

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Create output directories
    inputs_dir = output_dir / "inputs"
    generations_dir = output_dir / "generations"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)

    input_type = "render" if use_render else "pixel art"
    print(f"\n{'=' * 60}")
    print(f"🌙 Generating dark mode for tile at ({x}, {y})")
    print(f"   Input type: {input_type}")
    print(f"   Output: {output_dir}")
    print(f"{'=' * 60}")

    # Get all 4 quadrants for this 2x2 tile
    # Offsets from top-left (x, y): TL=(0,0), TR=(1,0), BL=(0,1), BR=(1,1)
    quadrant_offsets: list[QuadrantOffset] = [
      (0, 0),
      (1, 0),
      (0, 1),
      (1, 1),
    ]
    quadrant_images: dict[QuadrantOffset, Image.Image] = {}

    # Collect quadrant images (generation or render)
    for dx, dy in quadrant_offsets:
      qx, qy = x + dx, y + dy

      if not use_render:
        # Default: use pixel art generations
        gen_bytes = get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
          quadrant_images[(dx, dy)] = png_bytes_to_image(gen_bytes)
          print(f"   ✓ Using pixel art for ({qx}, {qy})")
          continue
        else:
          print(f"   ❌ No pixel art generation for ({qx}, {qy})")
          return False

      # use_render=True: use 3D renders
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes is None:
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)

      if render_bytes is None:
        print(f"   ❌ Failed to render quadrant ({qx}, {qy})")
        return False

      quadrant_images[(dx, dy)] = png_bytes_to_image(render_bytes)
      print(f"   ✓ Got render for ({qx}, {qy})")

    # Stitch quadrants into a single tile image
    tile_image = stitch_quadrants_to_tile(quadrant_images)
    print(f"   📐 Stitched tile: {tile_image.size[0]}x{tile_image.size[1]}")

    # Save the input image
    input_filename = f"{x}_{y}.png"
    input_path = inputs_dir / input_filename
    tile_image.save(input_path)
    print(f"   💾 Saved input: {input_path}")

    # Generate dark mode version
    debug_dir = output_dir / "debug" / f"{x}_{y}"
    dark_mode_image = call_gemini_dark_mode(
      input_image=tile_image,
      prompt=prompt,
      debug_dir=debug_dir,
    )

    # Resize to 1024x1024 if needed
    if dark_mode_image.size != (1024, 1024):
      print(f"   📐 Resizing from {dark_mode_image.size} to (1024, 1024)...")
      dark_mode_image = dark_mode_image.resize((1024, 1024), Image.Resampling.LANCZOS)

    # Convert to RGB
    dark_mode_rgb = dark_mode_image.convert("RGB")

    # Save the generation (dark mode) as PNG
    generation_filename = f"{x}_{y}.png"
    generation_path = generations_dir / generation_filename
    dark_mode_rgb.save(generation_path)
    print(f"   💾 Saved dark mode: {generation_path}")

    print(f"\n{'=' * 60}")
    print("✅ Dark mode generated successfully")
    print(f"   Input (day): {input_path}")
    print(f"   Output (night): {generation_path}")
    print(f"{'=' * 60}")

    return True

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate dark mode (nighttime) version of tiles using the Gemini nano banana model.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Generate dark mode for tile at (0,0) using pixel art:
  %(prog)s generations/nyc 0 0 --output-dir synthetic_data/datasets/dark_mode

  # Generate using 3D renders as input (rare):
  %(prog)s generations/nyc 0 0 --output-dir synthetic_data/datasets/dark_mode --use-render

  # Generate with custom prompt:
  %(prog)s generations/nyc 0 0 --output-dir synthetic_data/datasets/dark_mode \\
    --prompt "Transform to nighttime with neon lights"
""",
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "x",
    type=int,
    nargs="?",
    default=None,
    help="X coordinate of the top-left quadrant of a 2x2 tile",
  )
  parser.add_argument(
    "y",
    type=int,
    nargs="?",
    default=None,
    help="Y coordinate of the top-left quadrant of a 2x2 tile",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Output directory for saving inputs (inputs/<x>_<y>.png) and generations (generations/<x>_<y>.png)",
  )
  parser.add_argument(
    "--quadrants",
    "-q",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of quadrant coordinates to generate dark mode for. "
      "Accepts formats like --quadrants '(0,0),(2,0)' or --quadrants '(0,0)' '(2,0)'. "
      "Each pair of coordinates is treated as the TL of a 2x2 tile."
    ),
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
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Custom prompt text for dark mode generation (overrides default)",
  )
  parser.add_argument(
    "--use-render",
    action="store_true",
    help="Use 3D renders as input instead of pixel art generations (default: use pixel art)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()
  output_dir = args.output_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  # Parse quadrants if provided
  tile_coords: list[tuple[int, int]] = []

  if args.quadrants:
    try:
      for quad_str in args.quadrants:
        parsed = parse_quadrant_list(quad_str)
        tile_coords.extend(parsed)
      # Remove duplicates while preserving order
      seen = set()
      unique_coords = []
      for c in tile_coords:
        if c not in seen:
          seen.add(c)
          unique_coords.append(c)
      tile_coords = unique_coords
      print(f"📋 Parsed tile coordinates: {tile_coords}")
    except ValueError as e:
      print(f"❌ Error parsing --quadrants: {e}")
      return 1
  elif args.x is not None and args.y is not None:
    tile_coords = [(args.x, args.y)]
  else:
    print("❌ Error: Either --quadrants or both x and y coordinates must be provided")
    parser.print_help()
    return 1

  web_server = None

  try:
    # Only start web server if using renders (not the default)
    if args.use_render and not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    # Generate dark mode for each tile
    success_count = 0
    for tx, ty in tile_coords:
      try:
        success = generate_dark_mode_tile(
          generation_dir,
          x=tx,
          y=ty,
          output_dir=output_dir,
          port=args.port,
          prompt=args.prompt,
          use_render=args.use_render,
        )
        if success:
          success_count += 1
      except Exception as e:
        print(f"❌ Error generating dark mode for ({tx}, {ty}): {e}")

    print(f"\n{'=' * 60}")
    print(f"🏁 Completed: {success_count}/{len(tile_coords)} tiles processed")
    print(f"{'=' * 60}")

    return 0 if success_count == len(tile_coords) else 1

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
