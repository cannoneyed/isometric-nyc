# Nano banana generation task
_Exported on 12/18/2025 at 11:56:27 EST from Cursor (2.2.20)_

---

**User**

Implement @tasks/029_nano_banana_generation.md 

---

**Cursor**

I'll implement this task. Let me first read the relevant source files to understand the existing logic.

Let me also examine the shared module and the generate_omni.py to understand the template construction and pixel splicing logic:

Let me also look at the infill template module to understand the template building:

Now I have a comprehensive understanding of the codebase. Let me create the `generate_tile_nano_banana.py` script:

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text

  Returns:
    Generated PIL Image

  Raises:
    ValueError: If the API key is not found or generation fails
  """
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment")

  client = genai.Client(api_key=api_key)

  # Build contents list with images
  contents: list = []

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  print("   🤖 Calling Gemini API...")
  response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      img_bytes = image._pil_image.tobytes()
      pil_img = image._pil_image
      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print("   ⚠️ No reference coordinates provided, generation may have inconsistent style")
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status("complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)")
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

Let me check for lint errors and fix them:

Let me verify the script can be imported correctly by running a quick syntax check:

Let me verify the CLI help works correctly:

I've created the `generate_tile_nano_banana.py` script at `src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py`. Here's a summary of what was implemented:

## Features

The script accepts all the parameters specified in the task:

| Parameter | Description |
|-----------|-------------|
| `generation_dir` | Path to the generation directory containing `quadrants.db` |
| `x`, `y` | Quadrant coordinates (tile anchor or target position) |
| `--target-position` / `-t` | Position within 2x2 tile context (`tl`, `tr`, `bl`, `br`) |
| `--prompt` / `-p` | Additional prompt text for the generation model |
| `--no-save` | Skip saving generated quadrants to the database |
| `--references` / `-r` | List of `"(x,y)"` TL quadrant coordinates for style references |

## Key Implementation Details

1. **Template Construction**: Uses the same `TemplateBuilder` and `InfillRegion` classes from `generate_omni.py` to construct templates with red borders

2. **Reference Images**: Loads 2x2 tile images from the database by stitching 4 quadrants together based on the provided TL coordinates

3. **Gemini Generation**: Calls `gemini-2.0-flash-preview-image-generation` with the template and reference images

4. **Output Resizing**: Generated images are resized to 1024x1024 (resulting in 512x512 quadrants)

5. **Pixel Splicing**: Extracts quadrants from the generated image and saves them to the database using the same logic as `generate_tile_omni.py`

## Example Usage

```bash
# Generate a single quadrant at (0,0) positioned bottom-right in the 2x2 context
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 0 0 -t br \
  --references "(0,2)" "(2,0)" \
  --prompt "Include more trees in the scene"

# Generate all 4 quadrants starting at tile anchor (0,0)
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 0 0 \
  --references "(2,2)" "(4,0)"
```

```shellscript
# Generate a single quadrant at (0,0) positioned bottom-right in the 2x2 context
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 0 0 -t br \
  --references "(0,2)" "(2,0)" \
  --prompt "Include more trees in the scene"

# Generate all 4 quadrants starting at tile anchor (0,0)
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 0 0 \
  --references "(2,2)" "(4,0)"
```

---

**User**

Ensure that the prompt and reference images get logged before sending to the model

---

**Cursor**

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text

  Returns:
    Generated PIL Image

  Raises:
    ValueError: If the API key is not found or generation fails
  """
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment")

  client = genai.Client(api_key=api_key)

  # Build contents list with images
  contents: list = []

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print(f"\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
  print("-" * 60)
  print("=" * 60 + "\n")

  print("   🤖 Calling Gemini API...")
  response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=contents,
    config=types.GenerateContentConfig(
      response_modalities=["TEXT", "IMAGE"],
      image_config=types.ImageConfig(
        aspect_ratio="1:1",
      ),
    ),
  )

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      img_bytes = image._pil_image.tobytes()
      pil_img = image._pil_image
      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print("   ⚠️ No reference coordinates provided, generation may have inconsistent style")
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status("complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)")
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

Done! I've added comprehensive logging before the Gemini API call. Now the script will output:

```
============================================================
📤 GEMINI API REQUEST
============================================================
   📐 Template image size: (1024, 1024)
   📷 Reference images: 2
      - Reference 1: (1024, 1024)
      - Reference 2: (1024, 1024)

   📝 PROMPT:
------------------------------------------------------------
   **Task:** Fill in the outlined section (marked with red border)...
   [full prompt text]
------------------------------------------------------------
============================================================

   🤖 Calling Gemini API...
```

This logs:
- Template image dimensions
- Number of reference images with their individual sizes
- The complete prompt text (including any additional instructions from `--prompt`)

```plaintext
============================================================
📤 GEMINI API REQUEST
============================================================
   📐 Template image size: (1024, 1024)
   📷 Reference images: 2
      - Reference 1: (1024, 1024)
      - Reference 2: (1024, 1024)

   📝 PROMPT:
------------------------------------------------------------
   **Task:** Fill in the outlined section (marked with red border)...
   [full prompt text]
------------------------------------------------------------
============================================================

   🤖 Calling Gemini API...
```

---

**User**

Ensure you save the generated image and reference images to a `nano_banana` subdir in the generation_dir (for debugging)

---

**Cursor**

I'll update the code to save debug images to a `nano_banana` subdirectory. I need to modify multiple functions:

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text
    debug_dir: Optional directory to save debug images (template, references, output)

  Returns:
    Generated PIL Image

  Raises:
    ValueError: If the API key is not found or generation fails
  """
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment")

  client = genai.Client(api_key=api_key)

  # Build contents list with images
  contents: list = []

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print("\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
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

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      img_bytes = image._pil_image.tobytes()
      pil_img = image._pil_image
      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print(
      "   ⚠️ No reference coordinates provided, generation may have inconsistent style"
    )
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status(
      "complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)"
    )
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text
    debug_dir: Optional directory to save debug images (template, references, output)

  Returns:
    Generated PIL Image

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

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  # Save template to debug dir
  if debug_dir:
    template_debug_path = debug_dir / "template.png"
    template_image.save(template_debug_path)
    print(f"      ✓ Saved template: {template_debug_path}")

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    # Save reference to debug dir
    if debug_dir:
      ref_debug_path = debug_dir / f"reference_{i + 1}.png"
      ref_img.save(ref_debug_path)
      print(f"      ✓ Saved reference {i + 1}: {ref_debug_path}")

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print("\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
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

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      img_bytes = image._pil_image.tobytes()
      pil_img = image._pil_image
      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print(
      "   ⚠️ No reference coordinates provided, generation may have inconsistent style"
    )
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status(
      "complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)"
    )
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text
    debug_dir: Optional directory to save debug images (template, references, output)

  Returns:
    Generated PIL Image

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

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  # Save template to debug dir
  if debug_dir:
    template_debug_path = debug_dir / "template.png"
    template_image.save(template_debug_path)
    print(f"      ✓ Saved template: {template_debug_path}")

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    # Save reference to debug dir
    if debug_dir:
      ref_debug_path = debug_dir / f"reference_{i + 1}.png"
      ref_img.save(ref_debug_path)
      print(f"      ✓ Saved reference {i + 1}: {ref_debug_path}")

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print("\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
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

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      pil_img = image._pil_image

      # Save generated image to debug dir
      if debug_dir:
        generated_debug_path = debug_dir / "generated.png"
        pil_img.save(generated_debug_path)
        print(f"      ✓ Saved generated: {generated_debug_path}")

      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print(
      "   ⚠️ No reference coordinates provided, generation may have inconsistent style"
    )
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status(
      "complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)"
    )
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

Now let me update `run_nano_banana_generation` to accept and pass the debug directory:

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text
    debug_dir: Optional directory to save debug images (template, references, output)

  Returns:
    Generated PIL Image

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

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  # Save template to debug dir
  if debug_dir:
    template_debug_path = debug_dir / "template.png"
    template_image.save(template_debug_path)
    print(f"      ✓ Saved template: {template_debug_path}")

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    # Save reference to debug dir
    if debug_dir:
      ref_debug_path = debug_dir / f"reference_{i + 1}.png"
      ref_img.save(ref_debug_path)
      print(f"      ✓ Saved reference {i + 1}: {ref_debug_path}")

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print("\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
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

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      pil_img = image._pil_image

      # Save generated image to debug dir
      if debug_dir:
        generated_debug_path = debug_dir / "generated.png"
        pil_img.save(generated_debug_path)
        print(f"      ✓ Saved generated: {generated_debug_path}")

      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  generation_dir: Path | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context
    generation_dir: Optional path to generation directory for saving debug images

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print(
      "   ⚠️ No reference coordinates provided, generation may have inconsistent style"
    )
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status(
      "complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)"
    )
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text
    debug_dir: Optional directory to save debug images (template, references, output)

  Returns:
    Generated PIL Image

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

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  # Save template to debug dir
  if debug_dir:
    template_debug_path = debug_dir / "template.png"
    template_image.save(template_debug_path)
    print(f"      ✓ Saved template: {template_debug_path}")

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    # Save reference to debug dir
    if debug_dir:
      ref_debug_path = debug_dir / f"reference_{i + 1}.png"
      ref_img.save(ref_debug_path)
      print(f"      ✓ Saved reference {i + 1}: {ref_debug_path}")

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print("\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
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

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      pil_img = image._pil_image

      # Save generated image to debug dir
      if debug_dir:
        generated_debug_path = debug_dir / "generated.png"
        pil_img.save(generated_debug_path)
        print(f"      ✓ Saved generated: {generated_debug_path}")

      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  generation_dir: Path | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context
    generation_dir: Optional path to generation directory for saving debug images

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print(
      "   ⚠️ No reference coordinates provided, generation may have inconsistent style"
    )
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  # Create debug directory for this generation
  debug_dir = None
  if generation_dir:
    debug_dir = generation_dir / "nano_banana"

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
      debug_dir=debug_dir,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status(
      "complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)"
    )
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text
    debug_dir: Optional directory to save debug images (template, references, output)

  Returns:
    Generated PIL Image

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

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  # Save template to debug dir
  if debug_dir:
    template_debug_path = debug_dir / "template.png"
    template_image.save(template_debug_path)
    print(f"      ✓ Saved template: {template_debug_path}")

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    # Save reference to debug dir
    if debug_dir:
      ref_debug_path = debug_dir / f"reference_{i + 1}.png"
      ref_img.save(ref_debug_path)
      print(f"      ✓ Saved reference {i + 1}: {ref_debug_path}")

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print("\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
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

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      pil_img = image._pil_image

      # Save generated image to debug dir
      if debug_dir:
        generated_debug_path = debug_dir / "generated.png"
        pil_img.save(generated_debug_path)
        print(f"      ✓ Saved generated: {generated_debug_path}")

      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  generation_dir: Path | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context
    generation_dir: Optional path to generation directory for saving debug images

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print(
      "   ⚠️ No reference coordinates provided, generation may have inconsistent style"
    )
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  # Create debug directory for this generation
  debug_dir = None
  if generation_dir:
    debug_dir = generation_dir / "nano_banana"

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
      debug_dir=debug_dir,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status(
      "complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)"
    )
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
      status_callback=status_callback,
      generation_dir=generation_dir,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

Let me verify the script still works:

Let me also save the prompt text to the debug directory for complete debugging:

```python
"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <generation_dir> <x> <y>

Where x and y are the quadrant coordinates to generate.

With --target-position, you can specify which quadrant in a 2x2 tile context
should be generated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

This generates quadrant (0,0) positioned at bottom-right, using (-1,-1), (0,-1), (-1,0)
as context quadrants.

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import io
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Target Position Utilities (shared with generate_tile_omni.py)
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
# Reference Image Parsing
# =============================================================================


def parse_reference_coords(ref_str: str) -> tuple[int, int]:
  """
  Parse a reference coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    ref_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  ref_str = ref_str.strip()
  # Remove optional parentheses
  if ref_str.startswith("(") and ref_str.endswith(")"):
    ref_str = ref_str[1:-1]
  parts = ref_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid reference coordinate format: {ref_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def load_reference_tile_image(
  conn: sqlite3.Connection,
  tl_x: int,
  tl_y: int,
) -> Image.Image | None:
  """
  Load a 2x2 tile image from the database given the TL quadrant coordinates.

  Args:
    conn: Database connection
    tl_x: Top-left quadrant x coordinate
    tl_y: Top-left quadrant y coordinate

  Returns:
    PIL Image of the 2x2 tile (1024x1024), or None if any quadrant is missing
  """
  quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dx, dy in quadrant_offsets:
    qx, qy = tl_x + dx, tl_y + dy
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes is None:
      print(f"   ⚠️ Reference quadrant ({qx}, {qy}) has no generation")
      return None
    quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

  # Stitch quadrants into a tile
  return stitch_quadrants_to_tile(quadrants)


# =============================================================================
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
    conn: Database connection
    config: Generation config dict
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    port: Web server port for rendering

  Returns:
    PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

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

    # Get screenshot as bytes
    screenshot_bytes = page.screenshot(type="png")

    page.close()
    context.close()
    browser.close()

  # Open as PIL image and split into quadrants
  full_tile = Image.open(io.BytesIO(screenshot_bytes))
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Gemini Nano Banana Generation
# =============================================================================


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    prompt: Optional additional prompt text
    debug_dir: Optional directory to save debug images (template, references, output)

  Returns:
    Generated PIL Image

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

  # Add template image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    template_path = tmp.name
    template_image.save(template_path)

  # Save template to debug dir
  if debug_dir:
    template_debug_path = debug_dir / "template.png"
    template_image.save(template_debug_path)
    print(f"      ✓ Saved template: {template_debug_path}")

  try:
    template_ref = client.files.upload(file=template_path)
    contents.append(template_ref)
  finally:
    Path(template_path).unlink(missing_ok=True)

  # Add reference images
  ref_paths = []
  for i, ref_img in enumerate(reference_images):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_path = tmp.name
      ref_img.save(ref_path)
      ref_paths.append(ref_path)

    # Save reference to debug dir
    if debug_dir:
      ref_debug_path = debug_dir / f"reference_{i + 1}.png"
      ref_img.save(ref_debug_path)
      print(f"      ✓ Saved reference {i + 1}: {ref_debug_path}")

    ref_ref = client.files.upload(file=ref_path)
    contents.append(ref_ref)

  # Build the prompt
  image_refs = []
  image_refs.append("Image A (template with red border indicating infill area)")
  for i in range(len(reference_images)):
    letter = chr(ord("B") + i)
    image_refs.append(f"Image {letter} (style reference {i + 1})")

  ref_images_str = ", ".join(
    [f"Image {chr(ord('B') + i)}" for i in range(len(reference_images))]
  )

  generation_prompt = f"""
**Task:** Fill in the outlined section (marked with red border) in Image A with isometric pixel art in the style of the reference images.

**Input Images:**
{chr(10).join(f"- {ref}" for ref in image_refs)}

**Instructions:**

1. Look at the red border in Image A - this marks the area to be filled with generated pixels.
2. Study the style of {ref_images_str} carefully - these show the exact visual style to match.
3. Fill the bordered area with pixel art that:
   - Matches the isometric pixel art style of the reference images exactly
   - Seamlessly blends with any surrounding context in Image A
   - Follows the structure/shapes hinted at by the render pixels inside the border
   - Removes the red border completely in the output

**Critical Style Rules (SimCity 3000 Aesthetic):**
1. **Clean Pixel Art:** The output must be clean, detailed but low-resolution, as if on a VGA monitor.
2. **No Whitebox/Untextured:** Every building must be fully textured with brick, concrete, glass, etc.
3. **Straight Isometric Lines:** Use 2:1 pixel slope for isometric projection.
4. **Consistent Lighting:** Match the lighting direction of the reference images.

**Output:**
- Generate the complete image with the bordered area filled in
- The output should be 1024x1024 pixels
- Remove all traces of the red border
""".strip()

  if prompt:
    generation_prompt += f"\n\n**Additional Instructions:**\n{prompt}"

  contents.append(generation_prompt)

  # Save prompt to debug dir
  if debug_dir:
    prompt_debug_path = debug_dir / "prompt.txt"
    with open(prompt_debug_path, "w") as f:
      f.write(generation_prompt)
    print(f"      ✓ Saved prompt: {prompt_debug_path}")

  # Log the prompt and reference images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST")
  print("=" * 60)
  print(f"   📐 Template image size: {template_image.size}")
  print(f"   📷 Reference images: {len(reference_images)}")
  for i, ref_img in enumerate(reference_images):
    print(f"      - Reference {i + 1}: {ref_img.size}")
  print("\n   📝 PROMPT:")
  print("-" * 60)
  for line in generation_prompt.split("\n"):
    print(f"   {line}")
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

  # Clean up reference files
  for ref_path in ref_paths:
    Path(ref_path).unlink(missing_ok=True)

  # Extract the generated image
  for part in response.parts:
    if part.text is not None:
      print(f"   📝 Model response: {part.text}")
    elif image := part.as_image():
      print("   ✅ Received generated image")
      # Convert to PIL Image
      pil_img = image._pil_image

      # Save generated image to debug dir
      if debug_dir:
        generated_debug_path = debug_dir / "generated.png"
        pil_img.save(generated_debug_path)
        print(f"      ✓ Saved generated: {generated_debug_path}")

      return pil_img

  raise ValueError("No image in Gemini response")


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_nano_banana_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  reference_coords: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  generation_dir: Path | None = None,
) -> dict:
  """
  Run the full nano banana generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Loads reference images from specified coordinates
  5. Calls the Gemini API for generation
  6. Resizes to 1024x1024 and saves the generated quadrants to the database

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    reference_coords: List of (x, y) TL quadrant coordinates for reference tiles
    port: Web server port for rendering (default: 5173)
    prompt: Optional additional prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context
    generation_dir: Optional path to generation directory for saving debug images

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Verify Gemini API key
  if not os.getenv("GEMINI_API_KEY"):
    return {"success": False, "error": "GEMINI_API_KEY not found in environment"}

  # Create helper functions for validation
  def has_generation_in_db(qx: int, qy: int) -> bool:
    gen = get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True
    # For context quadrants, treat them as "generated" if they have a render
    if (qx, qy) in context_set:
      render = get_quadrant_render(conn, qx, qy)
      return render is not None
    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    if (qx, qy) in context_set:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result
  print(f"   📐 Template size: {template_image.size[0]}x{template_image.size[1]}")

  # Load reference images
  update_status("loading", "Loading reference images...")
  reference_images: list[Image.Image] = []

  if not reference_coords:
    print(
      "   ⚠️ No reference coordinates provided, generation may have inconsistent style"
    )
  else:
    print(f"   📷 Loading {len(reference_coords)} reference image(s)...")
    for ref_x, ref_y in reference_coords:
      ref_img = load_reference_tile_image(conn, ref_x, ref_y)
      if ref_img is not None:
        reference_images.append(ref_img)
        print(f"      ✓ Loaded reference from TL ({ref_x}, {ref_y})")
      else:
        print(f"      ⚠️ Could not load reference from TL ({ref_x}, {ref_y})")

  if not reference_images:
    return {
      "success": False,
      "error": "No valid reference images could be loaded",
    }

  # Call Gemini API
  update_status("generating", "Calling Gemini API (this may take a minute)...")
  print("🤖 Calling Gemini nano banana API...")

  # Create debug directory for this generation
  debug_dir = None
  if generation_dir:
    debug_dir = generation_dir / "nano_banana"

  try:
    generated_image = call_gemini_nano_banana(
      template_image=template_image,
      reference_images=reference_images,
      prompt=prompt,
      debug_dir=debug_dir,
    )
  except Exception as e:
    update_status("error", f"Generation failed: {e}")
    return {"success": False, "error": str(e)}

  # Resize to 1024x1024 if needed (to get 512x512 quadrants)
  if generated_image.size != (1024, 1024):
    print(f"   📐 Resizing from {generated_image.size} to (1024, 1024)...")
    generated_image = generated_image.resize((1024, 1024), Image.Resampling.LANCZOS)

  if not save:
    print("   ⏭️ Skipping save (--no-save flag)")
    update_status(
      "complete", f"Generated {len(primary_quadrants)} quadrant(s) (not saved)"
    )
    return {
      "success": True,
      "message": f"Generated {len(primary_quadrants)} quadrant{'s' if len(primary_quadrants) != 1 else ''} (not saved)",
      "quadrants": list(primary_quadrants),
    }

  # Extract quadrants from generated image and save to database
  update_status("saving", "Saving generated quadrants to database...")
  print("💾 Saving generated quadrants to database...")

  # Figure out what quadrants are in the infill region
  all_infill_quadrants = (
    placement.all_infill_quadrants
    if placement.all_infill_quadrants
    else region.overlapping_quadrants()
  )

  # For each infill quadrant, extract pixels from the generated image
  saved_count = 0
  for qx, qy in all_infill_quadrants:
    # Calculate position in the generated image
    quad_world_x = qx * QUADRANT_SIZE
    quad_world_y = qy * QUADRANT_SIZE

    template_x = quad_world_x - placement.world_offset_x
    template_y = quad_world_y - placement.world_offset_y

    # Crop this quadrant from the generated image
    crop_box = (
      template_x,
      template_y,
      template_x + QUADRANT_SIZE,
      template_y + QUADRANT_SIZE,
    )
    quad_img = generated_image.crop(crop_box)
    png_bytes = image_to_png_bytes(quad_img)

    # Only save primary quadrants (not padding)
    if (qx, qy) in primary_quadrants or (qx, qy) in [
      (q[0], q[1]) for q in primary_quadrants
    ]:
      if save_quadrant_generation(conn, config, qx, qy, png_bytes):
        print(f"   ✓ Saved generation for ({qx}, {qy})")
        saved_count += 1
      else:
        print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
    else:
      print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

  update_status("complete", f"Generated {saved_count} quadrant(s)")
  return {
    "success": True,
    "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
    "quadrants": list(primary_quadrants),
  }


# =============================================================================
# Main CLI
# =============================================================================


def generate_tile_nano_banana(
  generation_dir: Path,
  x: int,
  y: int,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

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
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants

  Returns:
    True if generation succeeded, False otherwise
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Parse reference coordinates
    reference_coords: list[tuple[int, int]] = []
    if references:
      for ref_str in references:
        try:
          ref_coord = parse_reference_coords(ref_str)
          reference_coords.append(ref_coord)
        except ValueError as e:
          print(f"   ⚠️ Invalid reference coordinate '{ref_str}': {e}")

    # Determine which quadrants to generate
    if target_position is not None:
      # Generate only the target quadrant, using surrounding context
      tile_x, tile_y = calculate_tile_anchor(x, y, target_position)
      quadrants_to_generate = [(x, y)]

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating quadrant ({x}, {y}) at position '{target_position}'")
      print(f"   Tile anchor: ({tile_x}, {tile_y})")
      print("   Context quadrants: ", end="")
      tile_quads = get_tile_quadrants(tile_x, tile_y)
      for qx, qy in tile_quads:
        is_target = (qx, qy) == (x, y)
        marker = " [TARGET]" if is_target else ""
        print(f"({qx},{qy}){marker}", end=" ")
      print()
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")
    else:
      # Generate all 4 quadrants in the tile (default behavior)
      quadrants_to_generate = get_tile_quadrants(x, y)

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating tile at ({x}, {y})")
      print(f"   Quadrants: {quadrants_to_generate}")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    # Create status callback for progress updates
    def status_callback(status: str, message: str) -> None:
      print(f"   [{status}] {message}")

    # Run generation
    result = run_nano_banana_generation(
      conn=conn,
      config=config,
      selected_quadrants=quadrants_to_generate,
      reference_coords=reference_coords,
      port=port,
      prompt=prompt,
      save=save,
      status_callback=status_callback,
      generation_dir=generation_dir,
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
    description="Generate pixel art for a tile using the Gemini nano banana model."
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
  parser.add_argument(
    "--prompt",
    "-p",
    type=str,
    default=None,
    help="Additional prompt text for generation",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Don't save generated quadrants to the database",
  )
  parser.add_argument(
    "--references",
    "-r",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of reference TL quadrant coordinates in format '(x,y)'. "
      "These will be loaded from the database and used as style references. "
      "Example: --references '(0,0)' '(2,0)' '(0,2)'"
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

    success = generate_tile_nano_banana(
      generation_dir,
      args.x,
      args.y,
      args.port,
      args.target_position,
      args.prompt,
      not args.no_save,
      args.references,
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

Done! The script now saves debug images to a `nano_banana` subdirectory in the generation directory. Here's what gets saved:

| File | Description |
|------|-------------|
| `template.png` | The template image with red border sent to the model |
| `reference_1.png`, `reference_2.png`, ... | All reference images used for style context |
| `prompt.txt` | The full prompt text sent to the model |
| `generated.png` | The generated output image from Gemini |

Example output:
```
   📁 Saving debug images to: /path/to/generations/v01/nano_banana
      ✓ Saved template: /path/to/generations/v01/nano_banana/template.png
      ✓ Saved reference 1: /path/to/generations/v01/nano_banana/reference_1.png
      ✓ Saved reference 2: /path/to/generations/v01/nano_banana/reference_2.png
      ✓ Saved prompt: /path/to/generations/v01/nano_banana/prompt.txt
   ...
      ✓ Saved generated: /path/to/generations/v01/nano_banana/generated.png
```

```plaintext
   📁 Saving debug images to: /path/to/generations/v01/nano_banana
      ✓ Saved template: /path/to/generations/v01/nano_banana/template.png
      ✓ Saved reference 1: /path/to/generations/v01/nano_banana/reference_1.png
      ✓ Saved reference 2: /path/to/generations/v01/nano_banana/reference_2.png
      ✓ Saved prompt: /path/to/generations/v01/nano_banana/prompt.txt
   ...
      ✓ Saved generated: /path/to/generations/v01/nano_banana/generated.png
```

---

**User**

Give me a command to test

---

**Cursor**

Here's a test command using existing data from `generations/v01`:

```bash
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 \
  4 0 \
  --target-position tl \
  --references "(0,0)" "(-2,0)" \
  --no-save \
  --no-start-server
```

This will:
- Generate quadrant `(4, 0)` positioned as top-left in a 2x2 context
- Use existing tiles at `(0,0)` and `(-2,0)` as style references  
- Skip saving to the database (`--no-save`)
- Assume web server is already running (`--no-start-server`)

**Note:** Make sure you have:
1. The web server running on port 5173 (or remove `--no-start-server`)
2. `GEMINI_API_KEY` set in your environment

If you want to start the web server automatically and save results, use:

```bash
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 \
  4 0 \
  --target-position tl \
  --references "(0,0)" "(-2,0)"
```

Debug images will be saved to `generations/v01/nano_banana/`.

```shellscript
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 \
  4 0 \
  --target-position tl \
  --references "(0,0)" "(-2,0)" \
  --no-save \
  --no-start-server
```

```shellscript
uv run python -m isometric_nyc.e2e_generation.generate_tile_nano_banana \
  generations/v01 \
  4 0 \
  --target-position tl \
  --references "(0,0)" "(-2,0)"
```

