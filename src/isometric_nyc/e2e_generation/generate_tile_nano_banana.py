"""
Generate pixel art for a tile using the Gemini nano banana model.

This script generates pixel art for quadrants at specific coordinates.
It uses the same template construction and pixel splicing as generate_tile_omni.py,
but uses the Gemini nano banana generation model instead of Oxen.

Supports two modes:
1. Arbitrary quadrant selection (preferred): Generate any combination of quadrants
   with context automatically calculated from adjacent generated quadrants.
2. Legacy 2x2 tile mode: Generate a 2x2 tile at specified coordinates.

Usage:
  # Generate a 2x2 tile anchored at (0,0):
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0

  # Generate arbitrary quadrants (context auto-calculated) - comma-separated:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> \\
    --quadrants "(0,0),(1,0),(0,1)"

  # Generate arbitrary quadrants (context auto-calculated) - multiple args:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> \\
    --quadrants "(0,0)" "(1,0)" "(0,1)"

  # Generate quadrants with explicit context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> \\
    --quadrants "(2,2)" --context "(1,2)" "(2,1)" "(1,1)"

  # Generate a single quadrant with target position (legacy 2x2 context):
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 -t br

Reference images can be specified with --references to provide style context:
  uv run python src/isometric_nyc/e2e_generation/generate_tile_nano_banana.py <gen_dir> 0 0 \\
    --references "(0,0)" "(1,0)" "(0,1)"
"""

import argparse
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from isometric_nyc.e2e_generation.infill_template import (
  QUADRANT_SIZE,
  InfillRegion,
  TemplateBuilder,
  validate_quadrant_selection,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  build_tile_render_url,
  ensure_quadrant_exists,
  get_generation_config,
  get_quadrant_generation,
  get_quadrant_render,
  image_to_png_bytes,
  png_bytes_to_image,
  render_url_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()

# =============================================================================
# Reference Image Upload Cache
# =============================================================================

# Cache for uploaded reference images to Gemini API
# Maps (tl_x, tl_y) -> uploaded file reference
# This avoids re-uploading the same reference images on every generation
_REFERENCE_UPLOAD_CACHE: dict[tuple[int, int], any] = {}


def clear_reference_cache():
  """Clear the reference upload cache. Call this if references are regenerated."""
  global _REFERENCE_UPLOAD_CACHE
  _REFERENCE_UPLOAD_CACHE.clear()
  print("   🗑️ Cleared reference upload cache")


def get_cache_stats() -> dict:
  """Get statistics about the reference upload cache."""
  return {
    "cached_references": len(_REFERENCE_UPLOAD_CACHE),
    "coordinates": list(_REFERENCE_UPLOAD_CACHE.keys()),
  }


# =============================================================================
# Context Quadrant Calculation (from app.py)
# =============================================================================


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


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
# Quadrant Parsing Utilities
# =============================================================================


def parse_quadrant_coords(coord_str: str) -> tuple[int, int]:
  """
  Parse a quadrant coordinate string like "(0,1)" or "0,1" into a tuple.

  Args:
    coord_str: String in format "(x,y)" or "x,y"

  Returns:
    Tuple of (x, y) coordinates

  Raises:
    ValueError: If the format is invalid
  """
  coord_str = coord_str.strip()
  # Remove optional parentheses
  if coord_str.startswith("(") and coord_str.endswith(")"):
    coord_str = coord_str[1:-1]
  parts = coord_str.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid coordinate format: {coord_str}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def parse_quadrant_list(quadrants_str: str) -> list[tuple[int, int]]:
  """
  Parse a list of quadrant coordinates.

  Supports formats:
  - "(0,0),(1,0),(0,1)"
  - "(0,0) (1,0) (0,1)"
  - "0,0 1,0 0,1"

  Args:
    quadrants_str: String containing quadrant coordinates

  Returns:
    List of (x, y) coordinate tuples

  Raises:
    ValueError: If the format is invalid
  """
  # Try to find all (x,y) patterns
  pattern = r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?"
  matches = re.findall(pattern, quadrants_str)
  if not matches:
    raise ValueError(f"No valid quadrant coordinates found in: {quadrants_str}")
  return [(int(x), int(y)) for x, y in matches]


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

  # Build URL and render using shared utilities
  url = build_tile_render_url(
    port=port,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    azimuth=config["camera_azimuth_degrees"],
    elevation=config["camera_elevation_degrees"],
    view_height=config.get("view_height_meters", 200),
  )

  full_tile = render_url_to_image(url, config["width_px"], config["height_px"])
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


def create_few_shot_template(
  reference_image: Image.Image,
  reference_tl: tuple[int, int],
  infill_region,
  conn: sqlite3.Connection,
  config: dict,
  port: int,
) -> Image.Image | None:
  """
  Create a few-shot "before" template from a reference tile.

  This recreates what the template would have looked like before generation,
  using the same infill region position as the current generation.

  The strategy is to take the infill region's RELATIVE position within a 2x2 tile
  and apply that same relative region to the reference tile.

  Args:
    reference_image: The reference tile (1024x1024 pixel art)
    reference_tl: Top-left coordinates of the reference tile (qx, qy)
    infill_region: The infill region from the current generation
    conn: Database connection
    config: Generation config
    port: Web server port for rendering

  Returns:
    PIL Image showing the reference tile with infill region replaced by renders + red border,
    or None if renders couldn't be created
  """
  from PIL import ImageDraw

  # Start with a copy of the reference image
  template = reference_image.copy()

  # Get overlapping quadrants from the infill region
  # These are in absolute world coordinates
  overlapping_quads = infill_region.overlapping_quadrants()

  ref_x, ref_y = reference_tl

  # For the few-shot example, we want to show the same RELATIVE quadrants
  # So if the infill region covers quadrants (5,6) and (6,6),
  # we want to show quadrants (ref_x, ref_y) and (ref_x+1, ref_y) in the reference
  # We need to translate the overlapping quadrants to be relative to (0,0) first
  if not overlapping_quads:
    return None

  # Find the bounds of the overlapping quadrants
  min_qx = min(q[0] for q in overlapping_quads)
  max_qx = max(q[0] for q in overlapping_quads)
  min_qy = min(q[1] for q in overlapping_quads)
  max_qy = max(q[1] for q in overlapping_quads)

  print(f"         📐 Infill region quadrants: {overlapping_quads}")
  print(f"         📐 Infill bounds: ({min_qx}, {min_qy}) to ({max_qx}, {max_qy})")
  print(f"         📐 Reference TL: ({ref_x}, {ref_y})")

  # Calculate the infill region in template-relative coordinates (0-1024)
  # Since the reference is always a 2x2 tile (1024x1024), we translate the region
  infill_rel_x = (min_qx - min_qx) * QUADRANT_SIZE  # Always 0 for the minimum
  infill_rel_y = (min_qy - min_qy) * QUADRANT_SIZE  # Always 0 for the minimum
  infill_width = (max_qx - min_qx + 1) * QUADRANT_SIZE
  infill_height = (max_qy - min_qy + 1) * QUADRANT_SIZE

  print(f"         📐 Template-relative infill: ({infill_rel_x}, {infill_rel_y}) size {infill_width}x{infill_height}px")

  # For each quadrant in the relative region, render it from the reference tile position
  print(f"         🎨 Rendering quadrants for few-shot template:")
  for rel_qx in range(max_qx - min_qx + 1):
    for rel_qy in range(max_qy - min_qy + 1):
      # Calculate the actual quadrant coordinates in the reference tile
      actual_qx = ref_x + rel_qx
      actual_qy = ref_y + rel_qy

      # Render this quadrant
      render_bytes = render_quadrant(conn, config, actual_qx, actual_qy, port)
      if render_bytes is None:
        print(f"            ⚠️ Could not render ({actual_qx}, {actual_qy}) for few-shot template")
        return None

      render_img = png_bytes_to_image(render_bytes)

      # Paste into template at the correct relative position
      paste_x = rel_qx * QUADRANT_SIZE
      paste_y = rel_qy * QUADRANT_SIZE
      template.paste(render_img, (paste_x, paste_y))
      print(f"            ✓ Rendered ({actual_qx}, {actual_qy}) → pasted at ({paste_x}, {paste_y})px")

  # Draw red border around the infill region (in template-relative coordinates)
  print(f"         🟥 Drawing red border at ({infill_rel_x}, {infill_rel_y}) size {infill_width}x{infill_height}px")
  draw = ImageDraw.Draw(template)
  border_width = 2

  # Draw rectangle border
  for i in range(border_width):
    draw.rectangle(
      [
        infill_rel_x + i,
        infill_rel_y + i,
        infill_rel_x + infill_width - 1 - i,
        infill_rel_y + infill_height - 1 - i,
      ],
      outline=(255, 0, 0),
      width=1,
    )

  print(f"         ✅ Few-shot template created: {template.size[0]}x{template.size[1]}px, mode={template.mode}")

  return template


def call_gemini_nano_banana(
  template_image: Image.Image,
  reference_images: list[Image.Image],
  reference_coords: list[tuple[int, int]] | None = None,
  infill_region=None,
  conn: sqlite3.Connection | None = None,
  config: dict | None = None,
  port: int | None = None,
  prompt: str | None = None,
  negative_prompt: str | None = None,
  debug_dir: Path | None = None,
) -> Image.Image:
  """
  Call the Gemini API to generate pixel art using the nano banana model.

  Args:
    template_image: The template image with render pixels and red border
    reference_images: List of reference images for style context
    reference_coords: Optional list of (x, y) TL coordinates for caching uploaded references
    infill_region: Optional infill region for creating few-shot examples
    conn: Optional database connection for few-shot examples
    config: Optional generation config for few-shot examples
    port: Optional web server port for few-shot examples
    prompt: Optional custom prompt text (overrides the default prompt)
    negative_prompt: Optional negative prompt text for generation
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

  # Simplified approach: Upload just ONE reference image (no few-shot)
  # Use the first reference image only
  print(f"\n   📷 Using simplified approach: template + 1 reference with LABELS")

  # Add visual labels to the template image
  from PIL import ImageDraw, ImageFont

  template_labeled = template_image.copy()
  draw = ImageDraw.Draw(template_labeled)

  # Try to use a larger font, fall back to default if not available
  try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
  except:
    font = ImageFont.load_default()

  # Add label at top of image
  label_text = "TRANSFORM THIS AREA ➜"
  bbox = draw.textbbox((0, 0), label_text, font=font)
  text_width = bbox[2] - bbox[0]
  text_height = bbox[3] - bbox[1]

  # Position at top center with black background
  x = (template_labeled.width - text_width) // 2
  y = 20

  # Draw black background rectangle
  padding = 10
  draw.rectangle(
    [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
    fill=(0, 0, 0)
  )

  # Draw text in bright yellow
  draw.text((x, y), label_text, fill=(255, 255, 0), font=font)

  print(f"      ✓ Added label to template: '{label_text}'")

  # Save labeled template to debug dir
  if debug_dir:
    template_labeled_path = debug_dir / "template_labeled.png"
    template_labeled.save(template_labeled_path)
    print(f"      ✓ Saved labeled template: {template_labeled_path}")

  # Re-upload the labeled template
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    labeled_template_path = tmp.name
    template_labeled.save(labeled_template_path)

  try:
    # Remove the original unlabeled template from contents
    contents.pop()
    # Upload the labeled version
    print(f"      ⬆️ Uploading labeled template to Gemini API...")
    template_labeled_ref = client.files.upload(file=labeled_template_path)
    contents.append(template_labeled_ref)
  finally:
    Path(labeled_template_path).unlink(missing_ok=True)

  if reference_images:
    ref_img = reference_images[0]
    ref_coord = reference_coords[0] if reference_coords else None

    # Add visual labels to the reference image
    ref_labeled = ref_img.copy()
    draw_ref = ImageDraw.Draw(ref_labeled)

    # Add label at top
    ref_label_text = "TARGET STYLE ⭐"
    bbox = draw_ref.textbbox((0, 0), ref_label_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (ref_labeled.width - text_width) // 2
    y = 20

    # Draw black background
    draw_ref.rectangle(
      [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
      fill=(0, 0, 0)
    )

    # Draw text in bright cyan
    draw_ref.text((x, y), ref_label_text, fill=(0, 255, 255), font=font)

    print(f"      ✓ Added label to reference: '{ref_label_text}'")

    # Save labeled reference to debug dir
    if debug_dir:
      ref_labeled_path = debug_dir / f"reference_labeled.png"
      ref_labeled.save(ref_labeled_path)
      print(f"      ✓ Saved labeled reference: {ref_labeled_path}")

    # Upload the labeled reference
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      ref_tmp_path = tmp.name
      ref_labeled.save(ref_tmp_path)

    try:
      print(f"      ⬆️ Uploading labeled reference to Gemini API...")
      ref_ref = client.files.upload(file=ref_tmp_path)
      contents.append(ref_ref)
      print(f"      ✅ Labeled reference uploaded")
    finally:
      Path(ref_tmp_path).unlink(missing_ok=True)

  # Build the prompt - custom prompt overrides default
  if prompt:
    # Use custom prompt entirely (override default)
    generation_prompt = prompt
    print(f"   📝 Using custom prompt (overriding default)")
  else:
    # Use default prompt
    generation_prompt = f"""
PIXEL ART STYLE TRANSFER TASK

**Images provided (with visual labels burned in):**
- Image A: Labeled "TRANSFORM THIS AREA ➜" - Template with 3D render inside red border
- Image B: Labeled "TARGET STYLE ⭐" - Pixel art reference showing the style to match

**YOUR TASK:**
Fill in the red-bordered area in Image A with NEW pixel art that:
1. Shows the SAME CONTENT as the 3D render (buildings, roads, trees in same positions)
2. Uses the ARTISTIC STYLE from Image B (pixel art aesthetic, colors, simplification)

**CRITICAL RULES:**
❌ DO NOT copy Image A as your output
❌ DO NOT copy Image B as your output
❌ DO NOT return any input image unchanged
✅ You MUST CREATE something NEW by transforming the render

**How to complete this task:**

STEP 1: Study Image B (TARGET STYLE ⭐)
- Memorize the pixel art technique: color choices, shading style, detail level
- Notice how buildings are simplified, how roads are drawn, how trees look
- This is your STYLE GUIDE - you will draw in this exact same style

STEP 2: Study Image A (TRANSFORM THIS AREA ➜)
- Look at the 3D render inside the red border
- Note what objects are there and where they're positioned
- This tells you WHAT to draw and WHERE - but NOT how to draw it

STEP 3: Create the output
- Draw the objects from the 3D render (the WHAT and WHERE)
- But draw them using the pixel art style from Image B (the HOW)
- Example: If the render shows a building, draw a pixelated building like ones in Image B
- Example: If the render shows a road, draw a pixelated road like roads in Image B

**Style requirements (copy from Image B):**
- Use Image B's color palette
- Use Image B's shading technique
- Use Image B's level of detail
- Use Image B's pixel grid aesthetic
- Use Image B's geometric simplification

**DO NOT include the labels in your output!**
The yellow "TRANSFORM THIS AREA" and cyan "TARGET STYLE" labels are just for your understanding.
Your output should have NO TEXT LABELS.

**OUTPUT REQUIREMENTS:**
- Full 1024x1024 pixel image
- Red-bordered area filled with NEWLY CREATED pixel art
- Pixel art matches Image B's style but shows Image A's content
- Remove red border and any text labels from output
- Areas outside red border preserved unchanged

Think of this like: "Redraw the 3D render as pixel art in the style of Image B"
""".strip()

  # Append negative prompt if provided
  if negative_prompt:
    generation_prompt += f"\n\n**NEGATIVE PROMPT (Things to AVOID):**\n{negative_prompt}"
    print(f"   🚫 Added negative prompt to generation prompt")

  contents.append(generation_prompt)

  # Save prompt to debug dir
  if debug_dir:
    prompt_debug_path = debug_dir / "prompt.txt"
    with open(prompt_debug_path, "w") as f:
      f.write(generation_prompt)
    print(f"      ✓ Saved prompt: {prompt_debug_path}")

  # Log the prompt and images before sending
  print("\n" + "=" * 60)
  print("📤 GEMINI API REQUEST (WITH VISUAL LABELS)")
  print("=" * 60)
  print(f"   📐 Image A: 'TRANSFORM THIS AREA ➜' (yellow label)")
  if reference_images:
    print(f"   📐 Image B: 'TARGET STYLE ⭐' (cyan label)")
  print(f"   📊 Total images: {len([c for c in contents if not isinstance(c, str)])}")
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
  negative_prompt: str | None = None,
  save: bool = True,
  status_callback: Callable[[str, str], None] | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  generation_dir: Path | None = None,
  model_config: "ModelConfig | None" = None,  # noqa: F821
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
    prompt: Optional custom prompt text (overrides the default prompt) for generation
    negative_prompt: Optional negative prompt text for generation
    save: Whether to save generated quadrants to database (default: True)
    status_callback: Optional callback(status, message) for progress updates
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as context
    generation_dir: Optional path to generation directory for saving debug images
    model_config: Optional model configuration for preprocessing parameters

  Returns:
    Dict with:
      - success: bool
      - message: str (on success)
      - error: str (on failure)
      - quadrants: list of generated quadrant coords (on success)
  """
  # Calculate context lazily if not explicitly provided
  # This ensures we use the most up-to-date context based on what's
  # actually generated at execution time
  if context_quadrants:
    print(
      f"   📋 Using explicit context: {len(context_quadrants)} quadrant(s): {context_quadrants}"
    )
  else:
    # Calculate context lazily based on current generation state
    context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
    if context_quadrants:
      print(
        f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s): {context_quadrants}"
      )
    else:
      print(
        "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
      )

  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )

  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  if negative_prompt:
    print(f"   🚫 Negative prompt: {negative_prompt}")

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
    region,
    has_generation_in_db,
    get_render_from_db_with_render,
    get_generation_from_db,
    model_config=model_config,
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

  # Show cache stats
  cache_stats = get_cache_stats()
  if cache_stats["cached_references"] > 0:
    print(f"   💾 Upload cache: {cache_stats['cached_references']} reference(s) cached")

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
      reference_coords=reference_coords,
      infill_region=region,
      conn=conn,
      config=config,
      port=port,
      prompt=prompt,
      negative_prompt=negative_prompt,
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
  x: int | None = None,
  y: int | None = None,
  port: int = DEFAULT_WEB_PORT,
  target_position: str | None = None,
  prompt: str | None = None,
  save: bool = True,
  references: list[str] | None = None,
  quadrants: list[tuple[int, int]] | None = None,
  context: list[tuple[int, int]] | None = None,
) -> bool:
  """
  Generate pixel art for quadrant(s) using the Gemini nano banana model.

  This function supports two modes:
  1. Arbitrary quadrant selection (preferred): Pass `quadrants` directly
  2. Legacy 2x2 tile mode: Pass `x`, `y`, and optionally `target_position`

  Args:
    generation_dir: Path to the generation directory
    x: Quadrant x coordinate (or target x if target_position is specified).
       Ignored if `quadrants` is provided.
    y: Quadrant y coordinate (or target y if target_position is specified).
       Ignored if `quadrants` is provided.
    port: Web server port for rendering
    target_position: Position of target quadrant within the 2x2 tile.
      One of "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right).
      Only used when `quadrants` is not provided.
    prompt: Additional prompt text for generation
    save: Whether to save generated quadrants to database
    references: List of reference coordinate strings like "(0,0)" for TL quadrants
    quadrants: List of (x, y) quadrant coordinates to generate.
      When provided, ignores x, y, and target_position.
      Context quadrants are automatically calculated from adjacent generations.
    context: Optional list of (x, y) context quadrant coordinates.
      If provided, these quadrants will provide surrounding pixel art context.
      If None, context is automatically calculated.

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
    if quadrants is not None:
      # Use directly provided quadrants (arbitrary selection mode)
      quadrants_to_generate = quadrants

      print(f"\n{'=' * 60}")
      print(f"🍌 Generating {len(quadrants_to_generate)} quadrant(s)")
      print(f"   Quadrants: {quadrants_to_generate}")
      if context:
        print(f"   Explicit context: {context}")
      else:
        print("   Context: will be calculated automatically")
      print(f"   References: {reference_coords}")
      print(f"{'=' * 60}")

    elif x is not None and y is not None:
      # Legacy 2x2 tile mode
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
    else:
      raise ValueError("Either 'quadrants' or both 'x' and 'y' must be provided")

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
      context_quadrants=context,  # Pass explicit context or None for auto-calculation
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
    description="Generate pixel art for a tile using the Gemini nano banana model.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Generate a 2x2 tile anchored at (0,0):
  %(prog)s generations/v01 0 0

  # Generate a single quadrant (0,0) with context from adjacent generated quadrants:
  %(prog)s generations/v01 0 0 -t br

  # Generate arbitrary quadrants (context auto-calculated) - comma-separated:
  %(prog)s generations/v01 --quadrants "(0,0),(1,0),(0,1)"

  # Generate arbitrary quadrants (context auto-calculated) - multiple args:
  %(prog)s generations/v01 --quadrants "(0,0)" "(1,0)" "(0,1)"

  # Generate quadrants with explicit context:
  %(prog)s generations/v01 --quadrants "(2,2)" --context "(1,2)" "(2,1)" "(1,1)"
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
    help="Quadrant x coordinate (tile anchor x, or target x if --target-position is used). Ignored if --quadrants is provided.",
  )
  parser.add_argument(
    "y",
    type=int,
    nargs="?",
    default=None,
    help="Quadrant y coordinate (tile anchor y, or target y if --target-position is used). Ignored if --quadrants is provided.",
  )
  parser.add_argument(
    "--quadrants",
    "-q",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of quadrant coordinates to generate. Accepts multiple formats: "
      "1) Single string: --quadrants '(0,0),(1,0),(0,1)' "
      "2) Multiple args: --quadrants '(0,0)' '(1,0)' '(0,1)'. "
      "When provided, ignores x, y, and --target-position. "
      "Context quadrants are automatically calculated from adjacent generations."
    ),
  )
  parser.add_argument(
    "--context",
    "-c",
    nargs="+",
    type=str,
    default=None,
    help=(
      "List of context quadrant coordinates. Accepts multiple formats: "
      "1) Single string: --context '(1,2),(2,1)' "
      "2) Multiple args: --context '(1,2)' '(2,1)'. "
      "These quadrants will be used as surrounding context for generation. "
      "If not provided, context is automatically calculated."
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
    "--target-position",
    "-t",
    choices=["tl", "tr", "bl", "br"],
    help=(
      "Position of target quadrant (x,y) within the 2x2 tile context. "
      "tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right. "
      "If specified, (x,y) is the target to generate and surrounding quadrants "
      "provide context. If not specified, (x,y) is the tile anchor and all "
      "4 quadrants are generated. Ignored if --quadrants is provided."
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

  # Parse quadrants if provided (supports multiple args or comma-separated)
  quadrants_list = None
  if args.quadrants:
    try:
      quadrants_list = []
      for quad_str in args.quadrants:
        parsed = parse_quadrant_list(quad_str)
        quadrants_list.extend(parsed)
      # Remove duplicates while preserving order
      seen = set()
      unique_quadrants = []
      for q in quadrants_list:
        if q not in seen:
          seen.add(q)
          unique_quadrants.append(q)
      quadrants_list = unique_quadrants
      print(f"📋 Parsed quadrants: {quadrants_list}")
    except ValueError as e:
      print(f"❌ Error parsing --quadrants: {e}")
      return 1

  # Parse context if provided (supports multiple args or comma-separated)
  context_list = None
  if args.context:
    try:
      context_list = []
      for ctx_str in args.context:
        parsed = parse_quadrant_list(ctx_str)
        context_list.extend(parsed)
      # Remove duplicates while preserving order
      seen = set()
      unique_context = []
      for c in context_list:
        if c not in seen:
          seen.add(c)
          unique_context.append(c)
      context_list = unique_context
      print(f"📋 Parsed context: {context_list}")
    except ValueError as e:
      print(f"❌ Error parsing --context: {e}")
      return 1

  # Validate that either quadrants or x,y are provided
  if quadrants_list is None and (args.x is None or args.y is None):
    print("❌ Error: Either --quadrants or both x and y coordinates must be provided")
    parser.print_help()
    return 1

  web_server = None

  try:
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    success = generate_tile_nano_banana(
      generation_dir,
      x=args.x,
      y=args.y,
      port=args.port,
      target_position=args.target_position,
      prompt=args.prompt,
      save=not args.no_save,
      references=args.references,
      quadrants=quadrants_list,
      context=context_list,
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
