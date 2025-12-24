"""
Generate template images for tile infill generation.

This script creates template images for infill generation by:
1. Validating that the selected quadrants form a legal selection
2. Finding optimal placement to maximize context
3. Fetching existing render/generation pixels from the database
4. Rendering new quadrants if needed
5. Creating the template image with red border

Usage:
  uv run python src/isometric_nyc/e2e_generation/generate_template.py <generation_dir> "(x,y),(x,y),..."

Examples:
  # Generate template for quadrants (2,1) and (2,2)
  uv run python src/isometric_nyc/e2e_generation/generate_template.py generations/test "(2,1),(2,2)"

  # Generate template for single quadrant
  uv run python src/isometric_nyc/e2e_generation/generate_template.py generations/test "(1,1)"

  # Debug mode - just validate, don't create template
  uv run python src/isometric_nyc/e2e_generation/generate_template.py generations/test "(0,0),(1,0)" --validate-only
"""

import argparse
import re
import sqlite3
from pathlib import Path

from PIL import Image

from isometric_nyc.e2e_generation.infill_template import (
  TEMPLATE_SIZE,
  InfillRegion,
  TemplateBuilder,
  TemplatePlacement,
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
  save_quadrant_render,
  split_tile_into_quadrants,
  start_web_server,
)

# =============================================================================
# Argument Parsing
# =============================================================================


def parse_quadrant_list(quadrant_str: str) -> list[tuple[int, int]]:
  """
  Parse a quadrant list string into a list of (x, y) tuples.

  Args:
    quadrant_str: String in format "(x,y),(x,y),..." or "(x,y)"

  Returns:
    List of (x, y) tuples

  Raises:
    ValueError: If the string format is invalid
  """
  # Match patterns like (0,1) or (10,20) or (-1,-2)
  pattern = r"\((-?\d+),(-?\d+)\)"
  matches = re.findall(pattern, quadrant_str)

  if not matches:
    raise ValueError(
      f"Invalid quadrant format: '{quadrant_str}'. "
      "Expected format: '(x,y)' or '(x,y),(x,y),...'"
    )

  return [(int(x), int(y)) for x, y in matches]


# =============================================================================
# Database Helpers
# =============================================================================


def has_generation_in_db(conn: sqlite3.Connection) -> callable:
  """Create a function to check if a quadrant has generation in the database."""

  def check(qx: int, qy: int) -> bool:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (qx, qy),
    )
    row = cursor.fetchone()
    return row is not None and row[0] is not None

  return check


def get_render_from_db(
  conn: sqlite3.Connection,
  config: dict,
  port: int,
  force_render: bool = False,
) -> callable:
  """Create a function to get render image from database (rendering if needed)."""

  def get_render(qx: int, qy: int) -> Image.Image | None:
    # Try to get from database first (unless force_render is set)
    if not force_render:
      render_bytes = get_quadrant_render(conn, qx, qy)
      if render_bytes:
        return png_bytes_to_image(render_bytes)

    # Need to render (or forced re-render)
    action = "Re-rendering" if force_render else "Rendering"
    print(f"   📦 {action} quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  return get_render


def get_generation_from_db(conn: sqlite3.Connection) -> callable:
  """Create a function to get generation image from database."""

  def get_gen(qx: int, qy: int) -> Image.Image | None:
    gen_bytes = get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)
    return None

  return get_gen


# =============================================================================
# Rendering
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

  Returns the PNG bytes of the rendered quadrant.
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")
  print(f"      Anchor: {quadrant['lat']:.6f}, {quadrant['lng']:.6f}")

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

  tile_image = render_url_to_image(url, config["width_px"], config["height_px"])

  # Split into quadrants and save all to database
  quadrant_images = split_tile_into_quadrants(tile_image)

  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)

    if save_quadrant_render(conn, config, qx, qy, png_bytes):
      print(f"      ✓ Saved render for quadrant ({qx}, {qy})")

    if dx == 0 and dy == 0:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Template Creation
# =============================================================================


def create_template(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  port: int,
  border_width: int = 2,
  force_render: bool = False,
  allow_expansion: bool = True,
) -> tuple[Image.Image, TemplatePlacement] | None:
  """
  Create a template image for the selected quadrants.

  Args:
    conn: Database connection
    config: Generation config
    selected_quadrants: List of (qx, qy) quadrant positions to infill
    port: Web server port
    border_width: Width of the red border
    force_render: If True, re-render quadrants even if they exist
    allow_expansion: If True, automatically expand infill to cover missing context

  Returns:
    Tuple of (template_image, placement) or None if invalid selection
  """
  # Create helper functions for the builder
  has_gen = has_generation_in_db(conn)
  get_render = get_render_from_db(conn, config, port, force_render)
  get_gen = get_generation_from_db(conn)

  # Validate selection first (with expansion if allowed)
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_gen, allow_expansion=allow_expansion
  )

  if not is_valid:
    print(f"❌ Invalid selection: {msg}")
    return None

  print(f"✅ {msg}")

  # Show expansion info if applicable
  if placement.is_expanded:
    print(f"   📦 Padding quadrants: {placement.padding_quadrants}")
    print("   (These will be filled with render pixels and discarded after generation)")

  # Create the infill region and builder (use expanded region if applicable)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  builder = TemplateBuilder(region, has_gen, get_render, get_gen)

  # Get detailed info for logging
  info = builder.get_validation_info()
  print(f"\n📋 Infill region: {info['region']}")
  print(
    f"   Area: {info['area']} pixels ({info['area'] * 100 // (TEMPLATE_SIZE * TEMPLATE_SIZE)}% of template)"
  )
  print(
    f"   Context: left={info['has_left_gen']}, right={info['has_right_gen']}, "
    f"top={info['has_top_gen']}, bottom={info['has_bottom_gen']}"
  )

  print("\n📋 Template placement:")
  print(f"   Infill position: ({placement.infill_x}, {placement.infill_y})")
  print(f"   World offset: ({placement.world_offset_x}, {placement.world_offset_y})")
  if placement.is_expanded:
    print(f"   Primary quadrants: {placement.primary_quadrants}")
    print(f"   Padding quadrants: {placement.padding_quadrants}")

  # Build the template
  print("\n📋 Building template...")
  result = builder.build(border_width, allow_expansion=allow_expansion)

  if result is None:
    print("❌ Failed to build template")
    return None

  return result


def visualize_grid(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
  padding: int = 2,
) -> str:
  """Create an ASCII visualization of the grid around the selection."""
  has_gen = has_generation_in_db(conn)

  # Find bounds
  min_qx = min(q[0] for q in selected_quadrants)
  max_qx = max(q[0] for q in selected_quadrants)
  min_qy = min(q[1] for q in selected_quadrants)
  max_qy = max(q[1] for q in selected_quadrants)

  # Extend with padding
  view_min_x = min_qx - padding
  view_max_x = max_qx + padding
  view_min_y = min_qy - padding
  view_max_y = max_qy + padding

  selected_set = set(selected_quadrants)
  lines = []

  for qy in range(view_min_y, view_max_y + 1):
    row = []
    for qx in range(view_min_x, view_max_x + 1):
      if (qx, qy) in selected_set:
        row.append("S")
      elif has_gen(qx, qy):
        row.append("G")
      else:
        row.append("x")
    lines.append(" ".join(row))

  return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
  parser = argparse.ArgumentParser(
    description="Generate template images for tile infill generation."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "quadrants",
    type=str,
    help='Quadrants to generate in format "(x,y),(x,y),..."',
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
    "--validate-only",
    action="store_true",
    help="Only validate the selection, don't create template",
  )
  parser.add_argument(
    "--output",
    type=Path,
    help="Output path for template image (default: generation_dir/templates/)",
  )
  parser.add_argument(
    "--border-width",
    type=int,
    default=2,
    help="Width of the red border in pixels (default: 2)",
  )
  parser.add_argument(
    "--force-render",
    action="store_true",
    help="Force re-rendering of quadrants even if they already exist in the database",
  )
  parser.add_argument(
    "--no-expansion",
    action="store_true",
    help="Don't automatically expand infill to cover missing context quadrants",
  )

  args = parser.parse_args()

  # Parse quadrant list
  try:
    selected_quadrants = parse_quadrant_list(args.quadrants)
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1

  print(f"\n{'=' * 60}")
  print("🎯 Template Generation")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {args.generation_dir}")
  print(f"   Selected quadrants: {selected_quadrants}")
  if args.force_render:
    print("   ⚠️  Force render: ON (will re-render existing quadrants)")

  # Resolve paths
  generation_dir = args.generation_dir.resolve()
  db_path = generation_dir / "quadrants.db"

  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  conn = sqlite3.connect(db_path)
  web_server = None

  try:
    config = get_generation_config(conn)

    # Show grid visualization
    print("\n📋 Current grid state:")
    print(visualize_grid(conn, selected_quadrants))

    # Validate using the new module
    allow_expansion = not args.no_expansion
    has_gen = has_generation_in_db(conn)
    is_valid, msg, placement = validate_quadrant_selection(
      selected_quadrants, has_gen, allow_expansion=allow_expansion
    )

    if not is_valid:
      print(f"\n❌ Invalid selection: {msg}")
      return 1

    print(f"\n✅ {msg}")

    # Show expansion info if applicable
    if placement and placement.is_expanded:
      print(f"   📦 Padding quadrants: {placement.padding_quadrants}")
      print(
        "   (These will be filled with render pixels and discarded after generation)"
      )

    if args.validate_only:
      print("\n🔍 Validation only mode - not creating template")

      # Show what placement would be
      if placement:
        print("\n📋 Optimal placement:")
        print(f"   Infill position: ({placement.infill_x}, {placement.infill_y})")
        print(
          f"   World offset: ({placement.world_offset_x}, {placement.world_offset_y})"
        )
        if placement.is_expanded:
          print(f"   Primary quadrants: {placement.primary_quadrants}")
          print(f"   Padding quadrants: {placement.padding_quadrants}")

      return 0

    # Start web server if needed
    if not args.no_start_server:
      web_server = start_web_server(WEB_DIR, args.port)

    # Create template
    result = create_template(
      conn,
      config,
      selected_quadrants,
      args.port,
      args.border_width,
      args.force_render,
      allow_expansion,
    )

    if result is None:
      return 1

    template, placement = result

    # Determine output path
    if args.output:
      output_path = args.output.resolve()
    else:
      templates_dir = generation_dir / "templates"
      templates_dir.mkdir(exist_ok=True)
      # Create filename from quadrant positions
      pos_str = "_".join(f"{x}_{y}" for x, y in selected_quadrants)
      output_path = templates_dir / f"template_{pos_str}.png"

    # Save template
    template.save(output_path)

    print(f"\n{'=' * 60}")
    print("✅ Template created successfully!")
    print(f"   Output: {output_path}")
    print(f"   Size: {template.size}")
    print(
      f"   Infill bounds: ({placement.infill_x}, {placement.infill_y}) to "
      f"({placement.infill_x + InfillRegion.from_quadrants(selected_quadrants).width}, "
      f"{placement.infill_y + InfillRegion.from_quadrants(selected_quadrants).height})"
    )
    print(f"{'=' * 60}")

    return 0

  except Exception as e:
    print(f"❌ Error: {e}")
    raise

  finally:
    conn.close()
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


if __name__ == "__main__":
  exit(main())
