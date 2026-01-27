"""
Template generation library for seamless tile infilling.

This module formalizes the rules for generating tiles with quadrant overlap,
ensuring no "seams" appear between generated regions. It provides utilities
to create template images and extract generated pixel data.

Key concepts:
- A tile is a 2x2 grid of quadrants
- Quadrants can be: Generated (G), Selected (S for infill), or Empty (x)
- Selected quadrants must form a contiguous region that can be generated
  without creating seams with adjacent generated quadrants

Usage:
  from isometric_nyc.generation.generate_template import (
      QuadrantGrid,
      QuadrantState,
      create_template_image,
      extract_generated_quadrants,
  )

  # Create a grid state
  grid = QuadrantGrid(width=6, height=4)
  grid.set_generated([(0, 1), (1, 1), (0, 2), (1, 2)])
  grid.set_selected([(2, 1), (2, 2)])

  # Validate the selection
  if grid.validate_selection():
      template = create_template_image(...)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from PIL import Image, ImageDraw


class QuadrantState(Enum):
  """State of a quadrant in the grid."""

  EMPTY = "x"  # Not yet generated
  GENERATED = "G"  # Already has generation
  SELECTED = "S"  # Selected for current generation


@dataclass(frozen=True)
class QuadrantPosition:
  """A position in the quadrant grid."""

  x: int
  y: int

  def __iter__(self):
    return iter((self.x, self.y))

  def neighbors(self) -> list["QuadrantPosition"]:
    """Get the 4-connected neighbors of this position."""
    return [
      QuadrantPosition(self.x - 1, self.y),
      QuadrantPosition(self.x + 1, self.y),
      QuadrantPosition(self.x, self.y - 1),
      QuadrantPosition(self.x, self.y + 1),
    ]


@dataclass
class BoundingBox:
  """A bounding box in pixel coordinates."""

  left: int
  top: int
  right: int
  bottom: int

  @property
  def width(self) -> int:
    return self.right - self.left

  @property
  def height(self) -> int:
    return self.bottom - self.top

  @property
  def area(self) -> int:
    return self.width * self.height

  def as_tuple(self) -> tuple[int, int, int, int]:
    return (self.left, self.top, self.right, self.bottom)


# =============================================================================
# Generic Infill Region Positioning
# =============================================================================


@dataclass
class InfillContext:
  """
  Describes the context (generated pixels) available around an infill region.

  Each boolean indicates whether there are generated pixels in that direction
  BEYOND the current template boundary.
  """

  has_left: bool = False
  has_right: bool = False
  has_top: bool = False
  has_bottom: bool = False

  @property
  def count(self) -> int:
    """Number of directions with generated context."""
    return sum([self.has_left, self.has_right, self.has_top, self.has_bottom])


class InfillPositioner:
  """
  Handles positioning and validation of arbitrary rectangular infill regions.

  Rules:
  1. Infill area must be ≤ 50% of template area
  2. If infill edge touches template boundary, pixels beyond must not be generated
  3. Position infill to maximize included context from generated neighbors
  """

  def __init__(self, template_size: int = 1024):
    self.template_size = template_size
    self.max_area = (template_size * template_size) // 2

  def find_optimal_position(
    self,
    infill_width: int,
    infill_height: int,
    context: InfillContext,
  ) -> tuple[int, int] | None:
    """
    Find the optimal (x, y) position for the infill in the template.

    The position maximizes context inclusion while avoiding seams.
    Returns None if no valid position exists.

    Args:
      infill_width: Width of infill region in pixels
      infill_height: Height of infill region in pixels
      context: Which directions have generated pixels beyond template

    Returns:
      (x, y) position for top-left of infill, or None if invalid
    """
    # Check area constraint (only when we have neighbors requiring context)
    # For fresh tiles with no neighbors, any size up to template is allowed
    if context.count > 0 and infill_width * infill_height > self.max_area:
      return None

    # Calculate available margins
    margin_h = self.template_size - infill_width
    margin_v = self.template_size - infill_height

    # Find valid x position
    x = self._find_axis_position(margin_h, context.has_left, context.has_right)
    if x is None:
      return None

    # Find valid y position
    y = self._find_axis_position(margin_v, context.has_top, context.has_bottom)
    if y is None:
      return None

    return (x, y)

  def _find_axis_position(
    self,
    margin: int,
    has_near_gen: bool,
    has_far_gen: bool,
  ) -> int | None:
    """
    Find position along one axis.

    Args:
      margin: Available margin (template_size - infill_size)
      has_near_gen: Generated pixels at near edge (left/top)
      has_far_gen: Generated pixels at far edge (right/bottom)

    Returns:
      Position value, or None if invalid
    """
    if has_near_gen and has_far_gen:
      # Need margin on both sides - must not touch either edge
      if margin < 2:
        # Not enough room for margins on both sides
        return None
      # Center to maximize context from both sides
      return margin // 2

    elif has_near_gen:
      # Need margin on near side, position as far as possible
      # But if margin is 0, we can't avoid touching the near edge → invalid
      if margin == 0:
        return None
      return margin

    elif has_far_gen:
      # Need margin on far side, position at near edge
      # But if margin is 0, we can't avoid touching the far edge → invalid
      if margin == 0:
        return None
      return 0

    else:
      # No constraint - center for balanced context (if any becomes available)
      return margin // 2

  def validate_position(
    self,
    x: int,
    y: int,
    infill_width: int,
    infill_height: int,
    context: InfillContext,
  ) -> tuple[bool, str]:
    """
    Validate that a position doesn't create seams.

    Args:
      x: Left edge of infill in template
      y: Top edge of infill in template
      infill_width: Width of infill region
      infill_height: Height of infill region
      context: Which directions have generated pixels

    Returns:
      (is_valid, error_message)
    """
    # Check area constraint
    if infill_width * infill_height > self.max_area:
      return (
        False,
        f"Infill area exceeds 50% of template ({infill_width * infill_height} > {self.max_area})",
      )

    # Check left edge
    if x == 0 and context.has_left:
      return (
        False,
        "Infill at left edge with generated pixels beyond (would create seam)",
      )

    # Check right edge
    if x + infill_width == self.template_size and context.has_right:
      return (
        False,
        "Infill at right edge with generated pixels beyond (would create seam)",
      )

    # Check top edge
    if y == 0 and context.has_top:
      return (
        False,
        "Infill at top edge with generated pixels beyond (would create seam)",
      )

    # Check bottom edge
    if y + infill_height == self.template_size and context.has_bottom:
      return (
        False,
        "Infill at bottom edge with generated pixels beyond (would create seam)",
      )

    return True, "Valid position"

  def get_infill_bounds(
    self,
    x: int,
    y: int,
    infill_width: int,
    infill_height: int,
  ) -> BoundingBox:
    """Get the bounding box for the infill at the given position."""
    return BoundingBox(
      left=x,
      top=y,
      right=x + infill_width,
      bottom=y + infill_height,
    )


def quadrant_selection_to_infill(
  selected_positions: list[QuadrantPosition],
  quadrant_size: int = 512,
) -> tuple[int, int]:
  """
  Convert a quadrant selection to infill dimensions.

  Args:
    selected_positions: List of selected quadrant positions
    quadrant_size: Size of each quadrant in pixels (default 512)

  Returns:
    (infill_width, infill_height) in pixels
  """
  if not selected_positions:
    raise ValueError("No quadrants selected")

  min_x = min(p.x for p in selected_positions)
  max_x = max(p.x for p in selected_positions)
  min_y = min(p.y for p in selected_positions)
  max_y = max(p.y for p in selected_positions)

  width_quadrants = max_x - min_x + 1
  height_quadrants = max_y - min_y + 1

  return (width_quadrants * quadrant_size, height_quadrants * quadrant_size)


def get_context_from_grid(
  grid: "QuadrantGrid",
  selected_positions: list[QuadrantPosition],
) -> InfillContext:
  """
  Determine the infill context from a quadrant grid.

  Checks for generated quadrants adjacent to the selection.

  Args:
    grid: The quadrant grid
    selected_positions: List of selected positions

  Returns:
    InfillContext describing available generated context
  """
  min_x = min(p.x for p in selected_positions)
  max_x = max(p.x for p in selected_positions)
  min_y = min(p.y for p in selected_positions)
  max_y = max(p.y for p in selected_positions)

  # Check for generated neighbors on each edge
  has_left = any(
    grid.get_state(min_x - 1, y) == QuadrantState.GENERATED
    for y in range(min_y, max_y + 1)
  )
  has_right = any(
    grid.get_state(max_x + 1, y) == QuadrantState.GENERATED
    for y in range(min_y, max_y + 1)
  )
  has_top = any(
    grid.get_state(x, min_y - 1) == QuadrantState.GENERATED
    for x in range(min_x, max_x + 1)
  )
  has_bottom = any(
    grid.get_state(x, max_y + 1) == QuadrantState.GENERATED
    for x in range(min_x, max_x + 1)
  )

  return InfillContext(
    has_left=has_left,
    has_right=has_right,
    has_top=has_top,
    has_bottom=has_bottom,
  )


# =============================================================================
# Grid State Management
# =============================================================================


class QuadrantGrid:
  """
  Manages the state of quadrants in a tile generation grid.

  The grid tracks which quadrants are:
  - GENERATED: Already have pixel art generation
  - SELECTED: Selected for current infill generation
  - EMPTY: Not yet generated

  Provides validation to ensure selected quadrants can be generated
  without creating seams.
  """

  def __init__(self, width: int = 6, height: int = 4):
    """
    Initialize a quadrant grid.

    Args:
        width: Number of quadrants horizontally
        height: Number of quadrants vertically
    """
    self.width = width
    self.height = height
    self._states: dict[QuadrantPosition, QuadrantState] = {}

    # Initialize all quadrants as empty
    for x in range(width):
      for y in range(height):
        self._states[QuadrantPosition(x, y)] = QuadrantState.EMPTY

  def get_state(self, x: int, y: int) -> QuadrantState:
    """Get the state of a quadrant at position (x, y)."""
    pos = QuadrantPosition(x, y)
    return self._states.get(pos, QuadrantState.EMPTY)

  def set_state(self, x: int, y: int, state: QuadrantState) -> None:
    """Set the state of a quadrant at position (x, y)."""
    pos = QuadrantPosition(x, y)
    if 0 <= x < self.width and 0 <= y < self.height:
      self._states[pos] = state

  def set_generated(self, positions: Sequence[tuple[int, int]]) -> None:
    """Mark multiple positions as generated."""
    for x, y in positions:
      self.set_state(x, y, QuadrantState.GENERATED)

  def set_selected(self, positions: Sequence[tuple[int, int]]) -> None:
    """Mark multiple positions as selected for generation."""
    for x, y in positions:
      self.set_state(x, y, QuadrantState.SELECTED)

  def get_selected_positions(self) -> list[QuadrantPosition]:
    """Get all positions marked as selected."""
    return [
      pos for pos, state in self._states.items() if state == QuadrantState.SELECTED
    ]

  def get_generated_positions(self) -> list[QuadrantPosition]:
    """Get all positions marked as generated."""
    return [
      pos for pos, state in self._states.items() if state == QuadrantState.GENERATED
    ]

  def validate_selection(self) -> tuple[bool, str]:
    """
    Validate that the current selection is legal for generation.

    A selection is legal if:
    1. Selected quadrants form a contiguous region
    2. The selection can fit within a 2x2 tile (template image)
    3. All generated neighbors can be included in the template without
       creating seams (contextless borders)

    Returns:
        Tuple of (is_valid, error_message)
    """
    selected = self.get_selected_positions()

    if not selected:
      return False, "No quadrants selected"

    # Check contiguity
    if not self._is_contiguous(selected):
      return False, "Selected quadrants are not contiguous"

    # Get bounding box of selection
    min_x = min(p.x for p in selected)
    max_x = max(p.x for p in selected)
    min_y = min(p.y for p in selected)
    max_y = max(p.y for p in selected)

    sel_width = max_x - min_x + 1
    sel_height = max_y - min_y + 1

    # Selection must fit in a 2x2 template
    if sel_width > 2 or sel_height > 2:
      return (
        False,
        f"Selection bounds ({sel_width}x{sel_height}) exceed 2x2 template size",
      )

    # Check for contextless borders with generated neighbors
    valid, error = self._check_neighbor_context(selected)
    if not valid:
      return False, error

    return True, "Valid selection"

  def _is_contiguous(self, positions: list[QuadrantPosition]) -> bool:
    """Check if a set of positions forms a contiguous region."""
    if len(positions) <= 1:
      return True

    pos_set = set(positions)
    visited = set()
    stack = [positions[0]]

    while stack:
      current = stack.pop()
      if current in visited:
        continue
      visited.add(current)

      for neighbor in current.neighbors():
        if neighbor in pos_set and neighbor not in visited:
          stack.append(neighbor)

    return len(visited) == len(positions)

  def _check_neighbor_context(
    self, selected: list[QuadrantPosition]
  ) -> tuple[bool, str]:
    """
    Check that all generated neighbors can be included in template.

    Uses the generic InfillPositioner to validate that a valid position exists
    for the infill region that doesn't create seams with generated neighbors.
    """
    # Get infill dimensions and context
    infill_width, infill_height = quadrant_selection_to_infill(selected)
    context = get_context_from_grid(self, selected)

    # Use the generic positioner to find a valid position
    positioner = InfillPositioner(template_size=1024)
    position = positioner.find_optimal_position(infill_width, infill_height, context)

    if position is None:
      return (
        False,
        f"No valid position for {infill_width}x{infill_height} infill with "
        f"generated neighbors (L={context.has_left}, R={context.has_right}, "
        f"T={context.has_top}, B={context.has_bottom})",
      )

    return True, ""

  def get_infill_position(
    self, quadrant_size: int = 512, template_size: int = 1024
  ) -> tuple[int, int]:
    """
    Get the optimal pixel position for the infill region in the template.

    Uses InfillPositioner to find the best position that:
    - Maximizes context from generated neighbors
    - Avoids seams at template edges

    Args:
      quadrant_size: Size of each quadrant in pixels (default 512)
      template_size: Size of the template in pixels (default 1024)

    Returns:
      (x, y) pixel position for top-left of infill region
    """
    selected = self.get_selected_positions()
    if not selected:
      raise ValueError("No quadrants selected")

    infill_width, infill_height = quadrant_selection_to_infill(selected, quadrant_size)
    context = get_context_from_grid(self, selected)

    positioner = InfillPositioner(template_size=template_size)
    position = positioner.find_optimal_position(infill_width, infill_height, context)

    if position is None:
      raise ValueError("No valid infill position exists")

    return position

  def get_infill_bounds(
    self, quadrant_size: int = 512, template_size: int = 1024
  ) -> BoundingBox:
    """
    Get the optimal bounding box for the infill region in the template.

    Returns:
      BoundingBox for the infill region in pixel coordinates
    """
    x, y = self.get_infill_position(quadrant_size, template_size)
    infill_width, infill_height = quadrant_selection_to_infill(
      self.get_selected_positions(), quadrant_size
    )
    return BoundingBox(left=x, top=y, right=x + infill_width, bottom=y + infill_height)

  def is_centered_infill(
    self, quadrant_size: int = 512, template_size: int = 1024
  ) -> bool:
    """
    Check if the infill position is centered (not aligned to quadrant grid).

    A centered infill means the position is not at a quadrant boundary,
    which requires special handling when compositing the template.

    Returns:
      True if the infill is centered (position has non-zero margins on both axes)
    """
    x, y = self.get_infill_position(quadrant_size, template_size)
    infill_width, infill_height = quadrant_selection_to_infill(
      self.get_selected_positions(), quadrant_size
    )

    margin_h = template_size - infill_width
    margin_v = template_size - infill_height

    # Centered if not at edge on both axes
    not_at_left_edge = x > 0
    not_at_right_edge = x + infill_width < template_size
    not_at_top_edge = y > 0
    not_at_bottom_edge = y + infill_height < template_size

    # It's "centered" if we have margin on BOTH sides of at least one axis
    horizontally_centered = not_at_left_edge and not_at_right_edge
    vertically_centered = not_at_top_edge and not_at_bottom_edge

    return horizontally_centered or vertically_centered

  def get_template_bounds(self) -> tuple[int, int, int, int]:
    """
    Get the optimal template bounds (anchor position) for the selection.

    DEPRECATED: Use get_infill_position() and get_infill_bounds() instead.
    This method is kept for backwards compatibility.

    Returns:
      Tuple of (anchor_x, anchor_y, template_width, template_height)
    """
    selected = self.get_selected_positions()
    if not selected:
      raise ValueError("No quadrants selected")

    # Get the infill position in pixels
    infill_x, infill_y = self.get_infill_position()

    # Convert pixel position to quadrant anchor
    # For quadrant-aligned positions, this is straightforward
    # For centered positions, we use the selection's min position
    min_x = min(p.x for p in selected)
    min_y = min(p.y for p in selected)

    # If infill is at position (0, 0) in template, anchor is at selection min
    # If infill is at position (512, 0), we need to include left neighbor
    if infill_x >= 512:
      anchor_x = min_x - 1
    else:
      anchor_x = min_x

    if infill_y >= 512:
      anchor_y = min_y - 1
    else:
      anchor_y = min_y

    return (anchor_x, anchor_y, 2, 2)

  def __str__(self) -> str:
    """Return a visual representation of the grid."""
    lines = []
    for y in range(self.height):
      row = []
      for x in range(self.width):
        state = self.get_state(x, y)
        row.append(state.value)
      lines.append(" ".join(row))
    return "\n".join(lines)


# =============================================================================
# Template Image Creation
# =============================================================================


def calculate_selection_pixel_bounds(
  selected_positions: list[QuadrantPosition],
  anchor_x: int,
  anchor_y: int,
  quadrant_width: int,
  quadrant_height: int,
) -> BoundingBox:
  """
  Calculate the pixel bounding box of selected quadrants within a template.

  Args:
      selected_positions: List of selected quadrant positions
      anchor_x: X coordinate of template's top-left quadrant
      anchor_y: Y coordinate of template's top-left quadrant
      quadrant_width: Width of each quadrant in pixels
      quadrant_height: Height of each quadrant in pixels

  Returns:
      BoundingBox in pixel coordinates relative to template
  """
  # Get bounds relative to template anchor
  rel_positions = [(pos.x - anchor_x, pos.y - anchor_y) for pos in selected_positions]

  min_dx = min(p[0] for p in rel_positions)
  max_dx = max(p[0] for p in rel_positions)
  min_dy = min(p[1] for p in rel_positions)
  max_dy = max(p[1] for p in rel_positions)

  return BoundingBox(
    left=min_dx * quadrant_width,
    top=min_dy * quadrant_height,
    right=(max_dx + 1) * quadrant_width,
    bottom=(max_dy + 1) * quadrant_height,
  )


def draw_red_border(
  image: Image.Image,
  box: BoundingBox,
  border_width: int = 2,
) -> Image.Image:
  """
  Draw a red border around a rectangular region.

  The border is drawn ON TOP of the image (no pixel displacement).

  Args:
      image: The image to draw on (will be copied)
      box: Bounding box to draw border around
      border_width: Width of the border in pixels (default: 2)

  Returns:
      New image with red border drawn
  """
  # Convert to RGBA if needed
  if image.mode != "RGBA":
    image = image.convert("RGBA")

  result = image.copy()
  draw = ImageDraw.Draw(result)

  red = (255, 0, 0, 255)

  # Draw rectangle outline with specified width
  for i in range(border_width):
    draw.rectangle(
      [box.left + i, box.top + i, box.right - 1 - i, box.bottom - 1 - i],
      outline=red,
      fill=None,
    )

  return result


def create_template_image(
  grid: QuadrantGrid,
  get_render: callable,
  get_generation: callable,
  quadrant_width: int = 512,
  quadrant_height: int = 512,
  border_width: int = 2,
) -> tuple[Image.Image, BoundingBox, tuple[int, int]]:
  """
  Create a template image for infill generation.

  The template is a 2x2 quadrant (tile-sized) image where:
  - Selected quadrants are filled with render pixels
  - Generated neighbor quadrants are filled with generation pixels
  - Empty quadrants are left transparent
  - A red border is drawn around the render region

  Args:
      grid: QuadrantGrid with current state
      get_render: Callable(x, y) -> Image.Image | None to get render for position
      get_generation: Callable(x, y) -> Image.Image | None to get generation for position
      quadrant_width: Width of each quadrant in pixels
      quadrant_height: Height of each quadrant in pixels
      border_width: Width of the red border in pixels

  Returns:
      Tuple of (template_image, render_bounds, (anchor_x, anchor_y))
  """
  # Validate selection first
  is_valid, error = grid.validate_selection()
  if not is_valid:
    raise ValueError(f"Invalid selection: {error}")

  # Get template bounds
  anchor_x, anchor_y, _, _ = grid.get_template_bounds()

  # Create template image (2x2 quadrants)
  template = Image.new("RGBA", (quadrant_width * 2, quadrant_height * 2), (0, 0, 0, 0))

  selected = grid.get_selected_positions()
  selected_set = set(selected)

  # Fill in quadrants
  for dx in range(2):
    for dy in range(2):
      qx = anchor_x + dx
      qy = anchor_y + dy
      pos = QuadrantPosition(qx, qy)
      paste_x = dx * quadrant_width
      paste_y = dy * quadrant_height

      if pos in selected_set:
        # Use render pixels for selected quadrants
        render_img = get_render(qx, qy)
        if render_img is not None:
          if render_img.mode != "RGBA":
            render_img = render_img.convert("RGBA")
          template.paste(render_img, (paste_x, paste_y))
      elif grid.get_state(qx, qy) == QuadrantState.GENERATED:
        # Use generation pixels for generated neighbors
        gen_img = get_generation(qx, qy)
        if gen_img is not None:
          if gen_img.mode != "RGBA":
            gen_img = gen_img.convert("RGBA")
          template.paste(gen_img, (paste_x, paste_y))
      # Empty quadrants stay transparent

  # Calculate render bounds and draw border
  render_bounds = calculate_selection_pixel_bounds(
    selected, anchor_x, anchor_y, quadrant_width, quadrant_height
  )

  template = draw_red_border(template, render_bounds, border_width)

  return template, render_bounds, (anchor_x, anchor_y)


def create_centered_template_image(
  grid: QuadrantGrid,
  get_render: callable,
  get_generation: callable,
  quadrant_width: int = 512,
  quadrant_height: int = 512,
  border_width: int = 2,
) -> tuple[Image.Image, BoundingBox, tuple[int, int]]:
  """
  Create a centered template image for a 1x1 selection with neighbors on multiple sides.

  For a single quadrant selection surrounded by generated neighbors, this creates
  a template where the selected 512x512 region is CENTERED in the 1024x1024 template,
  with 256px of context from each of the 4 neighboring generated quadrants.

  Args:
    grid: QuadrantGrid with current state (must have exactly 1 selected quadrant)
    get_render: Callable(x, y) -> Image.Image | None to get render for position
    get_generation: Callable(x, y) -> Image.Image | None to get generation for position
    quadrant_width: Width of each quadrant in pixels
    quadrant_height: Height of each quadrant in pixels
    border_width: Width of the red border in pixels

  Returns:
    Tuple of (template_image, render_bounds, (selected_x, selected_y))
  """
  selected = grid.get_selected_positions()
  if len(selected) != 1:
    raise ValueError("Centered template requires exactly 1 selected quadrant")

  pos = selected[0]
  half_w = quadrant_width // 2  # 256
  half_h = quadrant_height // 2  # 256

  # Create template image (1024x1024)
  template = Image.new("RGBA", (quadrant_width * 2, quadrant_height * 2), (0, 0, 0, 0))

  # Get the selected quadrant's render (centered in template)
  render_img = get_render(pos.x, pos.y)
  if render_img is not None:
    if render_img.mode != "RGBA":
      render_img = render_img.convert("RGBA")
    # Paste at center (256, 256)
    template.paste(render_img, (half_w, half_h))

  # Get context from 4 direct neighbors (edges)
  # Top neighbor: bottom 256 rows
  top_gen = get_generation(pos.x, pos.y - 1)
  if top_gen is not None:
    if top_gen.mode != "RGBA":
      top_gen = top_gen.convert("RGBA")
    # Crop bottom half
    cropped = top_gen.crop((0, half_h, quadrant_width, quadrant_height))
    template.paste(cropped, (half_w, 0))

  # Bottom neighbor: top 256 rows
  bottom_gen = get_generation(pos.x, pos.y + 1)
  if bottom_gen is not None:
    if bottom_gen.mode != "RGBA":
      bottom_gen = bottom_gen.convert("RGBA")
    # Crop top half
    cropped = bottom_gen.crop((0, 0, quadrant_width, half_h))
    template.paste(cropped, (half_w, half_h + quadrant_height))

  # Left neighbor: right 256 columns
  left_gen = get_generation(pos.x - 1, pos.y)
  if left_gen is not None:
    if left_gen.mode != "RGBA":
      left_gen = left_gen.convert("RGBA")
    # Crop right half
    cropped = left_gen.crop((half_w, 0, quadrant_width, quadrant_height))
    template.paste(cropped, (0, half_h))

  # Right neighbor: left 256 columns
  right_gen = get_generation(pos.x + 1, pos.y)
  if right_gen is not None:
    if right_gen.mode != "RGBA":
      right_gen = right_gen.convert("RGBA")
    # Crop left half
    cropped = right_gen.crop((0, 0, half_w, quadrant_height))
    template.paste(cropped, (half_w + quadrant_width, half_h))

  # Get context from 4 diagonal neighbors (corners)
  # Top-left corner: bottom-right 256x256
  tl_gen = get_generation(pos.x - 1, pos.y - 1)
  if tl_gen is not None:
    if tl_gen.mode != "RGBA":
      tl_gen = tl_gen.convert("RGBA")
    cropped = tl_gen.crop((half_w, half_h, quadrant_width, quadrant_height))
    template.paste(cropped, (0, 0))

  # Top-right corner: bottom-left 256x256
  tr_gen = get_generation(pos.x + 1, pos.y - 1)
  if tr_gen is not None:
    if tr_gen.mode != "RGBA":
      tr_gen = tr_gen.convert("RGBA")
    cropped = tr_gen.crop((0, half_h, half_w, quadrant_height))
    template.paste(cropped, (half_w + quadrant_width, 0))

  # Bottom-left corner: top-right 256x256
  bl_gen = get_generation(pos.x - 1, pos.y + 1)
  if bl_gen is not None:
    if bl_gen.mode != "RGBA":
      bl_gen = bl_gen.convert("RGBA")
    cropped = bl_gen.crop((half_w, 0, quadrant_width, half_h))
    template.paste(cropped, (0, half_h + quadrant_height))

  # Bottom-right corner: top-left 256x256
  br_gen = get_generation(pos.x + 1, pos.y + 1)
  if br_gen is not None:
    if br_gen.mode != "RGBA":
      br_gen = br_gen.convert("RGBA")
    cropped = br_gen.crop((0, 0, half_w, half_h))
    template.paste(cropped, (half_w + quadrant_width, half_h + quadrant_height))

  # The render region is centered at (256, 256) with size 512x512
  render_bounds = BoundingBox(
    left=half_w,
    top=half_h,
    right=half_w + quadrant_width,
    bottom=half_h + quadrant_height,
  )

  template = draw_red_border(template, render_bounds, border_width)

  return template, render_bounds, (pos.x, pos.y)


def extract_centered_quadrant(
  generated_image: Image.Image,
  quadrant_width: int = 512,
  quadrant_height: int = 512,
) -> Image.Image:
  """
  Extract the centered quadrant from a generated image.

  For images generated from centered templates, the selected quadrant
  is at the center of the image.

  Args:
    generated_image: The full generated tile image (1024x1024)
    quadrant_width: Width of the quadrant in pixels
    quadrant_height: Height of the quadrant in pixels

  Returns:
    The extracted quadrant image (512x512)
  """
  half_w = quadrant_width // 2
  half_h = quadrant_height // 2

  return generated_image.crop(
    (
      half_w,
      half_h,
      half_w + quadrant_width,
      half_h + quadrant_height,
    )
  )


# =============================================================================
# Generation Extraction
# =============================================================================


def extract_generated_quadrants(
  generated_image: Image.Image,
  selected_positions: list[QuadrantPosition],
  anchor_x: int,
  anchor_y: int,
  quadrant_width: int = 512,
  quadrant_height: int = 512,
) -> dict[QuadrantPosition, Image.Image]:
  """
  Extract the selected quadrants from a generated image.

  After the model generates the infilled image, this function extracts
  the quadrants that were selected for generation.

  Args:
      generated_image: The full generated tile image
      selected_positions: List of quadrant positions that were selected
      anchor_x: X coordinate of template's top-left quadrant
      anchor_y: Y coordinate of template's top-left quadrant
      quadrant_width: Width of each quadrant in pixels
      quadrant_height: Height of each quadrant in pixels

  Returns:
      Dict mapping QuadrantPosition to cropped quadrant Image
  """
  result = {}

  for pos in selected_positions:
    # Calculate position relative to template anchor
    dx = pos.x - anchor_x
    dy = pos.y - anchor_y

    # Calculate crop box
    left = dx * quadrant_width
    top = dy * quadrant_height
    right = left + quadrant_width
    bottom = top + quadrant_height

    # Crop the quadrant
    quadrant_img = generated_image.crop((left, top, right, bottom))
    result[pos] = quadrant_img

  return result


# =============================================================================
# Convenience Functions for Common Patterns
# =============================================================================


def create_half_template(
  side: str,
  get_render: callable,
  get_generation: callable,
  anchor_x: int = 0,
  anchor_y: int = 0,
  quadrant_width: int = 512,
  quadrant_height: int = 512,
) -> tuple[Image.Image, list[QuadrantPosition], tuple[int, int]]:
  """
  Create a template for half-tile generation (like the original generate_tile).

  Args:
      side: Which side to fill with renders: "left", "right", "top", "bottom"
      get_render: Callable(x, y) -> Image.Image | None
      get_generation: Callable(x, y) -> Image.Image | None
      anchor_x: X coordinate of template's top-left quadrant
      anchor_y: Y coordinate of template's top-left quadrant
      quadrant_width: Width of each quadrant in pixels
      quadrant_height: Height of each quadrant in pixels

  Returns:
      Tuple of (template_image, selected_positions, (anchor_x, anchor_y))
  """
  grid = QuadrantGrid(width=4, height=4)

  # Set up the generation state based on side
  if side == "left":
    # Right side is generated, left side selected
    grid.set_generated([(anchor_x + 1, anchor_y), (anchor_x + 1, anchor_y + 1)])
    grid.set_selected([(anchor_x, anchor_y), (anchor_x, anchor_y + 1)])
  elif side == "right":
    # Left side is generated, right side selected
    grid.set_generated([(anchor_x, anchor_y), (anchor_x, anchor_y + 1)])
    grid.set_selected([(anchor_x + 1, anchor_y), (anchor_x + 1, anchor_y + 1)])
  elif side == "top":
    # Bottom is generated, top selected
    grid.set_generated([(anchor_x, anchor_y + 1), (anchor_x + 1, anchor_y + 1)])
    grid.set_selected([(anchor_x, anchor_y), (anchor_x + 1, anchor_y)])
  elif side == "bottom":
    # Top is generated, bottom selected
    grid.set_generated([(anchor_x, anchor_y), (anchor_x + 1, anchor_y)])
    grid.set_selected([(anchor_x, anchor_y + 1), (anchor_x + 1, anchor_y + 1)])
  else:
    raise ValueError(f"Invalid side: {side}. Use 'left', 'right', 'top', or 'bottom'")

  template, bounds, anchor = create_template_image(
    grid, get_render, get_generation, quadrant_width, quadrant_height
  )

  return template, grid.get_selected_positions(), anchor


def create_single_quadrant_template(
  quadrant_dx: int,
  quadrant_dy: int,
  get_render: callable,
  get_generation: callable,
  anchor_x: int = 0,
  anchor_y: int = 0,
  quadrant_width: int = 512,
  quadrant_height: int = 512,
) -> tuple[Image.Image, list[QuadrantPosition], tuple[int, int]]:
  """
  Create a template for single quadrant generation.

  This is useful for filling in a single quadrant surrounded by up to
  3 generated neighbors.

  Args:
      quadrant_dx: X offset (0 or 1) of selected quadrant within 2x2 tile
      quadrant_dy: Y offset (0 or 1) of selected quadrant within 2x2 tile
      get_render: Callable(x, y) -> Image.Image | None
      get_generation: Callable(x, y) -> Image.Image | None
      anchor_x: X coordinate of template's top-left quadrant
      anchor_y: Y coordinate of template's top-left quadrant
      quadrant_width: Width of each quadrant in pixels
      quadrant_height: Height of each quadrant in pixels

  Returns:
      Tuple of (template_image, selected_positions, (anchor_x, anchor_y))
  """
  grid = QuadrantGrid(width=4, height=4)

  # Mark all quadrants in the 2x2 tile as generated except the selected one
  for dx in range(2):
    for dy in range(2):
      qx = anchor_x + dx
      qy = anchor_y + dy
      if dx == quadrant_dx and dy == quadrant_dy:
        grid.set_state(qx, qy, QuadrantState.SELECTED)
      else:
        grid.set_state(qx, qy, QuadrantState.GENERATED)

  template, bounds, anchor = create_template_image(
    grid, get_render, get_generation, quadrant_width, quadrant_height
  )

  return template, grid.get_selected_positions(), anchor


def create_middle_strip_template(
  orientation: str,
  get_render: callable,
  get_generation: callable,
  anchor_x: int = 0,
  anchor_y: int = 0,
  quadrant_width: int = 512,
  quadrant_height: int = 512,
) -> tuple[Image.Image, list[QuadrantPosition], tuple[int, int]]:
  """
  Create a template for middle strip generation (vertical or horizontal).

  This handles the case where selected quadrants are in the middle with
  generated neighbors on both sides (like the example in the task).

  Note: This requires 4 quadrants wide (vertical) or tall (horizontal)
  to properly represent the context.

  Args:
      orientation: "vertical" or "horizontal"
      get_render: Callable(x, y) -> Image.Image | None
      get_generation: Callable(x, y) -> Image.Image | None
      anchor_x: X coordinate of template's top-left quadrant
      anchor_y: Y coordinate of template's top-left quadrant
      quadrant_width: Width of each quadrant in pixels
      quadrant_height: Height of each quadrant in pixels

  Returns:
      Tuple of (template_image, selected_positions, (anchor_x, anchor_y))
  """
  # For middle strip, we create a special 4-quadrant wide template
  # that includes context from both sides

  if orientation == "vertical":
    # Vertical strip in the middle - need left, middle, middle, right
    # But our template is only 2x2, so we pack 25% left, 50% middle, 25% right
    # Actually, for the model this is complex. Let's stick with 2x2 for now
    # and handle this as a special case with partial quadrant generation
    raise NotImplementedError(
      "Middle strip templates require special handling not yet implemented"
    )
  elif orientation == "horizontal":
    raise NotImplementedError(
      "Middle strip templates require special handling not yet implemented"
    )
  else:
    raise ValueError(f"Invalid orientation: {orientation}")


# =============================================================================
# Testing Utilities
# =============================================================================


def visualize_grid(grid: QuadrantGrid) -> str:
  """
  Create an ASCII visualization of the grid state.

  Returns a string showing the grid with:
  - G: Generated quadrant
  - S: Selected quadrant (to be generated)
  - x: Empty quadrant
  """
  return str(grid)


def create_test_grid_state(scenario: str) -> QuadrantGrid:
  """
  Create a QuadrantGrid for common test scenarios.

  Scenarios:
  - "half_left": Left half generated, right half selected
  - "half_right": Right half generated, left half selected
  - "half_top": Top half generated, bottom half selected
  - "half_bottom": Bottom half generated, top half selected
  - "single_tl": TL quadrant selected, rest generated
  - "single_tr": TR quadrant selected, rest generated
  - "single_bl": BL quadrant selected, rest generated
  - "single_br": BR quadrant selected, rest generated
  - "full": All 4 quadrants selected (fresh tile)
  """
  grid = QuadrantGrid(width=4, height=4)

  if scenario == "half_left":
    grid.set_generated([(1, 0), (1, 1)])
    grid.set_selected([(0, 0), (0, 1)])
  elif scenario == "half_right":
    grid.set_generated([(0, 0), (0, 1)])
    grid.set_selected([(1, 0), (1, 1)])
  elif scenario == "half_top":
    grid.set_generated([(0, 1), (1, 1)])
    grid.set_selected([(0, 0), (1, 0)])
  elif scenario == "half_bottom":
    grid.set_generated([(0, 0), (1, 0)])
    grid.set_selected([(0, 1), (1, 1)])
  elif scenario == "single_tl":
    grid.set_generated([(1, 0), (0, 1), (1, 1)])
    grid.set_selected([(0, 0)])
  elif scenario == "single_tr":
    grid.set_generated([(0, 0), (0, 1), (1, 1)])
    grid.set_selected([(1, 0)])
  elif scenario == "single_bl":
    grid.set_generated([(0, 0), (1, 0), (1, 1)])
    grid.set_selected([(0, 1)])
  elif scenario == "single_br":
    grid.set_generated([(0, 0), (1, 0), (0, 1)])
    grid.set_selected([(1, 1)])
  elif scenario == "full":
    grid.set_selected([(0, 0), (1, 0), (0, 1), (1, 1)])
  else:
    raise ValueError(f"Unknown scenario: {scenario}")

  return grid


def run_validation_tests() -> bool:
  """Run validation tests and return True if all pass."""
  print("=" * 60)
  print("Testing QuadrantGrid validation")
  print("=" * 60)

  all_passed = True

  # Test valid scenarios
  valid_scenarios = [
    "half_left",
    "half_right",
    "half_top",
    "half_bottom",
    "single_tl",
    "single_tr",
    "single_bl",
    "single_br",
    "full",
  ]

  for scenario in valid_scenarios:
    grid = create_test_grid_state(scenario)
    is_valid, msg = grid.validate_selection()
    status = "✅" if is_valid else "❌"
    print(f"\n{status} {scenario}:")
    print(grid)
    print(f"   Valid: {is_valid}, Message: {msg}")
    if not is_valid:
      all_passed = False

  # Test invalid scenario from task description
  print("\n" + "=" * 60)
  print("Testing ILLEGAL scenario from task")
  print("=" * 60)

  # This is the illegal case:
  # G G G G G x
  # G G S G G x
  # G G S G G x
  # x x x x x x
  grid = QuadrantGrid(width=6, height=4)
  grid.set_generated(
    [
      (0, 0),
      (1, 0),
      (2, 0),
      (3, 0),
      (4, 0),
      (0, 1),
      (1, 1),
      (3, 1),
      (4, 1),
      (0, 2),
      (1, 2),
      (3, 2),
      (4, 2),
    ]
  )
  grid.set_selected([(2, 1), (2, 2)])

  is_valid, msg = grid.validate_selection()
  status = "✅ (correctly rejected)" if not is_valid else "❌ (should be invalid!)"
  print(f"\n{status} Illegal scenario:")
  print(grid)
  print(f"   Valid: {is_valid}, Message: {msg}")
  if is_valid:
    all_passed = False

  # Test the LEGAL version of the above
  print("\n" + "=" * 60)
  print("Testing LEGAL scenario (single quadrant version)")
  print("=" * 60)

  # G G G G G x
  # G G S G G x
  # G G x G G x
  # x x x x x x
  grid = QuadrantGrid(width=6, height=4)
  grid.set_generated(
    [
      (0, 0),
      (1, 0),
      (2, 0),
      (3, 0),
      (4, 0),
      (0, 1),
      (1, 1),
      (3, 1),
      (4, 1),
      (0, 2),
      (1, 2),
      (3, 2),
      (4, 2),
    ]
  )
  grid.set_selected([(2, 1)])  # Only select one quadrant

  is_valid, msg = grid.validate_selection()
  status = "✅" if is_valid else "❌"
  print(f"\n{status} Legal single quadrant scenario:")
  print(grid)
  print(f"   Valid: {is_valid}, Message: {msg}")
  if not is_valid:
    all_passed = False

  return all_passed


def run_template_creation_tests(output_dir: str | None = None) -> bool:
  """
  Run template creation tests with synthetic images.

  Args:
    output_dir: If provided, save test images to this directory

  Returns:
    True if all tests pass
  """
  print("\n" + "=" * 60)
  print("Testing template image creation")
  print("=" * 60)

  from pathlib import Path

  all_passed = True
  quad_size = 128  # Use smaller size for tests

  # Create synthetic render and generation images
  def create_test_image(color: tuple[int, int, int, int]) -> Image.Image:
    """Create a test quadrant image with a solid color."""
    return Image.new("RGBA", (quad_size, quad_size), color)

  # Create color-coded quadrant getters
  render_colors = {
    (0, 0): (255, 200, 200, 255),  # Light red - TL render
    (1, 0): (200, 255, 200, 255),  # Light green - TR render
    (0, 1): (200, 200, 255, 255),  # Light blue - BL render
    (1, 1): (255, 255, 200, 255),  # Light yellow - BR render
  }

  gen_colors = {
    (0, 0): (200, 50, 50, 255),  # Dark red - TL gen
    (1, 0): (50, 200, 50, 255),  # Dark green - TR gen
    (0, 1): (50, 50, 200, 255),  # Dark blue - BL gen
    (1, 1): (200, 200, 50, 255),  # Dark yellow - BR gen
  }

  def get_render(x: int, y: int) -> Image.Image | None:
    color = render_colors.get((x, y))
    if color:
      return create_test_image(color)
    return None

  def get_generation(x: int, y: int) -> Image.Image | None:
    color = gen_colors.get((x, y))
    if color:
      return create_test_image(color)
    return None

  # Test 1: Half-left selection (right side has generation)
  print("\n📋 Test 1: Half-left template")
  grid = create_test_grid_state("half_left")
  print(grid)

  try:
    template, bounds, anchor = create_template_image(
      grid, get_render, get_generation, quad_size, quad_size
    )
    print(f"   ✅ Template created: {template.size}")
    print(f"   Bounds: {bounds.as_tuple()}")
    print(f"   Anchor: {anchor}")

    if output_dir:
      out_path = Path(output_dir) / "test_half_left.png"
      out_path.parent.mkdir(parents=True, exist_ok=True)
      template.save(out_path)
      print(f"   Saved to: {out_path}")

  except Exception as e:
    print(f"   ❌ Error: {e}")
    all_passed = False

  # Test 2: Single quadrant selection (BR with 3 neighbors)
  print("\n📋 Test 2: Single quadrant (BR) template")
  grid = create_test_grid_state("single_br")
  print(grid)

  try:
    template, bounds, anchor = create_template_image(
      grid, get_render, get_generation, quad_size, quad_size
    )
    print(f"   ✅ Template created: {template.size}")
    print(f"   Bounds: {bounds.as_tuple()}")
    print(f"   Anchor: {anchor}")

    if output_dir:
      out_path = Path(output_dir) / "test_single_br.png"
      template.save(out_path)
      print(f"   Saved to: {out_path}")

  except Exception as e:
    print(f"   ❌ Error: {e}")
    all_passed = False

  # Test 3: Full tile selection (no neighbors)
  print("\n📋 Test 3: Full tile template")
  grid = create_test_grid_state("full")
  print(grid)

  try:
    template, bounds, anchor = create_template_image(
      grid, get_render, get_generation, quad_size, quad_size
    )
    print(f"   ✅ Template created: {template.size}")
    print(f"   Bounds: {bounds.as_tuple()}")
    print(f"   Anchor: {anchor}")

    if output_dir:
      out_path = Path(output_dir) / "test_full.png"
      template.save(out_path)
      print(f"   Saved to: {out_path}")

  except Exception as e:
    print(f"   ❌ Error: {e}")
    all_passed = False

  return all_passed


def run_extraction_tests() -> bool:
  """Run quadrant extraction tests."""
  print("\n" + "=" * 60)
  print("Testing quadrant extraction")
  print("=" * 60)

  all_passed = True
  quad_size = 128

  # Create a test "generated" image (2x2 quadrants)
  gen_image = Image.new("RGBA", (quad_size * 2, quad_size * 2))

  # Fill each quadrant with a different color
  colors = {
    (0, 0): (255, 0, 0, 255),  # Red TL
    (1, 0): (0, 255, 0, 255),  # Green TR
    (0, 1): (0, 0, 255, 255),  # Blue BL
    (1, 1): (255, 255, 0, 255),  # Yellow BR
  }

  for (dx, dy), color in colors.items():
    for px in range(quad_size):
      for py in range(quad_size):
        gen_image.putpixel((dx * quad_size + px, dy * quad_size + py), color)

  # Test extraction
  selected = [QuadrantPosition(0, 0), QuadrantPosition(1, 1)]
  anchor_x, anchor_y = 0, 0

  extracted = extract_generated_quadrants(
    gen_image, selected, anchor_x, anchor_y, quad_size, quad_size
  )

  print(f"\n📋 Extracted {len(extracted)} quadrants")

  for pos, img in extracted.items():
    # Verify the color of the extracted quadrant
    sample_color = img.getpixel((quad_size // 2, quad_size // 2))
    expected_color = colors[(pos.x - anchor_x, pos.y - anchor_y)]

    if sample_color == expected_color:
      print(f"   ✅ Quadrant ({pos.x}, {pos.y}): correct color")
    else:
      print(
        f"   ❌ Quadrant ({pos.x}, {pos.y}): wrong color "
        f"(got {sample_color}, expected {expected_color})"
      )
      all_passed = False

  return all_passed


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Test the generate_template library")
  parser.add_argument(
    "--output-dir",
    type=str,
    help="Directory to save test images (optional)",
  )
  args = parser.parse_args()

  # Run all tests
  validation_passed = run_validation_tests()
  template_passed = run_template_creation_tests(args.output_dir)
  extraction_passed = run_extraction_tests()

  print("\n" + "=" * 60)
  print("TEST SUMMARY")
  print("=" * 60)
  print(f"   Validation tests: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
  print(f"   Template tests:   {'✅ PASSED' if template_passed else '❌ FAILED'}")
  print(f"   Extraction tests: {'✅ PASSED' if extraction_passed else '❌ FAILED'}")

  all_passed = validation_passed and template_passed and extraction_passed
  print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed!'}")
  print("=" * 60)

  exit(0 if all_passed else 1)
