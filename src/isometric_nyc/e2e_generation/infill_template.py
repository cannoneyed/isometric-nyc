"""
Generic infill template generation for arbitrary rectangular regions.

This module handles creating template images for infill generation where:
- The infill region is an arbitrary rectangle (up to 50% of tile area)
- Context is maximized by optimal placement within the template
- Edge constraints ensure no seams with generated neighbors

Key concepts:
- InfillRegion: A rectangular region to be filled with generated pixels
- TemplateSpec: Specification for how to build the template
- The template is always 1024x1024 pixels
- Quadrants (512x512) are the storage unit in the database

Usage:
  from isometric_nyc.e2e_generation.infill_template import (
      InfillRegion,
      TemplateBuilder,
  )

  # Create an infill region (e.g., a single quadrant)
  region = InfillRegion.from_quadrant(x=2, y=3)

  # Or create a custom rectangle
  region = InfillRegion(x=256, y=256, width=512, height=512)

  # Build template
  builder = TemplateBuilder(region, get_generation_func)
  template, bounds = builder.build()
"""

from dataclasses import dataclass
from typing import Callable

from PIL import Image, ImageDraw

# Template and quadrant dimensions
TEMPLATE_SIZE = 1024
QUADRANT_SIZE = 512
MAX_INFILL_AREA = TEMPLATE_SIZE * TEMPLATE_SIZE // 2  # 50% of template


@dataclass
class InfillRegion:
  """
  A rectangular region to be infilled.

  Coordinates are in "world" pixel space, where:
  - (0, 0) is the top-left of quadrant (0, 0)
  - x increases to the right
  - y increases downward
  - Each quadrant is 512x512 pixels
  """

  x: int  # World x coordinate (top-left of region)
  y: int  # World y coordinate (top-left of region)
  width: int  # Width in pixels
  height: int  # Height in pixels

  @classmethod
  def from_quadrant(cls, qx: int, qy: int) -> "InfillRegion":
    """Create an infill region for a single quadrant."""
    return cls(
      x=qx * QUADRANT_SIZE,
      y=qy * QUADRANT_SIZE,
      width=QUADRANT_SIZE,
      height=QUADRANT_SIZE,
    )

  @classmethod
  def from_quadrants(cls, quadrants: list[tuple[int, int]]) -> "InfillRegion":
    """
    Create an infill region covering multiple quadrants.

    The quadrants must form a contiguous rectangle.
    """
    if not quadrants:
      raise ValueError("At least one quadrant required")

    min_qx = min(q[0] for q in quadrants)
    max_qx = max(q[0] for q in quadrants)
    min_qy = min(q[1] for q in quadrants)
    max_qy = max(q[1] for q in quadrants)

    return cls(
      x=min_qx * QUADRANT_SIZE,
      y=min_qy * QUADRANT_SIZE,
      width=(max_qx - min_qx + 1) * QUADRANT_SIZE,
      height=(max_qy - min_qy + 1) * QUADRANT_SIZE,
    )

  @property
  def area(self) -> int:
    """Total area in pixels."""
    return self.width * self.height

  @property
  def right(self) -> int:
    """Right edge x coordinate."""
    return self.x + self.width

  @property
  def bottom(self) -> int:
    """Bottom edge y coordinate."""
    return self.y + self.height

  def is_valid_size(self) -> bool:
    """Check if the region is within the allowed size (≤50% or exactly 100% of template)."""
    # Allow up to 50% OR exactly 100% (full tile)
    # Full tile is valid only if there are no generated neighbors (checked separately)
    return self.area <= MAX_INFILL_AREA or self.is_full_tile()

  def is_full_tile(self) -> bool:
    """Check if this region is exactly a full tile (1024x1024)."""
    return self.width == TEMPLATE_SIZE and self.height == TEMPLATE_SIZE

  def overlapping_quadrants(self) -> list[tuple[int, int]]:
    """Get list of quadrant (qx, qy) positions that overlap with this region."""
    quadrants = []

    # Find quadrant range
    start_qx = self.x // QUADRANT_SIZE
    end_qx = (self.right - 1) // QUADRANT_SIZE
    start_qy = self.y // QUADRANT_SIZE
    end_qy = (self.bottom - 1) // QUADRANT_SIZE

    for qx in range(start_qx, end_qx + 1):
      for qy in range(start_qy, end_qy + 1):
        quadrants.append((qx, qy))

    return quadrants

  def __str__(self) -> str:
    return f"InfillRegion(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


@dataclass
class TemplatePlacement:
  """
  Describes where to place the infill region within the template.

  The template is always 1024x1024. This specifies:
  - Where the infill region should be placed within it
  - The world coordinate offset for context pixels
  - Which quadrants are primary (user selected) vs padding (auto-expanded)
  """

  # Position of infill region within template (0-1024)
  infill_x: int
  infill_y: int

  # World coordinate of template's top-left corner
  world_offset_x: int
  world_offset_y: int

  @property
  def infill_right(self) -> int:
    return self.infill_x + self._infill_width

  @property
  def infill_bottom(self) -> int:
    return self.infill_y + self._infill_height

  def __post_init__(self):
    # These will be set by the builder
    self._infill_width = 0
    self._infill_height = 0
    # Primary quadrants are the user-selected ones
    # Padding quadrants are auto-added to cover missing context
    self._primary_quadrants: list[tuple[int, int]] = []
    self._padding_quadrants: list[tuple[int, int]] = []
    # The expanded infill region (if different from primary)
    self._expanded_region: InfillRegion | None = None

  @property
  def primary_quadrants(self) -> list[tuple[int, int]]:
    """Quadrants originally selected by user."""
    return self._primary_quadrants

  @property
  def padding_quadrants(self) -> list[tuple[int, int]]:
    """Quadrants auto-added to cover missing context."""
    return self._padding_quadrants

  @property
  def all_infill_quadrants(self) -> list[tuple[int, int]]:
    """All quadrants that will be filled with render pixels."""
    return self._primary_quadrants + self._padding_quadrants

  @property
  def is_expanded(self) -> bool:
    """Whether the infill region was expanded to cover missing context."""
    return len(self._padding_quadrants) > 0


class TemplateBuilder:
  """
  Builds template images for infill generation.

  Handles:
  - Optimal placement of infill region to maximize context
  - Validation of edge constraints
  - Assembly of template from quadrant data
  """

  def __init__(
    self,
    infill_region: InfillRegion,
    has_generation: Callable[[int, int], bool],
    get_render: Callable[[int, int], Image.Image | None] | None = None,
    get_generation: Callable[[int, int], Image.Image | None] | None = None,
  ):
    """
    Initialize the template builder.

    Args:
      infill_region: The region to be infilled
      has_generation: Callable(qx, qy) -> bool to check if quadrant has generation
      get_render: Callable(qx, qy) -> Image to get render for quadrant
      get_generation: Callable(qx, qy) -> Image to get generation for quadrant
    """
    self.region = infill_region
    self.has_generation = has_generation
    self.get_render = get_render
    self.get_generation = get_generation
    self._last_validation_error = ""

    # Validate region size
    if not infill_region.is_valid_size():
      raise ValueError(
        f"Infill region too large: {infill_region.area} pixels (max: {MAX_INFILL_AREA})"
      )

  def find_optimal_placement(
    self, allow_expansion: bool = False
  ) -> TemplatePlacement | None:
    """
    Find the optimal placement for the infill region within the template.

    Args:
      allow_expansion: If True, automatically expand infill region to cover
                       missing context quadrants (they'll be filled with render
                       pixels and discarded after generation)

    Returns None if no valid placement exists (would create seams).

    The strategy:
    1. Try to maximize context by checking generated neighbors
    2. Position infill to include as much context as possible
    3. Validate that edges touching template boundary have no generated neighbors
    4. If placement has missing context, try alternative placements
    5. If allow_expansion and context quadrants are still missing, expand the infill
    """
    # Try multiple placement strategies
    # Strategy 1: Maximize context (original approach)
    # Strategy 2+: Exclude problematic sides that would pull in non-generated quadrants

    placement = self._try_placement_with_context_preferences(
      include_left=True,
      include_right=True,
      include_top=True,
      include_bottom=True,
    )

    if placement is not None:
      missing = self._find_missing_context_quadrants(placement)
      if not missing:
        return placement

      # There are missing context quadrants - try alternative placements
      # that exclude the sides causing the problem
      alternative = self._try_alternative_placements(missing, allow_expansion)
      if alternative is not None:
        return alternative

      # All alternatives failed, try expansion on the original placement
      if allow_expansion:
        expanded_placement = self._expand_to_cover_missing(placement, missing)
        if expanded_placement is not None:
          return expanded_placement

      # Everything failed
      missing_str = ", ".join(f"({qx}, {qy})" for qx, qy in missing)
      self._last_validation_error = (
        f"Context quadrants missing generations: {missing_str}"
      )
      return None

    return placement

  def _try_placement_with_context_preferences(
    self,
    include_left: bool,
    include_right: bool,
    include_top: bool,
    include_bottom: bool,
  ) -> TemplatePlacement | None:
    """
    Try to find a valid placement with given context preferences.

    Args:
      include_left: Whether to try to include left context
      include_right: Whether to try to include right context
      include_top: Whether to try to include top context
      include_bottom: Whether to try to include bottom context

    Returns:
      TemplatePlacement if valid (passes seam check), None otherwise
    """
    # Calculate available margin on each side
    margin_x = TEMPLATE_SIZE - self.region.width
    margin_y = TEMPLATE_SIZE - self.region.height

    # Check for generated context on each side of the infill region
    has_left_gen = self._has_generated_context("left") if include_left else False
    has_right_gen = self._has_generated_context("right") if include_right else False
    has_top_gen = self._has_generated_context("top") if include_top else False
    has_bottom_gen = self._has_generated_context("bottom") if include_bottom else False

    # Determine infill position based on context preferences
    # Horizontal positioning
    if has_left_gen and has_right_gen:
      infill_x = margin_x // 2
    elif has_left_gen:
      infill_x = margin_x
    elif has_right_gen:
      infill_x = 0
    else:
      # No horizontal context to include - position to avoid seams
      # If we're NOT including left but there IS generated content on left,
      # push infill to include right side (avoid left seam)
      actual_left_gen = self._has_generated_context("left")
      actual_right_gen = self._has_generated_context("right")
      if actual_right_gen and not actual_left_gen:
        infill_x = 0
      elif actual_left_gen and not actual_right_gen:
        infill_x = margin_x
      else:
        infill_x = 0

    # Vertical positioning
    if has_top_gen and has_bottom_gen:
      infill_y = margin_y // 2
    elif has_top_gen:
      infill_y = margin_y
    elif has_bottom_gen:
      infill_y = 0
    else:
      # No vertical context to include - position to avoid seams
      actual_top_gen = self._has_generated_context("top")
      actual_bottom_gen = self._has_generated_context("bottom")
      if actual_bottom_gen and not actual_top_gen:
        infill_y = 0
      elif actual_top_gen and not actual_bottom_gen:
        infill_y = margin_y
      else:
        infill_y = 0

    # Calculate world offset
    world_offset_x = self.region.x - infill_x
    world_offset_y = self.region.y - infill_y

    placement = TemplatePlacement(
      infill_x=infill_x,
      infill_y=infill_y,
      world_offset_x=world_offset_x,
      world_offset_y=world_offset_y,
    )
    placement._infill_width = self.region.width
    placement._infill_height = self.region.height

    # Validate the placement (seams check)
    is_valid, error = self._validate_placement_seams(placement)
    if not is_valid:
      self._last_validation_error = error
      return None

    return placement

  def _try_alternative_placements(
    self,
    missing: list[tuple[int, int]],
    allow_expansion: bool,
  ) -> TemplatePlacement | None:
    """
    Try alternative placements that avoid missing context quadrants.

    When the optimal placement would include non-generated context quadrants,
    we try placements that exclude certain sides to avoid those quadrants.
    """
    # Determine which sides are causing problems
    # Missing quadrants are in certain positions relative to the infill
    infill_quadrants = set(self.region.overlapping_quadrants())
    infill_min_qx = min(q[0] for q in infill_quadrants)
    infill_max_qx = max(q[0] for q in infill_quadrants)
    infill_min_qy = min(q[1] for q in infill_quadrants)
    infill_max_qy = max(q[1] for q in infill_quadrants)

    problem_sides = set()
    for qx, qy in missing:
      if qx < infill_min_qx:
        problem_sides.add("left")
      if qx > infill_max_qx:
        problem_sides.add("right")
      if qy < infill_min_qy:
        problem_sides.add("top")
      if qy > infill_max_qy:
        problem_sides.add("bottom")

    # Try placements that exclude problem sides
    # Generate combinations of sides to exclude
    side_combinations = []

    # First, try excluding just the problem sides
    if problem_sides:
      side_combinations.append(problem_sides)

    # Then try excluding individual problem sides
    for side in problem_sides:
      side_combinations.append({side})

    # Try each combination
    for exclude_sides in side_combinations:
      placement = self._try_placement_with_context_preferences(
        include_left="left" not in exclude_sides,
        include_right="right" not in exclude_sides,
        include_top="top" not in exclude_sides,
        include_bottom="bottom" not in exclude_sides,
      )

      if placement is None:
        continue

      # Check if this placement has any missing context quadrants
      new_missing = self._find_missing_context_quadrants(placement)

      if not new_missing:
        # Found a valid placement!
        return placement

      # If allow_expansion and fewer missing quadrants, try expansion
      if allow_expansion and len(new_missing) < len(missing):
        expanded = self._expand_to_cover_missing(placement, new_missing)
        if expanded is not None:
          return expanded

    # Last resort: try placements that sacrifice context to avoid missing quadrants
    # This may create seams but is better than not being able to generate at all
    best_placement = self._try_seam_tolerant_placement(problem_sides)
    if best_placement is not None:
      return best_placement

    return None

  def _try_seam_tolerant_placement(
    self, problem_sides: set[str]
  ) -> TemplatePlacement | None:
    """
    Try to find a placement that avoids missing context quadrants,
    even if it might create seams with generated neighbors.

    This is a last resort when no seam-free placement exists.
    """
    margin_x = TEMPLATE_SIZE - self.region.width
    margin_y = TEMPLATE_SIZE - self.region.height

    # Determine which sides have generated content
    has_left_gen = self._has_generated_context("left")
    has_right_gen = self._has_generated_context("right")
    has_top_gen = self._has_generated_context("top")
    has_bottom_gen = self._has_generated_context("bottom")

    # For each problem side, we want to position to EXCLUDE that side's context
    # even if it means creating a seam

    # Try different positions that avoid problem sides
    positions_to_try = []

    # If left is problematic, push infill to left (exclude left context)
    if "left" in problem_sides:
      # Position infill at left, include right context if available
      infill_x = 0
      if has_top_gen and "top" not in problem_sides:
        infill_y = margin_y  # Include top
      elif has_bottom_gen and "bottom" not in problem_sides:
        infill_y = 0  # Include bottom
      else:
        infill_y = 0
      positions_to_try.append((infill_x, infill_y))

    # If right is problematic, push infill to right (exclude right context)
    if "right" in problem_sides:
      infill_x = margin_x
      if has_top_gen and "top" not in problem_sides:
        infill_y = margin_y
      elif has_bottom_gen and "bottom" not in problem_sides:
        infill_y = 0
      else:
        infill_y = 0
      positions_to_try.append((infill_x, infill_y))

    # If top is problematic, push infill to top (exclude top context)
    if "top" in problem_sides:
      infill_y = 0
      if has_right_gen and "right" not in problem_sides:
        infill_x = 0  # Include right
      elif has_left_gen and "left" not in problem_sides:
        infill_x = margin_x  # Include left
      else:
        infill_x = 0
      positions_to_try.append((infill_x, infill_y))

    # If bottom is problematic, push infill to bottom
    if "bottom" in problem_sides:
      infill_y = margin_y
      if has_right_gen and "right" not in problem_sides:
        infill_x = 0
      elif has_left_gen and "left" not in problem_sides:
        infill_x = margin_x
      else:
        infill_x = 0
      positions_to_try.append((infill_x, infill_y))

    # Also try all four corners to maximize options
    corners = [
      (0, 0),  # Top-left
      (margin_x, 0),  # Top-right
      (0, margin_y),  # Bottom-left
      (margin_x, margin_y),  # Bottom-right
    ]
    positions_to_try.extend(corners)

    # Try each position
    for infill_x, infill_y in positions_to_try:
      world_offset_x = self.region.x - infill_x
      world_offset_y = self.region.y - infill_y

      placement = TemplatePlacement(
        infill_x=infill_x,
        infill_y=infill_y,
        world_offset_x=world_offset_x,
        world_offset_y=world_offset_y,
      )
      placement._infill_width = self.region.width
      placement._infill_height = self.region.height

      # Check for missing context quadrants (skip seam check)
      missing = self._find_missing_context_quadrants(placement)

      if not missing:
        # Found a valid placement (may have seams but no missing context)
        return placement

    return None

  def _has_generated_context(self, side: str) -> bool:
    """Check if there are generated pixels adjacent to the infill region on the given side."""
    # Note: Python's // does floor division, so negative coords work correctly
    # e.g., -1 // 512 = -1, -512 // 512 = -1, -513 // 512 = -2

    if side == "left":
      # Check quadrants to the left of the region
      check_x = self.region.x - 1
      qx = check_x // QUADRANT_SIZE
      # Check all quadrants along the left edge
      start_qy = self.region.y // QUADRANT_SIZE
      end_qy = (self.region.bottom - 1) // QUADRANT_SIZE
      return any(self.has_generation(qx, qy) for qy in range(start_qy, end_qy + 1))

    elif side == "right":
      check_x = self.region.right
      qx = check_x // QUADRANT_SIZE
      start_qy = self.region.y // QUADRANT_SIZE
      end_qy = (self.region.bottom - 1) // QUADRANT_SIZE
      return any(self.has_generation(qx, qy) for qy in range(start_qy, end_qy + 1))

    elif side == "top":
      check_y = self.region.y - 1
      qy = check_y // QUADRANT_SIZE
      start_qx = self.region.x // QUADRANT_SIZE
      end_qx = (self.region.right - 1) // QUADRANT_SIZE
      return any(self.has_generation(qx, qy) for qx in range(start_qx, end_qx + 1))

    elif side == "bottom":
      check_y = self.region.bottom
      qy = check_y // QUADRANT_SIZE
      start_qx = self.region.x // QUADRANT_SIZE
      end_qx = (self.region.right - 1) // QUADRANT_SIZE
      return any(self.has_generation(qx, qy) for qx in range(start_qx, end_qx + 1))

    return False

  def _validate_placement_seams(self, placement: TemplatePlacement) -> tuple[bool, str]:
    """
    Validate that a placement doesn't create seams.

    A seam would occur if the infill region touches the template edge
    AND there are generated pixels beyond that edge.

    Returns:
      Tuple of (is_valid, error_message)
    """
    # Check left edge
    if placement.infill_x == 0:
      if self._has_generated_context("left"):
        return False, "Would create seam with generated pixels on left"

    # Check right edge
    if placement.infill_x + self.region.width == TEMPLATE_SIZE:
      if self._has_generated_context("right"):
        return False, "Would create seam with generated pixels on right"

    # Check top edge
    if placement.infill_y == 0:
      if self._has_generated_context("top"):
        return False, "Would create seam with generated pixels on top"

    # Check bottom edge
    if placement.infill_y + self.region.height == TEMPLATE_SIZE:
      if self._has_generated_context("bottom"):
        return False, "Would create seam with generated pixels on bottom"

    return True, ""

  def _find_missing_context_quadrants(
    self, placement: TemplatePlacement
  ) -> list[tuple[int, int]]:
    """
    Find context quadrants that don't have generated pixels.

    Returns list of (qx, qy) positions that are in the template but not
    in the infill region and don't have generations.
    """
    missing = []

    # Calculate which quadrants the template covers
    template_world_left = placement.world_offset_x
    template_world_right = placement.world_offset_x + TEMPLATE_SIZE
    template_world_top = placement.world_offset_y
    template_world_bottom = placement.world_offset_y + TEMPLATE_SIZE

    start_qx = template_world_left // QUADRANT_SIZE
    end_qx = (template_world_right - 1) // QUADRANT_SIZE
    start_qy = template_world_top // QUADRANT_SIZE
    end_qy = (template_world_bottom - 1) // QUADRANT_SIZE

    # Infill quadrants
    infill_quadrants = set(self.region.overlapping_quadrants())

    for qx in range(start_qx, end_qx + 1):
      for qy in range(start_qy, end_qy + 1):
        if (qx, qy) not in infill_quadrants:
          # This is a context quadrant - must have generation
          if not self.has_generation(qx, qy):
            missing.append((qx, qy))

    return missing

  def _expand_to_cover_missing(
    self,
    placement: TemplatePlacement,
    missing: list[tuple[int, int]],
  ) -> TemplatePlacement | None:
    """
    Try to expand the infill region to cover missing context quadrants.

    The expanded region must still fit within the template and not exceed
    the maximum allowed size.

    Returns:
      New TemplatePlacement with expanded infill, or None if expansion not possible
    """
    # Get current infill quadrants
    primary_quadrants = self.region.overlapping_quadrants()

    # Combine primary and missing to get all quadrants we need to cover
    all_quadrants = set(primary_quadrants + missing)

    # Find bounds of expanded region
    min_qx = min(q[0] for q in all_quadrants)
    max_qx = max(q[0] for q in all_quadrants)
    min_qy = min(q[1] for q in all_quadrants)
    max_qy = max(q[1] for q in all_quadrants)

    # Create expanded region
    expanded_region = InfillRegion(
      x=min_qx * QUADRANT_SIZE,
      y=min_qy * QUADRANT_SIZE,
      width=(max_qx - min_qx + 1) * QUADRANT_SIZE,
      height=(max_qy - min_qy + 1) * QUADRANT_SIZE,
    )

    # Check if expanded region is valid size
    if not expanded_region.is_valid_size():
      self._last_validation_error = (
        f"Cannot expand infill to cover missing quadrants: "
        f"expanded region would be {expanded_region.area} pixels "
        f"(max: {MAX_INFILL_AREA})"
      )
      return None

    # Create a new builder for the expanded region to find its placement
    expanded_builder = TemplateBuilder(expanded_region, self.has_generation)
    expanded_placement = expanded_builder.find_optimal_placement(allow_expansion=False)

    if expanded_placement is None:
      self._last_validation_error = expanded_builder._last_validation_error
      return None

    # Track primary vs padding quadrants
    expanded_placement._primary_quadrants = list(primary_quadrants)
    expanded_placement._padding_quadrants = list(missing)
    expanded_placement._expanded_region = expanded_region

    return expanded_placement

  def build(
    self,
    border_width: int = 2,
    allow_expansion: bool = False,
  ) -> tuple[Image.Image, TemplatePlacement] | None:
    """
    Build the template image.

    Args:
      border_width: Width of the red border around the infill region
      allow_expansion: If True, automatically expand infill region to cover
                       missing context quadrants

    Returns:
      Tuple of (template_image, placement) or None if no valid placement exists
    """
    if self.get_render is None or self.get_generation is None:
      raise ValueError("get_render and get_generation must be provided to build")

    placement = self.find_optimal_placement(allow_expansion=allow_expansion)
    if placement is None:
      return None

    # Determine the effective infill region (may be expanded)
    if placement._expanded_region is not None:
      effective_region = placement._expanded_region
    else:
      effective_region = self.region

    # Create template image
    template = Image.new("RGBA", (TEMPLATE_SIZE, TEMPLATE_SIZE), (0, 0, 0, 0))

    # Determine which quadrants we need to fetch
    # The template covers world coordinates:
    #   x: [world_offset_x, world_offset_x + 1024)
    #   y: [world_offset_y, world_offset_y + 1024)

    template_world_left = placement.world_offset_x
    template_world_right = placement.world_offset_x + TEMPLATE_SIZE
    template_world_top = placement.world_offset_y
    template_world_bottom = placement.world_offset_y + TEMPLATE_SIZE

    # Find all quadrants that overlap with the template
    start_qx = template_world_left // QUADRANT_SIZE
    end_qx = (template_world_right - 1) // QUADRANT_SIZE
    start_qy = template_world_top // QUADRANT_SIZE
    end_qy = (template_world_bottom - 1) // QUADRANT_SIZE

    # Infill quadrants (will use render) - use effective region for expanded infills
    infill_quadrants = set(effective_region.overlapping_quadrants())

    # Fill in the template
    for qx in range(start_qx, end_qx + 1):
      for qy in range(start_qy, end_qy + 1):
        # Calculate where this quadrant appears in the template
        quad_world_x = qx * QUADRANT_SIZE
        quad_world_y = qy * QUADRANT_SIZE

        # Position in template coordinates
        template_x = quad_world_x - template_world_left
        template_y = quad_world_y - template_world_top

        # Determine source image
        if (qx, qy) in infill_quadrants:
          # Use render for infill quadrants
          quad_img = self.get_render(qx, qy)
          if quad_img is None:
            continue
        else:
          # Use generation for context quadrants
          quad_img = self.get_generation(qx, qy)
          if quad_img is None:
            continue

        if quad_img.mode != "RGBA":
          quad_img = quad_img.convert("RGBA")

        # Calculate crop region if quadrant extends outside template
        crop_left = max(0, -template_x)
        crop_top = max(0, -template_y)
        crop_right = min(QUADRANT_SIZE, TEMPLATE_SIZE - template_x)
        crop_bottom = min(QUADRANT_SIZE, TEMPLATE_SIZE - template_y)

        if crop_left < crop_right and crop_top < crop_bottom:
          cropped = quad_img.crop((crop_left, crop_top, crop_right, crop_bottom))
          paste_x = max(0, template_x)
          paste_y = max(0, template_y)
          template.paste(cropped, (paste_x, paste_y))

    # Now we need to handle partial quadrant overlaps with the infill region
    # If the infill region doesn't align with quadrant boundaries,
    # we need to carefully composite render pixels only in the infill area
    self._apply_infill_mask(template, placement)

    # Draw red border around infill region
    template = self._draw_border(template, placement, border_width)

    return template, placement

  def _apply_infill_mask(
    self, template: Image.Image, placement: TemplatePlacement
  ) -> None:
    """
    Apply a mask to ensure only the infill region has render pixels.

    For quadrants that partially overlap the infill region, we need to
    composite render pixels (infill area) with generation pixels (context area).
    """
    # This is already handled by the quadrant-based approach when infill aligns
    # with quadrant boundaries. For non-aligned infills, we'd need more complex
    # masking. For now, we assume quadrant-aligned infills.
    pass

  def _draw_border(
    self,
    template: Image.Image,
    placement: TemplatePlacement,
    border_width: int,
  ) -> Image.Image:
    """Draw a red border around the infill region."""
    result = template.copy()
    draw = ImageDraw.Draw(result)

    red = (255, 0, 0, 255)

    left = placement.infill_x
    top = placement.infill_y
    right = placement.infill_x + self.region.width
    bottom = placement.infill_y + self.region.height

    for i in range(border_width):
      draw.rectangle(
        [left + i, top + i, right - 1 - i, bottom - 1 - i],
        outline=red,
        fill=None,
      )

    return result

  def get_validation_info(self) -> dict:
    """Get detailed validation information for debugging."""
    return {
      "region": str(self.region),
      "area": self.region.area,
      "max_area": MAX_INFILL_AREA,
      "valid_size": self.region.is_valid_size(),
      "has_left_gen": self._has_generated_context("left"),
      "has_right_gen": self._has_generated_context("right"),
      "has_top_gen": self._has_generated_context("top"),
      "has_bottom_gen": self._has_generated_context("bottom"),
      "overlapping_quadrants": self.region.overlapping_quadrants(),
      "last_validation_error": self._last_validation_error,
    }


def validate_quadrant_selection(
  quadrants: list[tuple[int, int]],
  has_generation: Callable[[int, int], bool],
  allow_expansion: bool = False,
) -> tuple[bool, str, TemplatePlacement | None]:
  """
  Validate a quadrant selection and find optimal placement.

  This is a convenience function for the common case of selecting
  whole quadrants for infill.

  Args:
    quadrants: List of (qx, qy) quadrant positions to infill
    has_generation: Callable to check if a quadrant has generation
    allow_expansion: If True, automatically expand infill region to cover
                     missing context quadrants (they'll be filled with render
                     pixels and discarded after generation)

  Returns:
    Tuple of (is_valid, message, placement)
  """
  if not quadrants:
    return False, "No quadrants selected", None

  # Check that quadrants form a rectangle
  min_qx = min(q[0] for q in quadrants)
  max_qx = max(q[0] for q in quadrants)
  min_qy = min(q[1] for q in quadrants)
  max_qy = max(q[1] for q in quadrants)

  expected_count = (max_qx - min_qx + 1) * (max_qy - min_qy + 1)
  if len(quadrants) != expected_count:
    return False, "Quadrants must form a contiguous rectangle", None

  # Check all expected positions are present
  expected = set()
  for qx in range(min_qx, max_qx + 1):
    for qy in range(min_qy, max_qy + 1):
      expected.add((qx, qy))

  if set(quadrants) != expected:
    return False, "Quadrants must form a contiguous rectangle", None

  # Create infill region and builder
  region = InfillRegion.from_quadrants(quadrants)

  if not region.is_valid_size():
    return (
      False,
      f"Selection too large: {region.area} pixels (max: {MAX_INFILL_AREA} or full tile)",
      None,
    )

  # For full tiles, check that there are no generated neighbors
  if region.is_full_tile():
    # Check all edges for generated neighbors
    has_any_gen_neighbor = False
    for qx, qy in quadrants:
      # Check all 4 sides of each edge quadrant
      if qx == min(q[0] for q in quadrants):  # Left edge
        if has_generation(qx - 1, qy):
          has_any_gen_neighbor = True
          break
      if qx == max(q[0] for q in quadrants):  # Right edge
        if has_generation(qx + 1, qy):
          has_any_gen_neighbor = True
          break
      if qy == min(q[1] for q in quadrants):  # Top edge
        if has_generation(qx, qy - 1):
          has_any_gen_neighbor = True
          break
      if qy == max(q[1] for q in quadrants):  # Bottom edge
        if has_generation(qx, qy + 1):
          has_any_gen_neighbor = True
          break

    if has_any_gen_neighbor:
      return (
        False,
        "Full tile (2x2) selection cannot have generated neighbors (would create seams)",
        None,
      )

    # Full tile with no neighbors - valid, place at origin
    placement = TemplatePlacement(
      infill_x=0,
      infill_y=0,
      world_offset_x=region.x,
      world_offset_y=region.y,
    )
    placement._infill_width = region.width
    placement._infill_height = region.height
    placement._primary_quadrants = list(quadrants)
    return True, "Valid selection (full tile)", placement

  builder = TemplateBuilder(region, has_generation)
  placement = builder.find_optimal_placement(allow_expansion=allow_expansion)

  if placement is None:
    # Use the specific error from the builder if available
    info = builder.get_validation_info()
    if info["last_validation_error"]:
      return False, info["last_validation_error"], None
    # Fallback to generic messages
    if info["has_left_gen"]:
      return False, "Would create seam with generated pixels on left", None
    if info["has_right_gen"]:
      return False, "Would create seam with generated pixels on right", None
    if info["has_top_gen"]:
      return False, "Would create seam with generated pixels on top", None
    if info["has_bottom_gen"]:
      return False, "Would create seam with generated pixels on bottom", None
    return False, "No valid placement found", None

  # Set primary quadrants if not already set (by expansion)
  if not placement._primary_quadrants:
    placement._primary_quadrants = list(quadrants)

  # Build appropriate message
  if placement.is_expanded:
    padding_str = ", ".join(f"({qx}, {qy})" for qx, qy in placement._padding_quadrants)
    return True, f"Valid selection (expanded to cover: {padding_str})", placement

  return True, "Valid selection", placement


# =============================================================================
# Testing
# =============================================================================


def _test_basic():
  """Run basic tests."""
  print("=" * 60)
  print("Testing InfillRegion")
  print("=" * 60)

  # Test single quadrant
  r1 = InfillRegion.from_quadrant(0, 0)
  print(f"\nSingle quadrant (0,0): {r1}")
  print(f"  Area: {r1.area} (valid: {r1.is_valid_size()})")
  print(f"  Overlapping quadrants: {r1.overlapping_quadrants()}")

  # Test 2x1 quadrants
  r2 = InfillRegion.from_quadrants([(0, 0), (1, 0)])
  print(f"\n2x1 quadrants: {r2}")
  print(f"  Area: {r2.area} (valid: {r2.is_valid_size()})")
  print(f"  Overlapping quadrants: {r2.overlapping_quadrants()}")

  # Test 2x2 quadrants (should be invalid - 100% of tile)
  r3 = InfillRegion.from_quadrants([(0, 0), (1, 0), (0, 1), (1, 1)])
  print(f"\n2x2 quadrants: {r3}")
  print(f"  Area: {r3.area} (valid: {r3.is_valid_size()})")

  print("\n" + "=" * 60)
  print("Testing TemplateBuilder")
  print("=" * 60)

  # Create a mock has_generation function
  # Simulate: quadrants (0,0) and (1,0) are NOT generated, (0,1) and (1,1) ARE generated
  generated = {(0, 1), (1, 1), (2, 0), (2, 1)}

  def has_gen(qx, qy):
    return (qx, qy) in generated

  # Test 1: Select quadrant (1, 0) with generated neighbor below
  region = InfillRegion.from_quadrant(1, 0)
  builder = TemplateBuilder(region, has_gen)
  info = builder.get_validation_info()
  placement = builder.find_optimal_placement()

  print("\nTest 1: Select quadrant (1, 0)")
  print(f"  Info: {info}")
  print(f"  Placement: {placement}")
  if placement:
    print(f"    Infill at: ({placement.infill_x}, {placement.infill_y})")
    print(f"    World offset: ({placement.world_offset_x}, {placement.world_offset_y})")

  # Test 2: Select quadrant (0, 1) surrounded by generated
  region2 = InfillRegion.from_quadrant(0, 1)
  builder2 = TemplateBuilder(region2, has_gen)
  info2 = builder2.get_validation_info()
  placement2 = builder2.find_optimal_placement()

  print("\nTest 2: Select quadrant (0, 1) - has generated neighbor to right")
  print(f"  Info: {info2}")
  print(f"  Placement: {placement2}")
  if placement2:
    print(f"    Infill at: ({placement2.infill_x}, {placement2.infill_y})")

  # Test 3: Validation convenience function
  print("\n" + "=" * 60)
  print("Testing validate_quadrant_selection")
  print("=" * 60)

  # Valid selection
  valid, msg, p = validate_quadrant_selection([(1, 0)], has_gen)
  print(f"\nSelect (1,0): valid={valid}, msg='{msg}'")

  # Invalid - non-contiguous
  valid, msg, p = validate_quadrant_selection([(0, 0), (1, 1)], has_gen)
  print(f"Select (0,0), (1,1) [diagonal]: valid={valid}, msg='{msg}'")

  # Valid 2x1
  valid, msg, p = validate_quadrant_selection([(0, 0), (1, 0)], has_gen)
  print(f"Select (0,0), (1,0) [2x1]: valid={valid}, msg='{msg}'")

  print("\n" + "=" * 60)
  print("All tests complete!")
  print("=" * 60)


if __name__ == "__main__":
  _test_basic()
