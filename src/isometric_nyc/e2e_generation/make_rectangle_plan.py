"""
Rectangle generation plan algorithm.

Creates a sequence of generation steps for filling a rectangular region
of quadrants, respecting pre-existing generated quadrants and following
tile placement rules.

Tile Placement Rules:
- 2x2: No side of the tile may touch any previously generated quadrants.
- 2x1/1x2: Both quadrants of the 2-long side must touch previously generated
  quadrants along that axis on ONE side. Neither quadrant may touch previously
  generated quadrants along the transverse (short) side.
- 1x1: Ideally generated when 3 other quadrants in a 2x2 block are generated.

Algorithm:
1. Place as many 2x2 tiles as possible
2. Place 2x1/1x2 tiles for remaining gaps
3. Fill remaining single quadrants with 1x1 tiles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point:
  """A 2D point representing a quadrant coordinate."""

  x: int
  y: int

  def __str__(self) -> str:
    return f"({self.x},{self.y})"

  def __add__(self, other: Point) -> Point:
    return Point(self.x + other.x, self.y + other.y)

  @classmethod
  def from_string(cls, s: str) -> Point:
    """Parse a string like '(x,y)' or 'x,y' into a Point."""
    s = s.strip().replace("(", "").replace(")", "").replace(" ", "")
    parts = s.split(",")
    if len(parts) != 2:
      raise ValueError(f"Invalid coordinate format: {s}")
    return cls(int(parts[0]), int(parts[1]))

  def to_tuple(self) -> tuple[int, int]:
    """Convert to a tuple."""
    return (self.x, self.y)


@dataclass
class RectBounds:
  """Bounds of the rectangle to generate."""

  top_left: Point
  bottom_right: Point

  @property
  def width(self) -> int:
    """Width of the rectangle (x extent)."""
    return self.bottom_right.x - self.top_left.x + 1

  @property
  def height(self) -> int:
    """Height of the rectangle (y extent)."""
    return self.bottom_right.y - self.top_left.y + 1

  @property
  def area(self) -> int:
    """Total number of quadrants in the rectangle."""
    return self.width * self.height

  def contains(self, p: Point) -> bool:
    """Check if a point is within the rectangle bounds."""
    return (
      self.top_left.x <= p.x <= self.bottom_right.x
      and self.top_left.y <= p.y <= self.bottom_right.y
    )

  def all_points(self) -> list[Point]:
    """Return all points within the rectangle bounds."""
    return [
      Point(x, y)
      for y in range(self.top_left.y, self.bottom_right.y + 1)
      for x in range(self.top_left.x, self.bottom_right.x + 1)
    ]


@dataclass
class GenerationStep:
  """A single generation step containing quadrants to generate together."""

  quadrants: list[Point]
  step_type: str = ""  # "2x2", "2x1", "1x2", "1x1" for debugging

  def to_dict(self) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    return {
      "quadrants": [(q.x, q.y) for q in self.quadrants],
      "type": self.step_type,
    }


@dataclass
class RectanglePlan:
  """A complete generation plan for a rectangle."""

  bounds: RectBounds
  steps: list[GenerationStep] = field(default_factory=list)
  pre_generated: set[Point] = field(default_factory=set)

  def to_dict(self) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    return {
      "bounds": {
        "top_left": self.bounds.top_left.to_tuple(),
        "bottom_right": self.bounds.bottom_right.to_tuple(),
      },
      "steps": [step.to_dict() for step in self.steps],
      "pre_generated": [p.to_tuple() for p in self.pre_generated],
    }


# =============================================================================
# 2x2 Tile Placement
# =============================================================================


def get_2x2_quadrants(top_left: Point) -> list[Point]:
  """Get the 4 quadrants of a 2x2 tile given its top-left corner."""
  x, y = top_left.x, top_left.y
  return [
    Point(x, y),
    Point(x + 1, y),
    Point(x, y + 1),
    Point(x + 1, y + 1),
  ]


def get_2x2_neighbors(top_left: Point) -> list[Point]:
  """
  Get all 8 exterior neighbors of a 2x2 tile.

  For a 2x2 at (x,y), (x+1,y), (x,y+1), (x+1,y+1):
  - Top: (x, y-1), (x+1, y-1)
  - Bottom: (x, y+2), (x+1, y+2)
  - Left: (x-1, y), (x-1, y+1)
  - Right: (x+2, y), (x+2, y+1)
  """
  x, y = top_left.x, top_left.y
  return [
    # Top
    Point(x, y - 1),
    Point(x + 1, y - 1),
    # Bottom
    Point(x, y + 2),
    Point(x + 1, y + 2),
    # Left
    Point(x - 1, y),
    Point(x - 1, y + 1),
    # Right
    Point(x + 2, y),
    Point(x + 2, y + 1),
  ]


def can_place_2x2(
  top_left: Point,
  bounds: RectBounds,
  generated: set[Point],
  scheduled: set[Point],
  allow_adjacent_scheduled: bool = False,
) -> bool:
  """
  Check if a 2x2 tile can be placed at the given top-left position.

  Rules:
  - All 4 quadrants must be within bounds
  - All 4 quadrants must not be already generated or scheduled
  - No neighbor of the 2x2 may be pre-generated (seam prevention)
  - If allow_adjacent_scheduled is False, no neighbor may be scheduled either

  Args:
      top_left: Top-left corner of the 2x2 tile
      bounds: Rectangle bounds
      generated: Pre-existing generated quadrants (must avoid adjacency)
      scheduled: Quadrants scheduled in this plan
      allow_adjacent_scheduled: If True, allow placement next to scheduled tiles.
                               This enables dense 2x2 packing for rectangle filling.
  """
  quadrants = get_2x2_quadrants(top_left)

  # All quadrants must be within bounds and unscheduled
  for q in quadrants:
    if not bounds.contains(q):
      return False
    if q in generated or q in scheduled:
      return False

  # No neighbor may be pre-generated (seam prevention)
  neighbors = get_2x2_neighbors(top_left)
  for n in neighbors:
    if n in generated:
      return False
    # Only check scheduled if we're not allowing adjacent scheduled tiles
    if not allow_adjacent_scheduled and n in scheduled:
      return False

  return True


def find_all_valid_2x2_positions(
  bounds: RectBounds,
  generated: set[Point],
  scheduled: set[Point],
  allow_adjacent_scheduled: bool = False,
) -> list[Point]:
  """Find all valid top-left positions for 2x2 tiles."""
  valid = []
  # Check all possible 2x2 positions (need room for 2x2)
  for y in range(bounds.top_left.y, bounds.bottom_right.y):  # -1 for 2x2
    for x in range(bounds.top_left.x, bounds.bottom_right.x):
      tl = Point(x, y)
      if can_place_2x2(tl, bounds, generated, scheduled, allow_adjacent_scheduled):
        valid.append(tl)
  return valid


def place_2x2_tiles(
  bounds: RectBounds,
  generated: set[Point],
) -> tuple[list[GenerationStep], set[Point]]:
  """
  Place as many 2x2 tiles as possible.

  Returns the generation steps and the set of scheduled quadrants.

  Strategy:
  1. First pass: Place 2x2 tiles with gaps to avoid pre-generated neighbors
  2. Second pass: Fill remaining 2x2-sized gaps (allow adjacent to scheduled)

  This allows dense packing when filling empty rectangles while still
  avoiding seams with pre-generated content.
  """
  steps: list[GenerationStep] = []
  scheduled: set[Point] = set()

  # First pass: Place 2x2 tiles avoiding both generated AND scheduled neighbors
  # This creates a pattern with gaps for bridging
  while True:
    valid_positions = find_all_valid_2x2_positions(
      bounds, generated, scheduled, allow_adjacent_scheduled=False
    )
    if not valid_positions:
      break

    tl = valid_positions[0]
    quadrants = get_2x2_quadrants(tl)
    steps.append(GenerationStep(quadrants=quadrants, step_type="2x2"))
    for q in quadrants:
      scheduled.add(q)

  # Second pass: Fill remaining 2x2-sized gaps by allowing adjacent to scheduled
  # This helps cover areas that can't be reached by bridges
  while True:
    valid_positions = find_all_valid_2x2_positions(
      bounds, generated, scheduled, allow_adjacent_scheduled=True
    )
    if not valid_positions:
      break

    tl = valid_positions[0]
    quadrants = get_2x2_quadrants(tl)
    steps.append(GenerationStep(quadrants=quadrants, step_type="2x2"))
    for q in quadrants:
      scheduled.add(q)

  return steps, scheduled


# =============================================================================
# 2x1 / 1x2 Tile Placement
# =============================================================================


def can_place_2x1_horizontal(
  left: Point,
  bounds: RectBounds,
  generated: set[Point],
  scheduled: set[Point],
) -> bool:
  """
  Check if a horizontal 2x1 tile can be placed at (left.x, left.y) and (left.x+1, left.y).

  Rules (following task 019 strip plan):
  - Both quadrants must be within bounds and unscheduled
  - At least one of top/bottom side must have BOTH neighbors generated
    (can be both sides, like when bridging between two 2x2 tiles)
  - Neither left nor right neighbor may be generated (transverse sides)
  """
  right = Point(left.x + 1, left.y)

  # Both must be in bounds and not already covered
  for q in [left, right]:
    if not bounds.contains(q):
      return False
    if q in generated or q in scheduled:
      return False

  # Check transverse (short) sides - must NOT be generated
  left_neighbor = Point(left.x - 1, left.y)
  right_neighbor = Point(right.x + 1, right.y)
  if left_neighbor in generated or left_neighbor in scheduled:
    return False
  if right_neighbor in generated or right_neighbor in scheduled:
    return False

  # Check long sides - AT LEAST ONE side must have BOTH neighbors generated
  # Note: Both sides CAN be generated (e.g., bridging between two 2x2 tiles)
  top_left = Point(left.x, left.y - 1)
  top_right = Point(right.x, right.y - 1)
  bottom_left = Point(left.x, left.y + 1)
  bottom_right = Point(right.x, right.y + 1)

  # "Generated" for the purpose of long-side check includes both
  # pre-existing generated AND scheduled quadrants
  combined = generated | scheduled

  top_both_generated = top_left in combined and top_right in combined
  bottom_both_generated = bottom_left in combined and bottom_right in combined

  # At least one side must be fully generated
  return top_both_generated or bottom_both_generated


def can_place_1x2_vertical(
  top: Point,
  bounds: RectBounds,
  generated: set[Point],
  scheduled: set[Point],
) -> bool:
  """
  Check if a vertical 1x2 tile can be placed at (top.x, top.y) and (top.x, top.y+1).

  Rules (following task 019 strip plan):
  - Both quadrants must be within bounds and unscheduled
  - At least one of left/right side must have BOTH neighbors generated
    (can be both sides, like when bridging between two 2x2 tiles)
  - Neither top nor bottom neighbor may be generated (transverse sides)
  """
  bottom = Point(top.x, top.y + 1)

  # Both must be in bounds and not already covered
  for q in [top, bottom]:
    if not bounds.contains(q):
      return False
    if q in generated or q in scheduled:
      return False

  # Check transverse (short) sides - must NOT be generated
  top_neighbor = Point(top.x, top.y - 1)
  bottom_neighbor = Point(bottom.x, bottom.y + 1)
  if top_neighbor in generated or top_neighbor in scheduled:
    return False
  if bottom_neighbor in generated or bottom_neighbor in scheduled:
    return False

  # Check long sides - AT LEAST ONE side must have BOTH neighbors generated
  # Note: Both sides CAN be generated (e.g., bridging between two 2x2 tiles)
  left_top = Point(top.x - 1, top.y)
  left_bottom = Point(top.x - 1, bottom.y)
  right_top = Point(top.x + 1, top.y)
  right_bottom = Point(top.x + 1, bottom.y)

  combined = generated | scheduled

  left_both_generated = left_top in combined and left_bottom in combined
  right_both_generated = right_top in combined and right_bottom in combined

  # At least one side must be fully generated
  return left_both_generated or right_both_generated


def find_all_valid_2x1_positions(
  bounds: RectBounds,
  generated: set[Point],
  scheduled: set[Point],
) -> list[tuple[Point, str]]:
  """
  Find all valid positions for 2x1 (horizontal) and 1x2 (vertical) tiles.

  Returns list of (position, type) where type is "2x1" or "1x2".

  Following task 019 strip plan pattern:
  - Vertical bridges (1x2) are processed first to connect 2x2 tiles
  - Horizontal bridges (2x1) are processed second to connect to the edge
  """
  valid: list[tuple[Point, str]] = []

  # Check vertical 1x2 tiles FIRST (bridges between 2x2 tiles)
  for y in range(bounds.top_left.y, bounds.bottom_right.y):  # -1 for 2 tall
    for x in range(bounds.top_left.x, bounds.bottom_right.x + 1):
      top = Point(x, y)
      if can_place_1x2_vertical(top, bounds, generated, scheduled):
        valid.append((top, "1x2"))

  # Check horizontal 2x1 tiles SECOND (connect to generation edge)
  for y in range(bounds.top_left.y, bounds.bottom_right.y + 1):
    for x in range(bounds.top_left.x, bounds.bottom_right.x):  # -1 for 2 wide
      left = Point(x, y)
      if can_place_2x1_horizontal(left, bounds, generated, scheduled):
        valid.append((left, "2x1"))

  return valid


def place_2x1_tiles(
  bounds: RectBounds,
  generated: set[Point],
  scheduled: set[Point],
) -> tuple[list[GenerationStep], set[Point]]:
  """
  Place 2x1 and 1x2 tiles where possible.

  These tiles bridge gaps between 2x2 tiles and the generation edge.
  """
  steps: list[GenerationStep] = []
  new_scheduled = set(scheduled)

  # Keep placing tiles until no more can be placed
  while True:
    valid_positions = find_all_valid_2x1_positions(bounds, generated, new_scheduled)
    if not valid_positions:
      break

    # Pick the first valid position
    pos, tile_type = valid_positions[0]

    if tile_type == "2x1":
      quadrants = [pos, Point(pos.x + 1, pos.y)]
    else:  # 1x2
      quadrants = [pos, Point(pos.x, pos.y + 1)]

    steps.append(GenerationStep(quadrants=quadrants, step_type=tile_type))
    for q in quadrants:
      new_scheduled.add(q)

  return steps, new_scheduled


# =============================================================================
# 1x1 Tile Placement
# =============================================================================


def get_2x2_block_positions(p: Point) -> list[list[Point]]:
  """
  Get all 2x2 blocks that contain the point p.

  A point can be in up to 4 different 2x2 blocks:
  - As top-left
  - As top-right
  - As bottom-left
  - As bottom-right
  """
  blocks = []
  # p as top-left
  blocks.append([p, Point(p.x + 1, p.y), Point(p.x, p.y + 1), Point(p.x + 1, p.y + 1)])
  # p as top-right
  blocks.append([Point(p.x - 1, p.y), p, Point(p.x - 1, p.y + 1), Point(p.x, p.y + 1)])
  # p as bottom-left
  blocks.append([Point(p.x, p.y - 1), Point(p.x + 1, p.y - 1), p, Point(p.x + 1, p.y)])
  # p as bottom-right
  blocks.append([Point(p.x - 1, p.y - 1), Point(p.x, p.y - 1), Point(p.x - 1, p.y), p])
  return blocks


def count_generated_in_block(block: list[Point], combined: set[Point]) -> int:
  """Count how many quadrants in a 2x2 block are generated/scheduled."""
  return sum(1 for p in block if p in combined)


def has_valid_2x2_context(
  quadrants: list[Point],
  combined: set[Point],
) -> bool:
  """
  Check if the given quadrants have valid 2x2 context for generation.

  For a generation to succeed, there must exist at least one 2x2 block
  where all 4 quadrants are either:
  - Part of the generation (in quadrants list)
  - Already generated/scheduled (in combined set)

  Args:
      quadrants: The quadrants being generated
      combined: Set of already generated or scheduled quadrants

  Returns:
      True if there's at least one valid 2x2 block providing full context
  """
  quadrant_set = set(quadrants)

  # For each quadrant, check all 2x2 blocks it belongs to
  for q in quadrants:
    blocks = get_2x2_block_positions(q)
    for block in blocks:
      # Check if all 4 positions in this block are covered
      all_covered = all(p in quadrant_set or p in combined for p in block)
      if all_covered:
        return True

  return False


def can_place_1x1(p: Point, combined: set[Point]) -> bool:
  """
  Check if a 1x1 tile can be placed at point p with valid context.

  A 1x1 tile requires that 3 of 4 quadrants in at least one 2x2 block
  containing p are already generated/scheduled (providing context).
  """
  blocks = get_2x2_block_positions(p)
  for block in blocks:
    # Count how many OTHER quadrants (not p) are generated
    other_generated = sum(1 for q in block if q != p and q in combined)
    if other_generated >= 3:
      return True
  return False


def place_1x1_tiles(
  bounds: RectBounds,
  generated: set[Point],
  scheduled: set[Point],
) -> list[GenerationStep]:
  """
  Fill remaining gaps with 1x1 tiles.

  A 1x1 tile can only be placed if 3 of 4 quadrants in at least one
  2x2 block containing it are already generated/scheduled.
  This ensures valid context for the generation.
  """
  steps: list[GenerationStep] = []
  combined = generated | scheduled
  new_scheduled = set(scheduled)

  # Find all remaining unscheduled quadrants within bounds
  remaining = [p for p in bounds.all_points() if p not in combined]

  # Sort by priority: quadrants with more generated neighbors in 2x2 blocks first
  # This helps ensure we place tiles in an order that maintains context
  def priority(p: Point) -> int:
    blocks = get_2x2_block_positions(p)
    max_generated = max(
      count_generated_in_block(block, combined | new_scheduled) for block in blocks
    )
    return -max_generated  # Negative for descending sort

  remaining.sort(key=priority)

  # Keep iterating until no more valid placements
  # (as each placement may enable new valid placements)
  changed = True
  while changed:
    changed = False
    for p in list(remaining):
      if p in new_scheduled:
        remaining.remove(p)
        continue

      # Check if this 1x1 has valid context
      if can_place_1x1(p, combined | new_scheduled):
        steps.append(GenerationStep(quadrants=[p], step_type="1x1"))
        new_scheduled.add(p)
        remaining.remove(p)
        changed = True

  return steps


# =============================================================================
# Main Algorithm
# =============================================================================


def create_rectangle_plan(
  bounds: RectBounds,
  generated: set[Point] | None = None,
  queued: set[Point] | None = None,
) -> RectanglePlan:
  """
  Create a generation plan for filling a rectangle.

  Args:
      bounds: The rectangle bounds (top-left to bottom-right inclusive)
      generated: Set of already-generated quadrant positions
      queued: Set of quadrant positions that are in-progress or queued for generation.
              These are treated as "will be generated" for seam detection purposes,
              meaning tiles cannot be placed adjacent to them (to avoid seams).

  Returns:
      RectanglePlan with the sequence of generation steps
  """
  generated = generated or set()
  queued = queued or set()

  # Combine generated and queued for seam detection
  # Queued quadrants are treated as if they will be generated,
  # so we cannot place tiles adjacent to them
  effective_generated = generated | queued

  # Filter out quadrants that are already generated OR queued from the rectangle
  points_to_generate = set(bounds.all_points()) - effective_generated

  if not points_to_generate:
    return RectanglePlan(bounds=bounds, steps=[], pre_generated=generated)

  # Phase 1: Place 2x2 tiles
  steps_2x2, scheduled = place_2x2_tiles(bounds, effective_generated)

  # Phase 2: Place 2x1/1x2 tiles
  steps_2x1, scheduled = place_2x1_tiles(bounds, effective_generated, scheduled)

  # Phase 3: Fill with 1x1 tiles
  steps_1x1 = place_1x1_tiles(bounds, effective_generated, scheduled)

  all_steps = steps_2x2 + steps_2x1 + steps_1x1

  return RectanglePlan(
    bounds=bounds,
    steps=all_steps,
    pre_generated=generated,
  )


def create_rectangle_plan_from_coords(
  tl: tuple[int, int],
  br: tuple[int, int],
  generated: set[tuple[int, int]] | None = None,
  queued: set[tuple[int, int]] | None = None,
) -> RectanglePlan:
  """
  Convenience function to create a plan from coordinate tuples.

  Args:
      tl: Top-left corner (x, y)
      br: Bottom-right corner (x, y)
      generated: Set of already-generated quadrant positions as (x, y) tuples
      queued: Set of quadrant positions that are in-progress or queued for generation
              as (x, y) tuples. These are treated as "will be generated" for seam
              detection purposes.

  Returns:
      RectanglePlan with the sequence of generation steps
  """
  bounds = RectBounds(Point(tl[0], tl[1]), Point(br[0], br[1]))
  gen_points = {Point(x, y) for x, y in (generated or set())}
  queued_points = {Point(x, y) for x, y in (queued or set())}
  return create_rectangle_plan(bounds, gen_points, queued_points)


# =============================================================================
# Validation
# =============================================================================


def validate_plan(plan: RectanglePlan) -> tuple[bool, list[str]]:
  """
  Validate that a plan covers all required quadrants exactly once.

  Returns (is_valid, error_messages).
  """
  errors: list[str] = []

  # Check that all required quadrants are covered
  required = set(plan.bounds.all_points()) - plan.pre_generated
  covered: set[Point] = set()

  for step in plan.steps:
    for q in step.quadrants:
      if q in covered:
        errors.append(f"Quadrant {q} is covered multiple times")
      covered.add(q)

  missing = required - covered
  if missing:
    errors.append(f"Missing quadrants: {sorted(missing, key=lambda p: (p.y, p.x))}")

  extra = covered - required
  if extra:
    errors.append(f"Extra quadrants: {sorted(extra, key=lambda p: (p.y, p.x))}")

  return len(errors) == 0, errors


def validate_plan_context(plan: RectanglePlan) -> tuple[bool, list[str]]:
  """
  Validate that all steps in a plan have valid 2x2 context for generation.

  Each generation step must have at least one 2x2 block where all 4 quadrants
  are either part of the step or already generated/scheduled.

  Returns (is_valid, error_messages).
  """
  errors: list[str] = []

  # Start with pre-generated quadrants as context
  generated_so_far = set(plan.pre_generated)

  for i, step in enumerate(plan.steps):
    # Check if this step has valid context
    if not has_valid_2x2_context(step.quadrants, generated_so_far):
      quadrant_strs = ", ".join(str(q) for q in step.quadrants)
      errors.append(
        f"Step {i + 1} ({step.step_type}): [{quadrant_strs}] lacks valid 2x2 context"
      )

    # Add this step's quadrants to the generated set for subsequent steps
    for q in step.quadrants:
      generated_so_far.add(q)

  return len(errors) == 0, errors


def get_plan_summary(plan: RectanglePlan) -> dict[str, Any]:
  """Get a summary of the plan for display."""
  by_type: dict[str, int] = {}
  for step in plan.steps:
    t = step.step_type or f"{len(step.quadrants)}-quad"
    by_type[t] = by_type.get(t, 0) + 1

  total_quadrants = sum(len(step.quadrants) for step in plan.steps)

  return {
    "bounds": {
      "tl": plan.bounds.top_left.to_tuple(),
      "br": plan.bounds.bottom_right.to_tuple(),
      "width": plan.bounds.width,
      "height": plan.bounds.height,
    },
    "pre_generated_count": len(plan.pre_generated),
    "total_steps": len(plan.steps),
    "total_quadrants": total_quadrants,
    "steps_by_type": by_type,
  }
