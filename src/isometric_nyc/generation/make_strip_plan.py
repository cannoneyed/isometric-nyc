"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0

  # For negative coordinates, use = and quotes:
  uv run python src/isometric_nyc/generation/make_strip_plan.py \\
    generations/nyc \\
    --tl='-12,-1' \\
    --br='-10,7'
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Data Structures
# =============================================================================


class Edge(Enum):
  """Possible generation edges."""

  TOP = "top"  # Edge is at the top (y = top_left.y - 1)
  BOTTOM = "bottom"  # Edge is at the bottom (y = bottom_right.y + 1)
  LEFT = "left"  # Edge is on the left (x = top_left.x - 1)
  RIGHT = "right"  # Edge is on the right (x = bottom_right.x + 1)


class StepStatus(Enum):
  """Status of a generation step."""

  PENDING = "pending"
  DONE = "done"
  ERROR = "error"


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


@dataclass
class StripBounds:
  """Bounds of the strip to generate."""

  top_left: Point
  bottom_right: Point

  @property
  def width(self) -> int:
    """Width of the strip (x extent)."""
    return self.bottom_right.x - self.top_left.x + 1

  @property
  def height(self) -> int:
    """Height of the strip (y extent)."""
    return self.bottom_right.y - self.top_left.y + 1

  @property
  def is_horizontal(self) -> bool:
    """True if width >= height."""
    return self.width >= self.height

  @property
  def depth(self) -> int:
    """Depth of the strip (perpendicular to progress direction)."""
    return self.height if self.is_horizontal else self.width

  @property
  def length(self) -> int:
    """Length of the strip (along progress direction)."""
    return self.width if self.is_horizontal else self.height

  def all_points(self) -> list[Point]:
    """Return all points within the strip bounds."""
    return [
      Point(x, y)
      for y in range(self.top_left.y, self.bottom_right.y + 1)
      for x in range(self.top_left.x, self.bottom_right.x + 1)
    ]


@dataclass
class GenerationStep:
  """A single generation step."""

  quadrants: list[Point]
  status: StepStatus = StepStatus.PENDING

  def to_dict(self) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    quadrants_str = ",".join(str(q) for q in self.quadrants)
    return {
      "quadrants": quadrants_str,
      "status": self.status.value,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> GenerationStep:
    """Create from JSON dict."""
    quadrants_str = data["quadrants"]
    quadrants = [
      Point.from_string(q)
      for q in quadrants_str.split("),(")
      if q.replace("(", "").replace(")", "").strip()
    ]
    # Handle edge cases in parsing
    if not quadrants:
      # Try alternate parsing
      parts = quadrants_str.replace("(", "").replace(")", "").split(",")
      quadrants = []
      for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
          quadrants.append(Point(int(parts[i]), int(parts[i + 1])))
    return cls(
      quadrants=quadrants,
      status=StepStatus(data.get("status", "pending")),
    )


# =============================================================================
# Database Operations
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


# =============================================================================
# Edge Detection
# =============================================================================


def find_generation_edge(bounds: StripBounds, generated: set[Point]) -> Edge | None:
  """
  Find the generation edge of the strip.

  The generation edge is the edge of the strip rectangle where ALL exterior
  neighboring quadrants are generated.

  Returns the edge or None if no valid edge is found.
  """
  # Check each edge
  edges_to_check = []

  # For horizontal strips, check top and bottom
  # For vertical strips, check left and right
  if bounds.is_horizontal:
    edges_to_check = [Edge.TOP, Edge.BOTTOM]
  else:
    edges_to_check = [Edge.LEFT, Edge.RIGHT]

  for edge in edges_to_check:
    if is_edge_fully_generated(bounds, edge, generated):
      return edge

  # Also check the perpendicular edges
  other_edges = (
    [Edge.LEFT, Edge.RIGHT] if bounds.is_horizontal else [Edge.TOP, Edge.BOTTOM]
  )
  for edge in other_edges:
    if is_edge_fully_generated(bounds, edge, generated):
      return edge

  return None


def is_edge_fully_generated(
  bounds: StripBounds, edge: Edge, generated: set[Point]
) -> bool:
  """Check if all exterior neighbors along an edge are generated."""
  exterior_neighbors = get_exterior_neighbors(bounds, edge)
  return all(p in generated for p in exterior_neighbors)


def get_exterior_neighbors(bounds: StripBounds, edge: Edge) -> list[Point]:
  """Get all exterior neighboring points along an edge."""
  neighbors = []

  if edge == Edge.TOP:
    y = bounds.top_left.y - 1
    for x in range(bounds.top_left.x, bounds.bottom_right.x + 1):
      neighbors.append(Point(x, y))
  elif edge == Edge.BOTTOM:
    y = bounds.bottom_right.y + 1
    for x in range(bounds.top_left.x, bounds.bottom_right.x + 1):
      neighbors.append(Point(x, y))
  elif edge == Edge.LEFT:
    x = bounds.top_left.x - 1
    for y in range(bounds.top_left.y, bounds.bottom_right.y + 1):
      neighbors.append(Point(x, y))
  elif edge == Edge.RIGHT:
    x = bounds.bottom_right.x + 1
    for y in range(bounds.top_left.y, bounds.bottom_right.y + 1):
      neighbors.append(Point(x, y))

  return neighbors


# =============================================================================
# Strip Generation Planning
# =============================================================================


def create_strip_plan(
  bounds: StripBounds,
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """
  Create a generation plan for the strip.

  The algorithm varies based on the depth of the strip:
  - Depth 1: 2x1 quadrants with 1 gap, then fill gaps
  - Depth 2: Apply depth-1 algorithm twice
  - Depth 3: 2x2 quadrants with gaps, then bridges, then fill
  - Depth > 3: First 3 using depth-3, then continue with remaining

  Args:
    bounds: The strip bounds
    generation_edge: The edge where generated content exists
    generated: Set of already-generated quadrant positions (for seam avoidance)
  """
  depth = bounds.depth
  generated = generated or set()

  if depth == 1:
    return create_depth_1_plan(bounds, generation_edge, generated)
  elif depth == 2:
    return create_depth_2_plan(bounds, generation_edge, generated)
  else:
    return create_depth_3_plus_plan(bounds, generation_edge, generated)


def create_depth_1_plan(
  bounds: StripBounds,
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """
  Create a plan for a depth-1 strip.

  Algorithm:
  1. Generate 2x1 quadrants (2 wide in direction of progress) with 1 gap
  2. Fill in the single-quadrant gaps

  If there's a generated quadrant at the start of the strip (perpendicular to
  the generation edge), we offset the start by 1 to avoid creating a seam.
  """
  steps: list[GenerationStep] = []
  generated = generated or set()
  is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

  if is_horizontal:
    # Progress left to right
    y = bounds.top_left.y
    x_start = bounds.top_left.x
    x_end = bounds.bottom_right.x

    # Check if there's a generated quadrant to the left of the strip start
    # If so, we need to offset by 1 to avoid a seam
    left_neighbor = Point(x_start - 1, y)
    if left_neighbor in generated:
      # Start with a single quadrant to create a gap, then continue with 2x1 pattern
      x_start_2x1 = x_start + 1
    else:
      x_start_2x1 = x_start

    # Track which positions are covered
    covered: set[int] = set()

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      covered.add(x)
      covered.add(x + 1)
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles and any remaining)
    for x in range(x_start, x_end + 1):
      if x not in covered:
        steps.append(GenerationStep([Point(x, y)]))

  else:
    # Progress top to bottom
    x = bounds.top_left.x
    y_start = bounds.top_left.y
    y_end = bounds.bottom_right.y

    # Check if there's a generated quadrant above the strip start
    # If so, we need to offset by 1 to avoid a seam
    top_neighbor = Point(x, y_start - 1)
    if top_neighbor in generated:
      y_start_2x1 = y_start + 1
    else:
      y_start_2x1 = y_start

    # Track which positions are covered
    covered: set[int] = set()

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      covered.add(y)
      covered.add(y + 1)
      y += 3

    # Phase 2: Fill single-quadrant gaps
    for y in range(y_start, y_end + 1):
      if y not in covered:
        steps.append(GenerationStep([Point(x, y)]))

  return steps


def create_depth_2_plan(
  bounds: StripBounds,
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """
  Create a plan for a depth-2 strip.

  Apply the depth-1 algorithm twice, once for each row/column.
  Start with the row/column closest to the generation edge.
  """
  steps: list[GenerationStep] = []
  generated = generated or set()
  is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

  if is_horizontal:
    # Two rows, progress left to right
    # Start with row closest to generation edge
    if generation_edge == Edge.BOTTOM:
      rows = [bounds.bottom_right.y, bounds.top_left.y]
    else:
      rows = [bounds.top_left.y, bounds.bottom_right.y]

    for y in rows:
      row_bounds = StripBounds(
        Point(bounds.top_left.x, y),
        Point(bounds.bottom_right.x, y),
      )
      row_steps = create_depth_1_plan(row_bounds, generation_edge, generated)
      steps.extend(row_steps)
  else:
    # Two columns, progress top to bottom
    # Start with column closest to generation edge
    if generation_edge == Edge.RIGHT:
      cols = [bounds.bottom_right.x, bounds.top_left.x]
    else:
      cols = [bounds.top_left.x, bounds.bottom_right.x]

    for x in cols:
      col_bounds = StripBounds(
        Point(x, bounds.top_left.y),
        Point(x, bounds.bottom_right.y),
      )
      col_steps = create_depth_1_plan(col_bounds, generation_edge, generated)
      steps.extend(col_steps)

  return steps


def create_depth_3_plus_plan(
  bounds: StripBounds,
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """
  Create a plan for a depth-3+ strip.

  For depth 3:
  1. Generate 2x2 quadrants away from edge with gaps
  2. Generate 1x2 bridges between 2x2 quadrants
  3. Generate 2x1 bridges back to edge
  4. Fill remaining gaps

  For depth > 3:
  - First 3 rows/cols using depth-3 formula
  - Continue with remaining using appropriate formula
  """
  steps: list[GenerationStep] = []
  generated = generated or set()
  is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

  if is_horizontal:
    steps = _create_horizontal_depth_3_plus_plan(bounds, generation_edge, generated)
  else:
    steps = _create_vertical_depth_3_plus_plan(bounds, generation_edge, generated)

  return steps


def _create_horizontal_depth_3_plus_plan(
  bounds: StripBounds,
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """Create depth-3+ plan for horizontal strip."""
  steps: list[GenerationStep] = []
  generated = generated or set()
  depth = bounds.depth

  # Determine y positions based on generation edge
  if generation_edge == Edge.BOTTOM:
    # Generated region is below, so we work from bottom-up
    # y positions from closest to farthest from edge
    y_positions = list(range(bounds.bottom_right.y, bounds.top_left.y - 1, -1))
  else:
    # Generated region is above, so we work from top-down
    y_positions = list(range(bounds.top_left.y, bounds.bottom_right.y + 1))

  x_start = bounds.top_left.x
  x_end = bounds.bottom_right.x

  # Process in chunks of 3 rows
  row_offset = 0
  while row_offset < depth:
    remaining_depth = depth - row_offset

    if remaining_depth >= 3:
      # Process 3 rows using the 3-deep formula
      y_rows = y_positions[row_offset : row_offset + 3]
      chunk_steps = _generate_3_row_chunk_horizontal(
        x_start, x_end, y_rows, generation_edge, generated
      )
      steps.extend(chunk_steps)
      row_offset += 3
    elif remaining_depth == 2:
      # Process 2 rows using depth-2 formula
      y_rows = y_positions[row_offset : row_offset + 2]
      chunk_bounds = StripBounds(
        Point(x_start, min(y_rows)),
        Point(x_end, max(y_rows)),
      )
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge, generated)
      steps.extend(chunk_steps)
      row_offset += 2
    else:
      # Process 1 row using depth-1 formula
      y = y_positions[row_offset]
      chunk_bounds = StripBounds(
        Point(x_start, y),
        Point(x_end, y),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge, generated)
      steps.extend(chunk_steps)
      row_offset += 1

  return steps


def _generate_3_row_chunk_horizontal(
  x_start: int,
  x_end: int,
  y_rows: list[int],
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """
  Generate steps for a 3-row horizontal chunk.

  y_rows should be ordered from closest to farthest from the generation edge.
  The 2x2 tiles go in the two rows FARTHEST from the generation edge,
  leaving the CLOSEST row as a gap/bridge row.

  IMPORTANT: 2x2 tiles must be 1 quadrant away from ALL edges with generated
  neighbors (left, right, and the generation edge). 2x1/1x2 tiles can only
  touch ONE previously generated edge.
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows is ordered: [closest to edge, middle, farthest from edge]
  # y_rows[0] = closest to generation edge (this becomes the bridge row)
  # y_rows[1] = middle
  # y_rows[2] = farthest from generation edge
  # 2x2 tiles use the two FARTHEST rows (y_rows[1] and y_rows[2])

  y_close = y_rows[0]  # Closest to generation edge - bridge row
  y_far_1 = y_rows[1]  # Middle - part of 2x2
  y_far_2 = y_rows[2]  # Farthest - part of 2x2

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check for generated neighbors on left and right edges
  # 2x2 tiles must be 1 quadrant away from ALL edges with neighbors
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  has_right_neighbor = any(Point(x_end + 1, y) in generated for y in y_rows)

  # Determine the valid range for 2x2 tiles (must have 1-quadrant gap from edges)
  x_2x2_start = x_start + 1 if has_left_neighbor else x_start
  x_2x2_end = x_end - 1 if has_right_neighbor else x_end

  # Track covered x positions for each row type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  # 2x2 tiles must stay within [x_2x2_start, x_2x2_end] to avoid touching edges
  x = x_2x2_start
  while x + 1 <= x_2x2_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x + 1, y_2x2_top),
          Point(x, y_2x2_bottom),
          Point(x + 1, y_2x2_bottom),
        ]
      )
    )
    covered_2x2.add(x)
    covered_2x2.add(x + 1)
    x += 3

  # Step 2: Generate 1x2 vertical bridges for gaps in 2x2 rows
  # These CAN touch one edge (left or right) but not both
  for x in range(x_start, x_end + 1):
    if x not in covered_2x2:
      steps.append(
        GenerationStep(
          [
            Point(x, y_2x2_top),
            Point(x, y_2x2_bottom),
          ]
        )
      )
      covered_2x2.add(x)

  # Step 3: Generate 2x1 horizontal bridges in the close row
  # These CAN touch one edge (the generation edge) since close row is adjacent to it
  # But must respect left/right edge gaps
  x = x_2x2_start
  while x + 1 <= x_2x2_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    covered_close.add(x)
    covered_close.add(x + 1)
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  for x in range(x_start, x_end + 1):
    if x not in covered_close:
      steps.append(GenerationStep([Point(x, y_close)]))

  return steps


def _create_vertical_depth_3_plus_plan(
  bounds: StripBounds,
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """Create depth-3+ plan for vertical strip."""
  steps: list[GenerationStep] = []
  generated = generated or set()
  depth = bounds.depth

  # Determine x positions based on generation edge
  if generation_edge == Edge.RIGHT:
    # Generated region is to the right, work from right-to-left
    x_positions = list(range(bounds.bottom_right.x, bounds.top_left.x - 1, -1))
  else:
    # Generated region is to the left, work from left-to-right
    x_positions = list(range(bounds.top_left.x, bounds.bottom_right.x + 1))

  y_start = bounds.top_left.y
  y_end = bounds.bottom_right.y

  # Process in chunks of 3 columns
  col_offset = 0
  while col_offset < depth:
    remaining_depth = depth - col_offset

    if remaining_depth >= 3:
      # Process 3 columns using the 3-deep formula
      x_cols = x_positions[col_offset : col_offset + 3]
      chunk_steps = _generate_3_col_chunk_vertical(
        y_start, y_end, x_cols, generation_edge, generated
      )
      steps.extend(chunk_steps)
      col_offset += 3
    elif remaining_depth == 2:
      # Process 2 columns using depth-2 formula
      x_cols = x_positions[col_offset : col_offset + 2]
      chunk_bounds = StripBounds(
        Point(min(x_cols), y_start),
        Point(max(x_cols), y_end),
      )
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge, generated)
      steps.extend(chunk_steps)
      col_offset += 2
    else:
      # Process 1 column using depth-1 formula
      x = x_positions[col_offset]
      chunk_bounds = StripBounds(
        Point(x, y_start),
        Point(x, y_end),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge, generated)
      steps.extend(chunk_steps)
      col_offset += 1

  return steps


def _generate_3_col_chunk_vertical(
  y_start: int,
  y_end: int,
  x_cols: list[int],
  generation_edge: Edge,
  generated: set[Point] | None = None,
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  The 2x2 tiles go in the two columns FARTHEST from the generation edge,
  leaving the CLOSEST column as a gap/bridge column.

  IMPORTANT: 2x2 tiles must be 1 quadrant away from ALL edges with generated
  neighbors (top, bottom, and the generation edge). 2x1/1x2 tiles can only
  touch ONE previously generated edge.
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # x_cols is ordered: [closest to edge, middle, farthest from edge]
  # x_cols[0] = closest to generation edge (this becomes the bridge column)
  # x_cols[1] = middle
  # x_cols[2] = farthest from generation edge
  # 2x2 tiles use the two FARTHEST columns (x_cols[1] and x_cols[2])

  x_close = x_cols[0]  # Closest to generation edge - bridge column
  x_far_1 = x_cols[1]  # Middle - part of 2x2
  x_far_2 = x_cols[2]  # Farthest - part of 2x2

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Check for generated neighbors on top and bottom edges
  # 2x2 tiles must be 1 quadrant away from ALL edges with neighbors
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  has_bottom_neighbor = any(Point(x, y_end + 1) in generated for x in x_cols)

  # Determine the valid range for 2x2 tiles (must have 1-quadrant gap from edges)
  y_2x2_start = y_start + 1 if has_top_neighbor else y_start
  y_2x2_end = y_end - 1 if has_bottom_neighbor else y_end

  # Track covered y positions for each column type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # 2x2 tiles must stay within [y_2x2_start, y_2x2_end] to avoid touching edges
  y = y_2x2_start
  while y + 1 <= y_2x2_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
          Point(x_2x2_left, y + 1),
          Point(x_2x2_right, y + 1),
        ]
      )
    )
    covered_2x2.add(y)
    covered_2x2.add(y + 1)
    y += 3

  # Step 2: Generate 2x1 horizontal bridges for gaps in 2x2 columns
  # These CAN touch one edge (top or bottom) but not both
  for y in range(y_start, y_end + 1):
    if y not in covered_2x2:
      steps.append(
        GenerationStep(
          [
            Point(x_2x2_left, y),
            Point(x_2x2_right, y),
          ]
        )
      )
      covered_2x2.add(y)

  # Step 3: Generate 1x2 vertical bridges in the close column
  # These CAN touch one edge (the generation edge) since close col is adjacent to it
  # But must respect top/bottom edge gaps
  y = y_2x2_start
  while y + 1 <= y_2x2_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    covered_close.add(y)
    covered_close.add(y + 1)
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  for y in range(y_start, y_end + 1):
    if y not in covered_close:
      steps.append(GenerationStep([Point(x_close, y)]))

  return steps


# =============================================================================
# Plan File Operations
# =============================================================================


def save_strip_plan(
  plan: list[GenerationStep], generation_dir: Path, tl: Point, br: Point
) -> Path:
  """Save the strip plan to a JSON file."""
  filename = f"generate_strip_{tl.x}_{tl.y}_{br.x}_{br.y}.json"
  path = generation_dir / filename

  plan_data = [step.to_dict() for step in plan]

  with open(path, "w") as f:
    json.dump(plan_data, f, indent=2)

  return path


def load_strip_plan(path: Path) -> list[GenerationStep]:
  """Load a strip plan from a JSON file."""
  with open(path) as f:
    data = json.load(f)
  return [GenerationStep.from_dict(item) for item in data]


# =============================================================================
# Main
# =============================================================================


def parse_coordinate(s: str) -> Point:
  """Parse a coordinate string like '(x,y)' or 'x,y' into a Point."""
  return Point.from_string(s)


def create_strip_plan_from_args(
  generation_dir: Path, tl: Point, br: Point
) -> tuple[list[GenerationStep], Edge]:
  """
  Create a strip generation plan.

  Returns the plan and the generation edge.
  Raises ValueError if no valid generation edge is found.
  """
  # Load database
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  conn = sqlite3.connect(db_path)
  try:
    generated = load_generated_quadrants(conn)
  finally:
    conn.close()

  bounds = StripBounds(tl, br)

  # Find generation edge
  edge = find_generation_edge(bounds, generated)
  if edge is None:
    raise ValueError(
      "No valid generation edge found. "
      "At least one edge of the strip must have all exterior neighbors generated."
    )

  # Create plan (pass generated set for seam avoidance)
  plan = create_strip_plan(bounds, edge, generated)

  return plan, edge


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create a strip generation plan.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--tl",
    type=str,
    required=True,
    help="Top-left corner of the strip (x,y). For negative coords use --tl='-1,-2'",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y). For negative coords use --br='-1,-2'",
  )

  args = parser.parse_args()

  try:
    tl = parse_coordinate(args.tl)
    br = parse_coordinate(args.br)
  except ValueError as e:
    print(f"❌ Error parsing coordinates: {e}")
    return 1

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    print("❌ Error: top-left must be above and to the left of bottom-right")
    return 1

  generation_dir = args.generation_dir.resolve()
  if not generation_dir.exists():
    print(f"❌ Error: Generation directory not found: {generation_dir}")
    return 1

  bounds = StripBounds(tl, br)
  print(f"📏 Strip bounds: {tl} to {br}")
  print(f"   Size: {bounds.width} x {bounds.height} (depth={bounds.depth})")

  try:
    plan, edge = create_strip_plan_from_args(generation_dir, tl, br)
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Error: {e}")
    return 1

  print(f"🧭 Generation edge: {edge.value}")
  print(f"📋 Generated {len(plan)} steps")

  # Save plan
  plan_path = save_strip_plan(plan, generation_dir, tl, br)
  print(f"💾 Saved plan to {plan_path}")

  # Print summary
  total_quadrants = sum(len(step.quadrants) for step in plan)
  by_size: dict[int, int] = {}
  for step in plan:
    size = len(step.quadrants)
    by_size[size] = by_size.get(size, 0) + 1

  print("\n📊 Summary:")
  print(f"   Total quadrants: {total_quadrants}")
  print("   Steps by tile size:")
  for size in sorted(by_size.keys(), reverse=True):
    label = {4: "2x2 tiles", 2: "2-quadrant tiles", 1: "single quadrants"}
    print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

  return 0


if __name__ == "__main__":
  exit(main())
