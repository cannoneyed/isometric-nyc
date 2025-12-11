# Generate specific plan document
_Exported on 12/11/2025 at 09:40:45 EST from Cursor (2.1.39)_

---

**User**

Generate the plan in @tasks/019_strip_plan.md 

---

**Cursor**

I'll implement the strip plan script as described in the task. Let me first explore the existing codebase to understand the structure and patterns used.

```plaintext
[{
  "quadrants": "(x,y),(x,y)",
  "status": "pending"|"done"|"error"
}, ...]
```

Now let me look at the test structure in the codebase:

Now I have a good understanding of the codebase. Let me create the `make_strip_plan.py` script and a test suite.

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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
            Point.from_string(q) for q in quadrants_str.split("),(")
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


def find_generation_edge(
    bounds: StripBounds, generated: set[Point]
) -> Edge | None:
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
    other_edges = [Edge.LEFT, Edge.RIGHT] if bounds.is_horizontal else [Edge.TOP, Edge.BOTTOM]
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
    bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
    """
    Create a generation plan for the strip.

    The algorithm varies based on the depth of the strip:
    - Depth 1: 2x1 quadrants with 1 gap, then fill gaps
    - Depth 2: Apply depth-1 algorithm twice
    - Depth 3: 2x2 quadrants with gaps, then bridges, then fill
    - Depth > 3: First 3 using depth-3, then continue with remaining
    """
    depth = bounds.depth

    if depth == 1:
        return create_depth_1_plan(bounds, generation_edge)
    elif depth == 2:
        return create_depth_2_plan(bounds, generation_edge)
    else:
        return create_depth_3_plus_plan(bounds, generation_edge)


def create_depth_1_plan(
    bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
    """
    Create a plan for a depth-1 strip.

    Algorithm:
    1. Generate 2x1 quadrants (2 wide in direction of progress) with 1 gap
    2. Fill in the single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

    if is_horizontal:
        # Progress left to right
        y = bounds.top_left.y
        x_start = bounds.top_left.x
        x_end = bounds.bottom_right.x

        # Phase 1: Generate 2x1 quadrants with 1 gap
        # Pattern: SS.SS.SS... (S=selected, .=gap)
        x = x_start
        while x + 1 <= x_end:
            steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
            x += 3  # Move by 3 (2 selected + 1 gap)

        # Phase 2: Fill single-quadrant gaps
        x = x_start + 2  # First gap position
        while x <= x_end:
            steps.append(GenerationStep([Point(x, y)]))
            x += 3
    else:
        # Progress top to bottom
        x = bounds.top_left.x
        y_start = bounds.top_left.y
        y_end = bounds.bottom_right.y

        # Phase 1: Generate 2x1 quadrants with 1 gap
        y = y_start
        while y + 1 <= y_end:
            steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
            y += 3

        # Phase 2: Fill single-quadrant gaps
        y = y_start + 2
        while y <= y_end:
            steps.append(GenerationStep([Point(x, y)]))
            y += 3

    return steps


def create_depth_2_plan(
    bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
    """
    Create a plan for a depth-2 strip.

    Apply the depth-1 algorithm twice, once for each row/column.
    Start with the row/column closest to the generation edge.
    """
    steps: list[GenerationStep] = []
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
            row_steps = create_depth_1_plan(row_bounds, generation_edge)
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
            col_steps = create_depth_1_plan(col_bounds, generation_edge)
            steps.extend(col_steps)

    return steps


def create_depth_3_plus_plan(
    bounds: StripBounds, generation_edge: Edge
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
    is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

    if is_horizontal:
        steps = _create_horizontal_depth_3_plus_plan(bounds, generation_edge)
    else:
        steps = _create_vertical_depth_3_plus_plan(bounds, generation_edge)

    return steps


def _create_horizontal_depth_3_plus_plan(
    bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
    """Create depth-3+ plan for horizontal strip."""
    steps: list[GenerationStep] = []
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
                x_start, x_end, y_rows, generation_edge
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
            chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
            steps.extend(chunk_steps)
            row_offset += 2
        else:
            # Process 1 row using depth-1 formula
            y = y_positions[row_offset]
            chunk_bounds = StripBounds(
                Point(x_start, y),
                Point(x_end, y),
            )
            chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
            steps.extend(chunk_steps)
            row_offset += 1

    return steps


def _generate_3_row_chunk_horizontal(
    x_start: int, x_end: int, y_rows: list[int], generation_edge: Edge
) -> list[GenerationStep]:
    """
    Generate steps for a 3-row horizontal chunk.

    y_rows should be ordered from closest to farthest from the generation edge.
    """
    steps: list[GenerationStep] = []

    # y_rows[0] = closest to edge (will be done last)
    # y_rows[1] = middle row
    # y_rows[2] = farthest from edge (will be done first with 2x2)

    # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
    # and y3 for the bridge row (closest to edge)
    y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
    y_far_2 = y_rows[1]
    y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

    # The 2x2 tiles should use the two rows farthest from edge
    # Ensure y_2x2_top < y_2x2_bottom
    y_2x2_top = min(y_far_1, y_far_2)
    y_2x2_bottom = max(y_far_1, y_far_2)

    # Step 1: Generate 2x2 quadrants with gaps
    # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
    x = x_start
    while x + 1 <= x_end:
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
        x += 3

    # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
    x = x_start + 2
    while x <= x_end:
        steps.append(
            GenerationStep(
                [
                    Point(x, y_2x2_top),
                    Point(x, y_2x2_bottom),
                ]
            )
        )
        x += 3

    # Step 3: Generate 2x1 horizontal bridges in the close row
    x = x_start
    while x + 1 <= x_end:
        steps.append(
            GenerationStep(
                [
                    Point(x, y_close),
                    Point(x + 1, y_close),
                ]
            )
        )
        x += 3

    # Step 4: Fill remaining single-quadrant gaps in the close row
    x = x_start + 2
    while x <= x_end:
        steps.append(GenerationStep([Point(x, y_close)]))
        x += 3

    return steps


def _create_vertical_depth_3_plus_plan(
    bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
    """Create depth-3+ plan for vertical strip."""
    steps: list[GenerationStep] = []
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
                y_start, y_end, x_cols, generation_edge
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
            chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
            steps.extend(chunk_steps)
            col_offset += 2
        else:
            # Process 1 column using depth-1 formula
            x = x_positions[col_offset]
            chunk_bounds = StripBounds(
                Point(x, y_start),
                Point(x, y_end),
            )
            chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
            steps.extend(chunk_steps)
            col_offset += 1

    return steps


def _generate_3_col_chunk_vertical(
    y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
    """
    Generate steps for a 3-column vertical chunk.

    x_cols should be ordered from closest to farthest from the generation edge.
    """
    steps: list[GenerationStep] = []

    # x_cols[0] = closest to edge
    # x_cols[2] = farthest from edge

    x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
    x_far_2 = x_cols[1]
    x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

    x_2x2_left = min(x_far_1, x_far_2)
    x_2x2_right = max(x_far_1, x_far_2)

    # Step 1: Generate 2x2 quadrants with gaps
    y = y_start
    while y + 1 <= y_end:
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
        y += 3

    # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
    y = y_start + 2
    while y <= y_end:
        steps.append(
            GenerationStep(
                [
                    Point(x_2x2_left, y),
                    Point(x_2x2_right, y),
                ]
            )
        )
        y += 3

    # Step 3: Generate 1x2 vertical bridges in the close column
    y = y_start
    while y + 1 <= y_end:
        steps.append(
            GenerationStep(
                [
                    Point(x_close, y),
                    Point(x_close, y + 1),
                ]
            )
        )
        y += 3

    # Step 4: Fill remaining single-quadrant gaps in the close column
    y = y_start + 2
    while y <= y_end:
        steps.append(GenerationStep([Point(x_close, y)]))
        y += 3

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

    # Create plan
    plan = create_strip_plan(bounds, edge)

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
        help="Top-left corner of the strip (x,y)",
    )
    parser.add_argument(
        "--br",
        type=str,
        required=True,
        help="Bottom-right corner of the strip (x,y)",
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

    print(f"\n📊 Summary:")
    print(f"   Total quadrants: {total_quadrants}")
    print(f"   Steps by tile size:")
    for size in sorted(by_size.keys(), reverse=True):
        label = {4: "2x2 tiles", 2: "2-quadrant tiles", 1: "single quadrants"}
        print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

    return 0


if __name__ == "__main__":
    exit(main())

```

Now let me create the test suite:

```python
"""
Tests for make_strip_plan.py

These tests verify the strip generation planning algorithms without
needing a database connection.
"""

import pytest

from isometric_nyc.e2e_generation.make_strip_plan import (
    Edge,
    GenerationStep,
    Point,
    StepStatus,
    StripBounds,
    create_depth_1_plan,
    create_depth_2_plan,
    create_depth_3_plus_plan,
    create_strip_plan,
    find_generation_edge,
    get_exterior_neighbors,
    is_edge_fully_generated,
)


# =============================================================================
# Point Tests
# =============================================================================


class TestPoint:
    def test_str(self) -> None:
        p = Point(3, 5)
        assert str(p) == "(3,5)"

    def test_add(self) -> None:
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        result = p1 + p2
        assert result == Point(4, 6)

    def test_from_string_simple(self) -> None:
        p = Point.from_string("3,5")
        assert p == Point(3, 5)

    def test_from_string_with_parens(self) -> None:
        p = Point.from_string("(3,5)")
        assert p == Point(3, 5)

    def test_from_string_with_spaces(self) -> None:
        p = Point.from_string("( 3 , 5 )")
        assert p == Point(3, 5)

    def test_from_string_negative(self) -> None:
        p = Point.from_string("-3,-5")
        assert p == Point(-3, -5)

    def test_from_string_invalid(self) -> None:
        with pytest.raises(ValueError):
            Point.from_string("invalid")


# =============================================================================
# StripBounds Tests
# =============================================================================


class TestStripBounds:
    def test_width_height(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(10, 2))
        assert bounds.width == 11
        assert bounds.height == 3

    def test_is_horizontal(self) -> None:
        # Horizontal strip (wider than tall)
        h_bounds = StripBounds(Point(0, 0), Point(10, 2))
        assert h_bounds.is_horizontal is True

        # Vertical strip (taller than wide)
        v_bounds = StripBounds(Point(0, 0), Point(2, 10))
        assert v_bounds.is_horizontal is False

        # Square (treated as horizontal)
        sq_bounds = StripBounds(Point(0, 0), Point(5, 5))
        assert sq_bounds.is_horizontal is True

    def test_depth(self) -> None:
        # Horizontal strip: depth = height
        h_bounds = StripBounds(Point(0, 0), Point(10, 2))
        assert h_bounds.depth == 3

        # Vertical strip: depth = width
        v_bounds = StripBounds(Point(0, 0), Point(2, 10))
        assert v_bounds.depth == 3

    def test_length(self) -> None:
        # Horizontal strip: length = width
        h_bounds = StripBounds(Point(0, 0), Point(10, 2))
        assert h_bounds.length == 11

        # Vertical strip: length = height
        v_bounds = StripBounds(Point(0, 0), Point(2, 10))
        assert v_bounds.length == 11

    def test_all_points(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(2, 1))
        points = bounds.all_points()
        expected = [
            Point(0, 0),
            Point(1, 0),
            Point(2, 0),
            Point(0, 1),
            Point(1, 1),
            Point(2, 1),
        ]
        assert points == expected


# =============================================================================
# Edge Detection Tests
# =============================================================================


class TestGetExteriorNeighbors:
    def test_top_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(2, 0))
        neighbors = get_exterior_neighbors(bounds, Edge.TOP)
        assert neighbors == [Point(0, -1), Point(1, -1), Point(2, -1)]

    def test_bottom_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(2, 0))
        neighbors = get_exterior_neighbors(bounds, Edge.BOTTOM)
        assert neighbors == [Point(0, 1), Point(1, 1), Point(2, 1)]

    def test_left_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(0, 2))
        neighbors = get_exterior_neighbors(bounds, Edge.LEFT)
        assert neighbors == [Point(-1, 0), Point(-1, 1), Point(-1, 2)]

    def test_right_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(0, 2))
        neighbors = get_exterior_neighbors(bounds, Edge.RIGHT)
        assert neighbors == [Point(1, 0), Point(1, 1), Point(1, 2)]


class TestIsEdgeFullyGenerated:
    def test_fully_generated_top(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(2, 0))
        generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
        assert is_edge_fully_generated(bounds, Edge.TOP, generated) is True

    def test_partially_generated_top(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(2, 0))
        generated = {Point(0, -1), Point(2, -1)}  # Missing Point(1, -1)
        assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False

    def test_not_generated_top(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(2, 0))
        generated: set[Point] = set()
        assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False


class TestFindGenerationEdge:
    def test_horizontal_strip_bottom_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(10, 0))
        # Generate all quadrants below the strip
        generated = {Point(x, 1) for x in range(11)}
        edge = find_generation_edge(bounds, generated)
        assert edge == Edge.BOTTOM

    def test_horizontal_strip_top_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(10, 0))
        # Generate all quadrants above the strip
        generated = {Point(x, -1) for x in range(11)}
        edge = find_generation_edge(bounds, generated)
        assert edge == Edge.TOP

    def test_vertical_strip_left_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(0, 10))
        # Generate all quadrants to the left
        generated = {Point(-1, y) for y in range(11)}
        edge = find_generation_edge(bounds, generated)
        assert edge == Edge.LEFT

    def test_vertical_strip_right_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(0, 10))
        # Generate all quadrants to the right
        generated = {Point(1, y) for y in range(11)}
        edge = find_generation_edge(bounds, generated)
        assert edge == Edge.RIGHT

    def test_no_valid_edge(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(10, 0))
        generated: set[Point] = set()  # Nothing generated
        edge = find_generation_edge(bounds, generated)
        assert edge is None


# =============================================================================
# Depth 1 Plan Tests
# =============================================================================


class TestDepth1Plan:
    def test_horizontal_depth_1_simple(self) -> None:
        """Test a simple 11-wide depth-1 horizontal strip."""
        bounds = StripBounds(Point(0, 0), Point(10, 0))
        plan = create_depth_1_plan(bounds, Edge.BOTTOM)

        # Should have 2x1 tiles first, then single fills
        # Pattern: SS.SS.SS.SS (positions 0-1, 3-4, 6-7, 9-10)
        # Then fills: positions 2, 5, 8

        # Extract all quadrants
        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        # Should cover all 11 positions
        expected = {Point(x, 0) for x in range(11)}
        assert set(all_quadrants) == expected

    def test_horizontal_depth_1_pattern(self) -> None:
        """Verify the 2x1 + gap + fill pattern."""
        bounds = StripBounds(Point(0, 0), Point(7, 0))
        plan = create_depth_1_plan(bounds, Edge.BOTTOM)

        # Phase 1: 2x1 tiles at positions 0-1, 3-4, 6-7
        # Phase 2: Single tiles at positions 2, 5

        # First step should be 2x1 at (0,0)
        assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

        # Second step should be 2x1 at (3,0)
        assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]

        # Third step should be 2x1 at (6,0)
        assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]

        # Fourth step should be single at (2,0)
        assert plan[3].quadrants == [Point(2, 0)]

        # Fifth step should be single at (5,0)
        assert plan[4].quadrants == [Point(5, 0)]

    def test_vertical_depth_1(self) -> None:
        """Test a vertical depth-1 strip."""
        bounds = StripBounds(Point(0, 0), Point(0, 7))
        plan = create_depth_1_plan(bounds, Edge.LEFT)

        # Should have 2-quadrant tiles first, then singles
        # Pattern: SS.SS.SS.SS (y positions 0-1, 3-4, 6-7)
        # Then fills: positions 2, 5

        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        expected = {Point(0, y) for y in range(8)}
        assert set(all_quadrants) == expected


# =============================================================================
# Depth 2 Plan Tests
# =============================================================================


class TestDepth2Plan:
    def test_horizontal_depth_2_coverage(self) -> None:
        """Test that depth-2 plan covers all quadrants."""
        bounds = StripBounds(Point(0, 0), Point(7, 1))
        plan = create_depth_2_plan(bounds, Edge.BOTTOM)

        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        expected = {Point(x, y) for x in range(8) for y in range(2)}
        assert set(all_quadrants) == expected

    def test_horizontal_depth_2_order(self) -> None:
        """Test that depth-2 processes row closest to edge first."""
        bounds = StripBounds(Point(0, 0), Point(4, 1))
        plan = create_depth_2_plan(bounds, Edge.BOTTOM)

        # With BOTTOM edge, should process y=1 first, then y=0
        # First steps should all be at y=1
        first_row_steps = []
        for step in plan:
            if all(q.y == 1 for q in step.quadrants):
                first_row_steps.append(step)
            elif len(first_row_steps) > 0:
                break  # Found a different row, stop

        assert len(first_row_steps) > 0
        assert all(q.y == 1 for step in first_row_steps for q in step.quadrants)


# =============================================================================
# Depth 3+ Plan Tests
# =============================================================================


class TestDepth3PlusPlan:
    def test_horizontal_depth_3_coverage(self) -> None:
        """Test that depth-3 plan covers all quadrants."""
        bounds = StripBounds(Point(0, 0), Point(7, 2))
        plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        expected = {Point(x, y) for x in range(8) for y in range(3)}
        assert set(all_quadrants) == expected

    def test_depth_3_has_2x2_tiles(self) -> None:
        """Test that depth-3 plan uses 2x2 tiles."""
        bounds = StripBounds(Point(0, 0), Point(7, 2))
        plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

        # Should have at least one 2x2 tile
        has_2x2 = any(len(step.quadrants) == 4 for step in plan)
        assert has_2x2, "Depth-3 plan should contain 2x2 tiles"

    def test_depth_4_coverage(self) -> None:
        """Test that depth-4 (3+1) plan covers all quadrants."""
        bounds = StripBounds(Point(0, 0), Point(7, 3))
        plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        expected = {Point(x, y) for x in range(8) for y in range(4)}
        assert set(all_quadrants) == expected

    def test_depth_5_coverage(self) -> None:
        """Test that depth-5 (3+2) plan covers all quadrants."""
        bounds = StripBounds(Point(0, 0), Point(7, 4))
        plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        expected = {Point(x, y) for x in range(8) for y in range(5)}
        assert set(all_quadrants) == expected

    def test_depth_6_coverage(self) -> None:
        """Test that depth-6 (3+3) plan covers all quadrants."""
        bounds = StripBounds(Point(0, 0), Point(7, 5))
        plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        expected = {Point(x, y) for x in range(8) for y in range(6)}
        assert set(all_quadrants) == expected


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateStripPlan:
    def test_depth_1_uses_depth_1_plan(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(10, 0))
        plan = create_strip_plan(bounds, Edge.BOTTOM)

        # Depth 1 should not have any 4-quadrant tiles
        assert all(len(step.quadrants) <= 2 for step in plan)

    def test_depth_2_uses_depth_2_plan(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(10, 1))
        plan = create_strip_plan(bounds, Edge.BOTTOM)

        # Depth 2 should not have any 4-quadrant tiles
        assert all(len(step.quadrants) <= 2 for step in plan)

    def test_depth_3_uses_depth_3_plan(self) -> None:
        bounds = StripBounds(Point(0, 0), Point(10, 2))
        plan = create_strip_plan(bounds, Edge.BOTTOM)

        # Depth 3 should have some 4-quadrant tiles
        has_4_quad = any(len(step.quadrants) == 4 for step in plan)
        assert has_4_quad


# =============================================================================
# GenerationStep Serialization Tests
# =============================================================================


class TestGenerationStepSerialization:
    def test_to_dict(self) -> None:
        step = GenerationStep(
            quadrants=[Point(0, 0), Point(1, 0)],
            status=StepStatus.PENDING,
        )
        d = step.to_dict()
        assert d["quadrants"] == "(0,0),(1,0)"
        assert d["status"] == "pending"

    def test_from_dict(self) -> None:
        d = {
            "quadrants": "(0,0),(1,0)",
            "status": "done",
        }
        step = GenerationStep.from_dict(d)
        assert step.quadrants == [Point(0, 0), Point(1, 0)]
        assert step.status == StepStatus.DONE

    def test_roundtrip(self) -> None:
        original = GenerationStep(
            quadrants=[Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)],
            status=StepStatus.PENDING,
        )
        d = original.to_dict()
        restored = GenerationStep.from_dict(d)
        assert restored.quadrants == original.quadrants
        assert restored.status == original.status


# =============================================================================
# Task Example Tests (from 019_strip_plan.md)
# =============================================================================


class TestTaskExamples:
    def test_depth_1_example_from_task(self) -> None:
        """
        Test the example from the task file:
        tl=(0,0) br=(10,0), depth=1

        Expected pattern:
        Step 1: (0,0),(1,0)
        Step 2: (3,0),(4,0)
        Step 3: (6,0),(7,0)
        Step 4: (9,0),(10,0)
        Step 5: (2,0)
        Step 6: (5,0)
        Step 7: (8,0)
        """
        bounds = StripBounds(Point(0, 0), Point(10, 0))
        plan = create_depth_1_plan(bounds, Edge.BOTTOM)

        # Verify first 4 steps are 2x1 tiles
        assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]
        assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]
        assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]
        assert plan[3].quadrants == [Point(9, 0), Point(10, 0)]

        # Verify remaining steps are single quadrants
        assert plan[4].quadrants == [Point(2, 0)]
        assert plan[5].quadrants == [Point(5, 0)]
        assert plan[6].quadrants == [Point(8, 0)]

        # Total should be 7 steps
        assert len(plan) == 7

    def test_depth_3_example_from_task(self) -> None:
        """
        Test the example from the task file:
        tl=(0,0) br=(7,2), depth=3

        This tests the general pattern of:
        1. 2x2 tiles away from edge
        2. 1x2 bridges between 2x2s
        3. 2x1 bridges back to edge
        4. Single quadrant fills
        """
        bounds = StripBounds(Point(0, 0), Point(7, 2))
        plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

        # Collect all quadrants
        all_quadrants: list[Point] = []
        for step in plan:
            all_quadrants.extend(step.quadrants)

        # Should cover all 24 quadrants (8 x 3)
        expected = {Point(x, y) for x in range(8) for y in range(3)}
        assert set(all_quadrants) == expected

        # First steps should be 2x2 tiles
        four_quad_steps = [s for s in plan if len(s.quadrants) == 4]
        assert len(four_quad_steps) >= 2, "Should have at least 2 2x2 tiles"

```

Now let me run the tests to verify everything works:

Let me check if pytest is installed:

---

**User**

Keep going!

---

**Cursor**

All tests pass. Here's a summary of what was created:

## Summary

### Files Created

1. **`src/isometric_nyc/e2e_generation/make_strip_plan.py`** - The main script that creates strip generation plans
2. **`tests/test_make_strip_plan.py`** - 42 tests covering all the functionality

### Usage

```bash
uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \
  <generation_dir> \
  --tl <x>,<y> \
  --br <x>,<y>
```

### Algorithm Implemented

The script implements the strip generation algorithm from the task:

1. **Generation Edge Detection**: Finds the edge of the strip rectangle where all exterior neighboring quadrants are already generated

2. **Depth-based Planning**:
   - **Depth 1**: Generate 2×1 quadrants with 1-quadrant gaps, then fill single gaps
   - **Depth 2**: Apply depth-1 algorithm twice (once per row/column)
   - **Depth 3**: 
     - Generate 2×2 quadrants away from edge with gaps
     - Bridge with 1×2 vertical tiles
     - Bridge back with 2×1 horizontal tiles
     - Fill remaining single-quadrant gaps
   - **Depth > 3**: Process in chunks of 3, then 2, then 1

3. **Output**: Creates a JSON file `generate_strip_<tl_x>_<tl_y>_<br_x>_<br_y>.json` with the schema:
   ```json
   [{
     "quadrants": "(x,y),(x,y)",
     "status": "pending"|"done"|"error"
   }, ...]
   ```

### Test Coverage

The test suite covers:
- Point parsing and operations
- StripBounds calculations (width, height, depth, length)
- Edge detection (all 4 edges)
- Depth-1, 2, 3, 4, 5, 6 plan generation
- JSON serialization roundtrips
- The specific examples from the task document

```shellscript
uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \
  <generation_dir> \
  --tl <x>,<y> \
  --br <x>,<y>
```

```json
   [{
     "quadrants": "(x,y),(x,y)",
     "status": "pending"|"done"|"error"
   }, ...]
```

---

**User**

For the 1 deep or 2 deep strip, ensure that no seam would be created with the first 2x1 generation (i.e. if there's a generated quadrant to the chunk's left/right) - if there is, move it one quadrant over so there's always a 1 quadrant gap

---

**Cursor**

I understand - we need to ensure that the first 2×1 generation has a gap from any generated quadrants at the ends of the strip (perpendicular to the generation edge). Let me update the algorithm.

Now I need to update the `create_depth_1_plan` and related functions to check for generated quadrants at the ends of the strip and offset the starting position if needed to avoid seams.

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """
  Create a plan for a depth-1 strip.

  Algorithm:
  1. Generate 2x1 quadrants (2 wide in direction of progress) with 1 gap
  2. Fill in the single-quadrant gaps
  """
  steps: list[GenerationStep] = []
  is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

  if is_horizontal:
    # Progress left to right
    y = bounds.top_left.y
    x_start = bounds.top_left.x
    x_end = bounds.bottom_right.x

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps
    x = x_start + 2  # First gap position
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3
  else:
    # Progress top to bottom
    x = bounds.top_left.x
    y_start = bounds.top_left.y
    y_end = bounds.bottom_right.y

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

  return steps


def create_depth_2_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """
  Create a plan for a depth-2 strip.

  Apply the depth-1 algorithm twice, once for each row/column.
  Start with the row/column closest to the generation edge.
  """
  steps: list[GenerationStep] = []
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
      row_steps = create_depth_1_plan(row_bounds, generation_edge)
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
      col_steps = create_depth_1_plan(col_bounds, generation_edge)
      steps.extend(col_steps)

  return steps


def create_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
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
  is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

  if is_horizontal:
    steps = _create_horizontal_depth_3_plus_plan(bounds, generation_edge)
  else:
    steps = _create_vertical_depth_3_plus_plan(bounds, generation_edge)

  return steps


def _create_horizontal_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for horizontal strip."""
  steps: list[GenerationStep] = []
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
        x_start, x_end, y_rows, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 2
    else:
      # Process 1 row using depth-1 formula
      y = y_positions[row_offset]
      chunk_bounds = StripBounds(
        Point(x_start, y),
        Point(x_end, y),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 1

  return steps


def _generate_3_row_chunk_horizontal(
  x_start: int, x_end: int, y_rows: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-row horizontal chunk.

  y_rows should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  return steps


def _create_vertical_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for vertical strip."""
  steps: list[GenerationStep] = []
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
        y_start, y_end, x_cols, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 2
    else:
      # Process 1 column using depth-1 formula
      x = x_positions[col_offset]
      chunk_bounds = StripBounds(
        Point(x, y_start),
        Point(x, y_end),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 1

  return steps


def _generate_3_col_chunk_vertical(
  y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

  return steps


def create_depth_2_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """
  Create a plan for a depth-2 strip.

  Apply the depth-1 algorithm twice, once for each row/column.
  Start with the row/column closest to the generation edge.
  """
  steps: list[GenerationStep] = []
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
      row_steps = create_depth_1_plan(row_bounds, generation_edge)
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
      col_steps = create_depth_1_plan(col_bounds, generation_edge)
      steps.extend(col_steps)

  return steps


def create_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
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
  is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

  if is_horizontal:
    steps = _create_horizontal_depth_3_plus_plan(bounds, generation_edge)
  else:
    steps = _create_vertical_depth_3_plus_plan(bounds, generation_edge)

  return steps


def _create_horizontal_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for horizontal strip."""
  steps: list[GenerationStep] = []
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
        x_start, x_end, y_rows, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 2
    else:
      # Process 1 row using depth-1 formula
      y = y_positions[row_offset]
      chunk_bounds = StripBounds(
        Point(x_start, y),
        Point(x_end, y),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 1

  return steps


def _generate_3_row_chunk_horizontal(
  x_start: int, x_end: int, y_rows: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-row horizontal chunk.

  y_rows should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  return steps


def _create_vertical_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for vertical strip."""
  steps: list[GenerationStep] = []
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
        y_start, y_end, x_cols, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 2
    else:
      # Process 1 column using depth-1 formula
      x = x_positions[col_offset]
      chunk_bounds = StripBounds(
        Point(x, y_start),
        Point(x, y_end),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 1

  return steps


def _generate_3_col_chunk_vertical(
  y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

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
  bounds: StripBounds, generation_edge: Edge
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
  is_horizontal = generation_edge in [Edge.TOP, Edge.BOTTOM]

  if is_horizontal:
    steps = _create_horizontal_depth_3_plus_plan(bounds, generation_edge)
  else:
    steps = _create_vertical_depth_3_plus_plan(bounds, generation_edge)

  return steps


def _create_horizontal_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for horizontal strip."""
  steps: list[GenerationStep] = []
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
        x_start, x_end, y_rows, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 2
    else:
      # Process 1 row using depth-1 formula
      y = y_positions[row_offset]
      chunk_bounds = StripBounds(
        Point(x_start, y),
        Point(x_end, y),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 1

  return steps


def _generate_3_row_chunk_horizontal(
  x_start: int, x_end: int, y_rows: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-row horizontal chunk.

  y_rows should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  return steps


def _create_vertical_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for vertical strip."""
  steps: list[GenerationStep] = []
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
        y_start, y_end, x_cols, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 2
    else:
      # Process 1 column using depth-1 formula
      x = x_positions[col_offset]
      chunk_bounds = StripBounds(
        Point(x, y_start),
        Point(x, y_end),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 1

  return steps


def _generate_3_col_chunk_vertical(
  y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

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
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for horizontal strip."""
  steps: list[GenerationStep] = []
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
        x_start, x_end, y_rows, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 2
    else:
      # Process 1 row using depth-1 formula
      y = y_positions[row_offset]
      chunk_bounds = StripBounds(
        Point(x_start, y),
        Point(x_end, y),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      row_offset += 1

  return steps


def _generate_3_row_chunk_horizontal(
  x_start: int, x_end: int, y_rows: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-row horizontal chunk.

  y_rows should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  return steps


def _create_vertical_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for vertical strip."""
  steps: list[GenerationStep] = []
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
        y_start, y_end, x_cols, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 2
    else:
      # Process 1 column using depth-1 formula
      x = x_positions[col_offset]
      chunk_bounds = StripBounds(
        Point(x, y_start),
        Point(x, y_end),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 1

  return steps


def _generate_3_col_chunk_vertical(
  y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

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
  x_start: int, x_end: int, y_rows: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-row horizontal chunk.

  y_rows should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  return steps


def _create_vertical_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for vertical strip."""
  steps: list[GenerationStep] = []
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
        y_start, y_end, x_cols, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 2
    else:
      # Process 1 column using depth-1 formula
      x = x_positions[col_offset]
      chunk_bounds = StripBounds(
        Point(x, y_start),
        Point(x, y_end),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 1

  return steps


def _generate_3_col_chunk_vertical(
  y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start_offset
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  # Step 5: If we offset the start, fill in the gap column at the start
  if x_start_offset != x_start:
    # Fill the 2x2 area rows
    steps.append(
      GenerationStep(
        [
          Point(x_start, y_2x2_top),
          Point(x_start, y_2x2_bottom),
        ]
      )
    )
    # Fill the close row
    steps.append(GenerationStep([Point(x_start, y_close)]))

  return steps


def _create_vertical_depth_3_plus_plan(
  bounds: StripBounds, generation_edge: Edge
) -> list[GenerationStep]:
  """Create depth-3+ plan for vertical strip."""
  steps: list[GenerationStep] = []
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
        y_start, y_end, x_cols, generation_edge
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
      chunk_steps = create_depth_2_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 2
    else:
      # Process 1 column using depth-1 formula
      x = x_positions[col_offset]
      chunk_bounds = StripBounds(
        Point(x, y_start),
        Point(x, y_end),
      )
      chunk_steps = create_depth_1_plan(chunk_bounds, generation_edge)
      steps.extend(chunk_steps)
      col_offset += 1

  return steps


def _generate_3_col_chunk_vertical(
  y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start_offset
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  # Step 5: If we offset the start, fill in the gap column at the start
  if x_start_offset != x_start:
    # Fill the 2x2 area rows
    steps.append(
      GenerationStep(
        [
          Point(x_start, y_2x2_top),
          Point(x_start, y_2x2_bottom),
        ]
      )
    )
    # Fill the close row
    steps.append(GenerationStep([Point(x_start, y_close)]))

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
  y_start: int, y_end: int, x_cols: list[int], generation_edge: Edge
) -> list[GenerationStep]:
  """
  Generate steps for a 3-column vertical chunk.

  x_cols should be ordered from closest to farthest from the generation edge.
  """
  steps: list[GenerationStep] = []

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start_offset
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  # Step 5: If we offset the start, fill in the gap column at the start
  if x_start_offset != x_start:
    # Fill the 2x2 area rows
    steps.append(
      GenerationStep(
        [
          Point(x_start, y_2x2_top),
          Point(x_start, y_2x2_bottom),
        ]
      )
    )
    # Fill the close row
    steps.append(GenerationStep([Point(x_start, y_close)]))

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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start_offset
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

  # Step 5: If we offset the start, fill in the gap row at the start
  if y_start_offset != y_start:
    # Fill the 2x2 area columns
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y_start),
          Point(x_2x2_right, y_start),
        ]
      )
    )
    # Fill the close column
    steps.append(GenerationStep([Point(x_close, y_start)]))

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

  # Create plan
  plan = create_strip_plan(bounds, edge)

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    # Pattern: SS.SS.SS... (S=selected, .=gap)
    x = x_start_2x1
    while x + 1 <= x_end:
      steps.append(GenerationStep([Point(x, y), Point(x + 1, y)]))
      x += 3  # Move by 3 (2 selected + 1 gap)

    # Phase 2: Fill single-quadrant gaps (between the 2x1 tiles)
    x = x_start_2x1 + 2  # First gap position after 2x1 tiles
    while x <= x_end:
      steps.append(GenerationStep([Point(x, y)]))
      x += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if x_start_2x1 != x_start:
      steps.append(GenerationStep([Point(x_start, y)]))

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

    # Phase 1: Generate 2x1 quadrants with 1 gap
    y = y_start_2x1
    while y + 1 <= y_end:
      steps.append(GenerationStep([Point(x, y), Point(x, y + 1)]))
      y += 3

    # Phase 2: Fill single-quadrant gaps
    y = y_start_2x1 + 2
    while y <= y_end:
      steps.append(GenerationStep([Point(x, y)]))
      y += 3

    # Phase 3: If we offset the start, fill in the gap at the start
    if y_start_2x1 != y_start:
      steps.append(GenerationStep([Point(x, y_start)]))

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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start_offset
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  # Step 5: If we offset the start, fill in the gap column at the start
  if x_start_offset != x_start:
    # Fill the 2x2 area rows
    steps.append(
      GenerationStep(
        [
          Point(x_start, y_2x2_top),
          Point(x_start, y_2x2_bottom),
        ]
      )
    )
    # Fill the close row
    steps.append(GenerationStep([Point(x_start, y_close)]))

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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start_offset
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

  # Step 5: If we offset the start, fill in the gap row at the start
  if y_start_offset != y_start:
    # Fill the 2x2 area columns
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y_start),
          Point(x_2x2_right, y_start),
        ]
      )
    )
    # Fill the close column
    steps.append(GenerationStep([Point(x_close, y_start)]))

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

Now let me update the tests to add coverage for the seam avoidance logic:

```python
"""
Tests for make_strip_plan.py

These tests verify the strip generation planning algorithms without
needing a database connection.
"""

import pytest

from isometric_nyc.e2e_generation.make_strip_plan import (
  Edge,
  GenerationStep,
  Point,
  StepStatus,
  StripBounds,
  create_depth_1_plan,
  create_depth_2_plan,
  create_depth_3_plus_plan,
  create_strip_plan,
  find_generation_edge,
  get_exterior_neighbors,
  is_edge_fully_generated,
)

# =============================================================================
# Point Tests
# =============================================================================


class TestPoint:
  def test_str(self) -> None:
    p = Point(3, 5)
    assert str(p) == "(3,5)"

  def test_add(self) -> None:
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = p1 + p2
    assert result == Point(4, 6)

  def test_from_string_simple(self) -> None:
    p = Point.from_string("3,5")
    assert p == Point(3, 5)

  def test_from_string_with_parens(self) -> None:
    p = Point.from_string("(3,5)")
    assert p == Point(3, 5)

  def test_from_string_with_spaces(self) -> None:
    p = Point.from_string("( 3 , 5 )")
    assert p == Point(3, 5)

  def test_from_string_negative(self) -> None:
    p = Point.from_string("-3,-5")
    assert p == Point(-3, -5)

  def test_from_string_invalid(self) -> None:
    with pytest.raises(ValueError):
      Point.from_string("invalid")


# =============================================================================
# StripBounds Tests
# =============================================================================


class TestStripBounds:
  def test_width_height(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert bounds.width == 11
    assert bounds.height == 3

  def test_is_horizontal(self) -> None:
    # Horizontal strip (wider than tall)
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.is_horizontal is True

    # Vertical strip (taller than wide)
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.is_horizontal is False

    # Square (treated as horizontal)
    sq_bounds = StripBounds(Point(0, 0), Point(5, 5))
    assert sq_bounds.is_horizontal is True

  def test_depth(self) -> None:
    # Horizontal strip: depth = height
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.depth == 3

    # Vertical strip: depth = width
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.depth == 3

  def test_length(self) -> None:
    # Horizontal strip: length = width
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.length == 11

    # Vertical strip: length = height
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.length == 11

  def test_all_points(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 1))
    points = bounds.all_points()
    expected = [
      Point(0, 0),
      Point(1, 0),
      Point(2, 0),
      Point(0, 1),
      Point(1, 1),
      Point(2, 1),
    ]
    assert points == expected


# =============================================================================
# Edge Detection Tests
# =============================================================================


class TestGetExteriorNeighbors:
  def test_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.TOP)
    assert neighbors == [Point(0, -1), Point(1, -1), Point(2, -1)]

  def test_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.BOTTOM)
    assert neighbors == [Point(0, 1), Point(1, 1), Point(2, 1)]

  def test_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.LEFT)
    assert neighbors == [Point(-1, 0), Point(-1, 1), Point(-1, 2)]

  def test_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.RIGHT)
    assert neighbors == [Point(1, 0), Point(1, 1), Point(1, 2)]


class TestIsEdgeFullyGenerated:
  def test_fully_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is True

  def test_partially_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(2, -1)}  # Missing Point(1, -1)
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False

  def test_not_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated: set[Point] = set()
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False


class TestFindGenerationEdge:
  def test_horizontal_strip_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants below the strip
    generated = {Point(x, 1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.BOTTOM

  def test_horizontal_strip_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants above the strip
    generated = {Point(x, -1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.TOP

  def test_vertical_strip_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the left
    generated = {Point(-1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.LEFT

  def test_vertical_strip_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the right
    generated = {Point(1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.RIGHT

  def test_no_valid_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    generated: set[Point] = set()  # Nothing generated
    edge = find_generation_edge(bounds, generated)
    assert edge is None


# =============================================================================
# Depth 1 Plan Tests
# =============================================================================


class TestDepth1Plan:
  def test_horizontal_depth_1_simple(self) -> None:
    """Test a simple 11-wide depth-1 horizontal strip."""
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Should have 2x1 tiles first, then single fills
    # Pattern: SS.SS.SS.SS (positions 0-1, 3-4, 6-7, 9-10)
    # Then fills: positions 2, 5, 8

    # Extract all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 11 positions
    expected = {Point(x, 0) for x in range(11)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_1_pattern(self) -> None:
    """Verify the 2x1 + gap + fill pattern."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Phase 1: 2x1 tiles at positions 0-1, 3-4, 6-7
    # Phase 2: Single tiles at positions 2, 5

    # First step should be 2x1 at (0,0)
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

    # Second step should be 2x1 at (3,0)
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]

    # Third step should be 2x1 at (6,0)
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]

    # Fourth step should be single at (2,0)
    assert plan[3].quadrants == [Point(2, 0)]

    # Fifth step should be single at (5,0)
    assert plan[4].quadrants == [Point(5, 0)]

  def test_vertical_depth_1(self) -> None:
    """Test a vertical depth-1 strip."""
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    plan = create_depth_1_plan(bounds, Edge.LEFT)

    # Should have 2-quadrant tiles first, then singles
    # Pattern: SS.SS.SS.SS (y positions 0-1, 3-4, 6-7)
    # Then fills: positions 2, 5

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected


# =============================================================================
# Depth 2 Plan Tests
# =============================================================================


class TestDepth2Plan:
  def test_horizontal_depth_2_coverage(self) -> None:
    """Test that depth-2 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_2_order(self) -> None:
    """Test that depth-2 processes row closest to edge first."""
    bounds = StripBounds(Point(0, 0), Point(4, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    # With BOTTOM edge, should process y=1 first, then y=0
    # First steps should all be at y=1
    first_row_steps = []
    for step in plan:
      if all(q.y == 1 for q in step.quadrants):
        first_row_steps.append(step)
      elif len(first_row_steps) > 0:
        break  # Found a different row, stop

    assert len(first_row_steps) > 0
    assert all(q.y == 1 for step in first_row_steps for q in step.quadrants)


# =============================================================================
# Depth 3+ Plan Tests
# =============================================================================


class TestDepth3PlusPlan:
  def test_horizontal_depth_3_coverage(self) -> None:
    """Test that depth-3 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

  def test_depth_3_has_2x2_tiles(self) -> None:
    """Test that depth-3 plan uses 2x2 tiles."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Should have at least one 2x2 tile
    has_2x2 = any(len(step.quadrants) == 4 for step in plan)
    assert has_2x2, "Depth-3 plan should contain 2x2 tiles"

  def test_depth_4_coverage(self) -> None:
    """Test that depth-4 (3+1) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 3))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(4)}
    assert set(all_quadrants) == expected

  def test_depth_5_coverage(self) -> None:
    """Test that depth-5 (3+2) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 4))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(5)}
    assert set(all_quadrants) == expected

  def test_depth_6_coverage(self) -> None:
    """Test that depth-6 (3+3) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 5))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(6)}
    assert set(all_quadrants) == expected


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateStripPlan:
  def test_depth_1_uses_depth_1_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 1 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_2_uses_depth_2_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 1))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 2 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_3_uses_depth_3_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 3 should have some 4-quadrant tiles
    has_4_quad = any(len(step.quadrants) == 4 for step in plan)
    assert has_4_quad


# =============================================================================
# GenerationStep Serialization Tests
# =============================================================================


class TestGenerationStepSerialization:
  def test_to_dict(self) -> None:
    step = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0)],
      status=StepStatus.PENDING,
    )
    d = step.to_dict()
    assert d["quadrants"] == "(0,0),(1,0)"
    assert d["status"] == "pending"

  def test_from_dict(self) -> None:
    d = {
      "quadrants": "(0,0),(1,0)",
      "status": "done",
    }
    step = GenerationStep.from_dict(d)
    assert step.quadrants == [Point(0, 0), Point(1, 0)]
    assert step.status == StepStatus.DONE

  def test_roundtrip(self) -> None:
    original = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)],
      status=StepStatus.PENDING,
    )
    d = original.to_dict()
    restored = GenerationStep.from_dict(d)
    assert restored.quadrants == original.quadrants
    assert restored.status == original.status


# =============================================================================
# Seam Avoidance Tests
# =============================================================================


class TestSeamAvoidance:
  """Tests for seam avoidance when there's a generated quadrant at strip ends."""

  def test_depth_1_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip with generated quadrant to the left.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    # Generated quadrant at (-1, 0) - to the left of the strip
    generated = {Point(-1, 0)}
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # With left neighbor, should start at x=1 instead of x=0
    # So first 2x1 should be at (1,0),(2,0), not (0,0),(1,0)
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]

    # The gap at x=0 should be filled last
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should still cover all positions
    expected = {Point(x, 0) for x in range(8)}
    assert set(all_quadrants) == expected

    # x=0 should be generated as a single quadrant (last step)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    assert Point(0, 0) in [q for s in single_steps for q in s.quadrants]

  def test_depth_1_horizontal_no_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip without left neighbor.
    Should NOT offset start.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated: set[Point] = set()  # No neighbors
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # Without left neighbor, should start at x=0
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

  def test_depth_1_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-1 vertical strip with generated quadrant above.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    # Generated quadrant at (0, -1) - above the strip
    generated = {Point(0, -1)}
    plan = create_depth_1_plan(bounds, Edge.LEFT, generated)

    # With top neighbor, should start at y=1 instead of y=0
    assert plan[0].quadrants == [Point(0, 1), Point(0, 2)]

    # Should still cover all positions
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected

  def test_depth_2_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-2 horizontal strip with left neighbor.
    Both rows should be offset.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    # Generated quadrants to the left of both rows
    generated = {Point(-1, 0), Point(-1, 1)}
    plan = create_depth_2_plan(bounds, Edge.BOTTOM, generated)

    # First 2x1 in each row should be offset
    # (With BOTTOM edge, row y=1 is processed first)
    # First step should start at x=1, not x=0
    assert plan[0].quadrants[0].x == 1

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_depth_3_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-3 horizontal strip with left neighbor.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    # Generated quadrants to the left
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # First 2x2 should be offset to start at x=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.x >= 1 for q in first_4_quad.quadrants)

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

  def test_depth_3_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-3 vertical strip with top neighbor.
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    # Generated quadrants above
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # First 2x2 should be offset to start at y=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.y >= 1 for q in first_4_quad.quadrants)

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(3) for y in range(8)}
    assert set(all_quadrants) == expected

  def test_create_strip_plan_passes_generated(self) -> None:
    """Test that create_strip_plan passes generated set correctly."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated = {Point(-1, 0)}  # Left neighbor
    plan = create_strip_plan(bounds, Edge.BOTTOM, generated)

    # Should be offset due to left neighbor
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]


# =============================================================================
# Task Example Tests (from 019_strip_plan.md)
# =============================================================================


class TestTaskExamples:
  def test_depth_1_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(10,0), depth=1

    Expected pattern:
    Step 1: (0,0),(1,0)
    Step 2: (3,0),(4,0)
    Step 3: (6,0),(7,0)
    Step 4: (9,0),(10,0)
    Step 5: (2,0)
    Step 6: (5,0)
    Step 7: (8,0)
    """
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Verify first 4 steps are 2x1 tiles
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]
    assert plan[3].quadrants == [Point(9, 0), Point(10, 0)]

    # Verify remaining steps are single quadrants
    assert plan[4].quadrants == [Point(2, 0)]
    assert plan[5].quadrants == [Point(5, 0)]
    assert plan[6].quadrants == [Point(8, 0)]

    # Total should be 7 steps
    assert len(plan) == 7

  def test_depth_3_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(7,2), depth=3

    This tests the general pattern of:
    1. 2x2 tiles away from edge
    2. 1x2 bridges between 2x2s
    3. 2x1 bridges back to edge
    4. Single quadrant fills
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Collect all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 24 quadrants (8 x 3)
    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

    # First steps should be 2x2 tiles
    four_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    assert len(four_quad_steps) >= 2, "Should have at least 2 2x2 tiles"

```

The tests reveal a bug - when we offset the start, the pattern doesn't properly cover the end of the strip. Let me fix the algorithm to handle the tail correctly:

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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
    x += 3

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_2x2_top),
          Point(x, y_2x2_bottom),
        ]
      )
    )
    x += 3

  # Step 3: Generate 2x1 horizontal bridges in the close row
  x = x_start_offset
  while x + 1 <= x_end:
    steps.append(
      GenerationStep(
        [
          Point(x, y_close),
          Point(x + 1, y_close),
        ]
      )
    )
    x += 3

  # Step 4: Fill remaining single-quadrant gaps in the close row
  x = x_start_offset + 2
  while x <= x_end:
    steps.append(GenerationStep([Point(x, y_close)]))
    x += 3

  # Step 5: If we offset the start, fill in the gap column at the start
  if x_start_offset != x_start:
    # Fill the 2x2 area rows
    steps.append(
      GenerationStep(
        [
          Point(x_start, y_2x2_top),
          Point(x_start, y_2x2_bottom),
        ]
      )
    )
    # Fill the close row
    steps.append(GenerationStep([Point(x_start, y_close)]))

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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start_offset
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

  # Step 5: If we offset the start, fill in the gap row at the start
  if y_start_offset != y_start:
    # Fill the 2x2 area columns
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y_start),
          Point(x_2x2_right, y_start),
        ]
      )
    )
    # Fill the close column
    steps.append(GenerationStep([Point(x_close, y_start)]))

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Track covered x positions for each row type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles (gaps in 2x2 rows)
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
  x = x_start_offset
  while x + 1 <= x_end:
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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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
    y += 3

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y),
          Point(x_2x2_right, y),
        ]
      )
    )
    y += 3

  # Step 3: Generate 1x2 vertical bridges in the close column
  y = y_start_offset
  while y + 1 <= y_end:
    steps.append(
      GenerationStep(
        [
          Point(x_close, y),
          Point(x_close, y + 1),
        ]
      )
    )
    y += 3

  # Step 4: Fill remaining single-quadrant gaps in the close column
  y = y_start_offset + 2
  while y <= y_end:
    steps.append(GenerationStep([Point(x_close, y)]))
    y += 3

  # Step 5: If we offset the start, fill in the gap row at the start
  if y_start_offset != y_start:
    # Fill the 2x2 area columns
    steps.append(
      GenerationStep(
        [
          Point(x_2x2_left, y_start),
          Point(x_2x2_right, y_start),
        ]
      )
    )
    # Fill the close column
    steps.append(GenerationStep([Point(x_close, y_start)]))

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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Track covered x positions for each row type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles (gaps in 2x2 rows)
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
  x = x_start_offset
  while x + 1 <= x_end:
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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # x_cols[0] = closest to edge
  # x_cols[2] = farthest from edge

  x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
  x_far_2 = x_cols[1]
  x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]

  x_2x2_left = min(x_far_1, x_far_2)
  x_2x2_right = max(x_far_1, x_far_2)

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Track covered y positions for each column type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles (gaps in 2x2 cols)
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
  y = y_start_offset
  while y + 1 <= y_end:
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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

All 49 tests pass. Let me format the code:

Done! I've updated the strip plan algorithm to avoid seams when there's a generated quadrant at the start of the strip (perpendicular to the generation edge).

## Changes Made

### Seam Avoidance Logic

For depth-1 and depth-2 strips:
- Before generating 2×1 tiles, check if there's a generated quadrant to the **left** (for horizontal strips) or **above** (for vertical strips) of the strip start
- If found, offset the first 2×1 tile by 1 quadrant to create a gap
- Fill all remaining gaps (including the offset position) as single quadrants

For depth-3+ strips:
- Same logic applied to both the 2×2 tile generation and the 2×1 bridge generation
- The algorithm now tracks covered positions and fills all gaps at the end

### Function Signature Changes

All plan creation functions now accept an optional `generated: set[Point] | None` parameter:
- `create_strip_plan(bounds, edge, generated)`
- `create_depth_1_plan(bounds, edge, generated)`
- `create_depth_2_plan(bounds, edge, generated)`  
- `create_depth_3_plus_plan(bounds, edge, generated)`
- Internal chunk functions also updated

### New Tests Added

7 new tests in `TestSeamAvoidance` class:
- `test_depth_1_horizontal_with_left_neighbor` - verifies offset and coverage
- `test_depth_1_horizontal_no_left_neighbor` - verifies no offset when not needed
- `test_depth_1_vertical_with_top_neighbor` - vertical equivalent
- `test_depth_2_horizontal_with_left_neighbor` - depth-2 offset
- `test_depth_3_horizontal_with_left_neighbor` - depth-3 offset
- `test_depth_3_vertical_with_top_neighbor` - vertical depth-3
- `test_create_strip_plan_passes_generated` - integration test

All **49 tests pass**.

---

**User**

Ensure that the same rule applies to depth 3 strips

---

**Cursor**

Let me verify the current depth-3 implementation and add more explicit tests to ensure the seam avoidance rule applies to all parts of the depth-3 algorithm (including the 2×1 bridges in the close row):

The implementation already applies the seam avoidance to depth-3 strips - I can see `x_start_offset` is used for both the 2×2 tiles (line 541) and the 2×1 bridges in the close row (line 571). Let me add more comprehensive tests to verify all parts of the depth-3 algorithm respect the seam avoidance:

```python
"""
Tests for make_strip_plan.py

These tests verify the strip generation planning algorithms without
needing a database connection.
"""

import pytest

from isometric_nyc.e2e_generation.make_strip_plan import (
  Edge,
  GenerationStep,
  Point,
  StepStatus,
  StripBounds,
  create_depth_1_plan,
  create_depth_2_plan,
  create_depth_3_plus_plan,
  create_strip_plan,
  find_generation_edge,
  get_exterior_neighbors,
  is_edge_fully_generated,
)

# =============================================================================
# Point Tests
# =============================================================================


class TestPoint:
  def test_str(self) -> None:
    p = Point(3, 5)
    assert str(p) == "(3,5)"

  def test_add(self) -> None:
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = p1 + p2
    assert result == Point(4, 6)

  def test_from_string_simple(self) -> None:
    p = Point.from_string("3,5")
    assert p == Point(3, 5)

  def test_from_string_with_parens(self) -> None:
    p = Point.from_string("(3,5)")
    assert p == Point(3, 5)

  def test_from_string_with_spaces(self) -> None:
    p = Point.from_string("( 3 , 5 )")
    assert p == Point(3, 5)

  def test_from_string_negative(self) -> None:
    p = Point.from_string("-3,-5")
    assert p == Point(-3, -5)

  def test_from_string_invalid(self) -> None:
    with pytest.raises(ValueError):
      Point.from_string("invalid")


# =============================================================================
# StripBounds Tests
# =============================================================================


class TestStripBounds:
  def test_width_height(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert bounds.width == 11
    assert bounds.height == 3

  def test_is_horizontal(self) -> None:
    # Horizontal strip (wider than tall)
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.is_horizontal is True

    # Vertical strip (taller than wide)
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.is_horizontal is False

    # Square (treated as horizontal)
    sq_bounds = StripBounds(Point(0, 0), Point(5, 5))
    assert sq_bounds.is_horizontal is True

  def test_depth(self) -> None:
    # Horizontal strip: depth = height
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.depth == 3

    # Vertical strip: depth = width
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.depth == 3

  def test_length(self) -> None:
    # Horizontal strip: length = width
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.length == 11

    # Vertical strip: length = height
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.length == 11

  def test_all_points(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 1))
    points = bounds.all_points()
    expected = [
      Point(0, 0),
      Point(1, 0),
      Point(2, 0),
      Point(0, 1),
      Point(1, 1),
      Point(2, 1),
    ]
    assert points == expected


# =============================================================================
# Edge Detection Tests
# =============================================================================


class TestGetExteriorNeighbors:
  def test_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.TOP)
    assert neighbors == [Point(0, -1), Point(1, -1), Point(2, -1)]

  def test_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.BOTTOM)
    assert neighbors == [Point(0, 1), Point(1, 1), Point(2, 1)]

  def test_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.LEFT)
    assert neighbors == [Point(-1, 0), Point(-1, 1), Point(-1, 2)]

  def test_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.RIGHT)
    assert neighbors == [Point(1, 0), Point(1, 1), Point(1, 2)]


class TestIsEdgeFullyGenerated:
  def test_fully_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is True

  def test_partially_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(2, -1)}  # Missing Point(1, -1)
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False

  def test_not_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated: set[Point] = set()
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False


class TestFindGenerationEdge:
  def test_horizontal_strip_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants below the strip
    generated = {Point(x, 1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.BOTTOM

  def test_horizontal_strip_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants above the strip
    generated = {Point(x, -1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.TOP

  def test_vertical_strip_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the left
    generated = {Point(-1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.LEFT

  def test_vertical_strip_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the right
    generated = {Point(1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.RIGHT

  def test_no_valid_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    generated: set[Point] = set()  # Nothing generated
    edge = find_generation_edge(bounds, generated)
    assert edge is None


# =============================================================================
# Depth 1 Plan Tests
# =============================================================================


class TestDepth1Plan:
  def test_horizontal_depth_1_simple(self) -> None:
    """Test a simple 11-wide depth-1 horizontal strip."""
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Should have 2x1 tiles first, then single fills
    # Pattern: SS.SS.SS.SS (positions 0-1, 3-4, 6-7, 9-10)
    # Then fills: positions 2, 5, 8

    # Extract all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 11 positions
    expected = {Point(x, 0) for x in range(11)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_1_pattern(self) -> None:
    """Verify the 2x1 + gap + fill pattern."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Phase 1: 2x1 tiles at positions 0-1, 3-4, 6-7
    # Phase 2: Single tiles at positions 2, 5

    # First step should be 2x1 at (0,0)
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

    # Second step should be 2x1 at (3,0)
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]

    # Third step should be 2x1 at (6,0)
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]

    # Fourth step should be single at (2,0)
    assert plan[3].quadrants == [Point(2, 0)]

    # Fifth step should be single at (5,0)
    assert plan[4].quadrants == [Point(5, 0)]

  def test_vertical_depth_1(self) -> None:
    """Test a vertical depth-1 strip."""
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    plan = create_depth_1_plan(bounds, Edge.LEFT)

    # Should have 2-quadrant tiles first, then singles
    # Pattern: SS.SS.SS.SS (y positions 0-1, 3-4, 6-7)
    # Then fills: positions 2, 5

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected


# =============================================================================
# Depth 2 Plan Tests
# =============================================================================


class TestDepth2Plan:
  def test_horizontal_depth_2_coverage(self) -> None:
    """Test that depth-2 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_2_order(self) -> None:
    """Test that depth-2 processes row closest to edge first."""
    bounds = StripBounds(Point(0, 0), Point(4, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    # With BOTTOM edge, should process y=1 first, then y=0
    # First steps should all be at y=1
    first_row_steps = []
    for step in plan:
      if all(q.y == 1 for q in step.quadrants):
        first_row_steps.append(step)
      elif len(first_row_steps) > 0:
        break  # Found a different row, stop

    assert len(first_row_steps) > 0
    assert all(q.y == 1 for step in first_row_steps for q in step.quadrants)


# =============================================================================
# Depth 3+ Plan Tests
# =============================================================================


class TestDepth3PlusPlan:
  def test_horizontal_depth_3_coverage(self) -> None:
    """Test that depth-3 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

  def test_depth_3_has_2x2_tiles(self) -> None:
    """Test that depth-3 plan uses 2x2 tiles."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Should have at least one 2x2 tile
    has_2x2 = any(len(step.quadrants) == 4 for step in plan)
    assert has_2x2, "Depth-3 plan should contain 2x2 tiles"

  def test_depth_4_coverage(self) -> None:
    """Test that depth-4 (3+1) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 3))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(4)}
    assert set(all_quadrants) == expected

  def test_depth_5_coverage(self) -> None:
    """Test that depth-5 (3+2) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 4))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(5)}
    assert set(all_quadrants) == expected

  def test_depth_6_coverage(self) -> None:
    """Test that depth-6 (3+3) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 5))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(6)}
    assert set(all_quadrants) == expected


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateStripPlan:
  def test_depth_1_uses_depth_1_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 1 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_2_uses_depth_2_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 1))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 2 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_3_uses_depth_3_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 3 should have some 4-quadrant tiles
    has_4_quad = any(len(step.quadrants) == 4 for step in plan)
    assert has_4_quad


# =============================================================================
# GenerationStep Serialization Tests
# =============================================================================


class TestGenerationStepSerialization:
  def test_to_dict(self) -> None:
    step = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0)],
      status=StepStatus.PENDING,
    )
    d = step.to_dict()
    assert d["quadrants"] == "(0,0),(1,0)"
    assert d["status"] == "pending"

  def test_from_dict(self) -> None:
    d = {
      "quadrants": "(0,0),(1,0)",
      "status": "done",
    }
    step = GenerationStep.from_dict(d)
    assert step.quadrants == [Point(0, 0), Point(1, 0)]
    assert step.status == StepStatus.DONE

  def test_roundtrip(self) -> None:
    original = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)],
      status=StepStatus.PENDING,
    )
    d = original.to_dict()
    restored = GenerationStep.from_dict(d)
    assert restored.quadrants == original.quadrants
    assert restored.status == original.status


# =============================================================================
# Seam Avoidance Tests
# =============================================================================


class TestSeamAvoidance:
  """Tests for seam avoidance when there's a generated quadrant at strip ends."""

  def test_depth_1_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip with generated quadrant to the left.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    # Generated quadrant at (-1, 0) - to the left of the strip
    generated = {Point(-1, 0)}
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # With left neighbor, should start at x=1 instead of x=0
    # So first 2x1 should be at (1,0),(2,0), not (0,0),(1,0)
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]

    # The gap at x=0 should be filled last
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should still cover all positions
    expected = {Point(x, 0) for x in range(8)}
    assert set(all_quadrants) == expected

    # x=0 should be generated as a single quadrant (last step)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    assert Point(0, 0) in [q for s in single_steps for q in s.quadrants]

  def test_depth_1_horizontal_no_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip without left neighbor.
    Should NOT offset start.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated: set[Point] = set()  # No neighbors
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # Without left neighbor, should start at x=0
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

  def test_depth_1_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-1 vertical strip with generated quadrant above.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    # Generated quadrant at (0, -1) - above the strip
    generated = {Point(0, -1)}
    plan = create_depth_1_plan(bounds, Edge.LEFT, generated)

    # With top neighbor, should start at y=1 instead of y=0
    assert plan[0].quadrants == [Point(0, 1), Point(0, 2)]

    # Should still cover all positions
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected

  def test_depth_2_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-2 horizontal strip with left neighbor.
    Both rows should be offset.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    # Generated quadrants to the left of both rows
    generated = {Point(-1, 0), Point(-1, 1)}
    plan = create_depth_2_plan(bounds, Edge.BOTTOM, generated)

    # First 2x1 in each row should be offset
    # (With BOTTOM edge, row y=1 is processed first)
    # First step should start at x=1, not x=0
    assert plan[0].quadrants[0].x == 1

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_depth_3_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-3 horizontal strip with left neighbor.
    All generation steps should respect the offset:
    - 2x2 tiles should start at x=1
    - 2x1 bridges in close row should start at x=1
    - x=0 should be filled last as single quadrants
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    # Generated quadrants to the left
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # First 2x2 should be offset to start at x=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.x >= 1 for q in first_4_quad.quadrants), "2x2 tiles should be offset"

    # All 2x2 tiles should not touch x=0
    all_4_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    for step in all_4_quad_steps:
      assert all(q.x >= 1 for q in step.quadrants), f"2x2 at x=0 would create seam: {step.quadrants}"

    # Find 2x1 horizontal tiles (2 quadrants in same row)
    two_quad_horizontal = [
      s for s in plan
      if len(s.quadrants) == 2 and s.quadrants[0].y == s.quadrants[1].y
    ]
    # First 2x1 horizontal should be offset (not starting at x=0)
    if two_quad_horizontal:
      first_2x1_h = two_quad_horizontal[0]
      assert first_2x1_h.quadrants[0].x >= 1, "First 2x1 horizontal should be offset"

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

    # x=0 should be generated as single quadrants (filled last)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    x0_singles = [s for s in single_steps if s.quadrants[0].x == 0]
    assert len(x0_singles) >= 1, "x=0 positions should be filled as singles"

  def test_depth_3_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-3 vertical strip with top neighbor.
    All generation steps should respect the offset.
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    # Generated quadrants above
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # First 2x2 should be offset to start at y=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.y >= 1 for q in first_4_quad.quadrants), "2x2 tiles should be offset"

    # All 2x2 tiles should not touch y=0
    all_4_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    for step in all_4_quad_steps:
      assert all(q.y >= 1 for q in step.quadrants), f"2x2 at y=0 would create seam: {step.quadrants}"

    # Find 1x2 vertical tiles (2 quadrants in same column)
    two_quad_vertical = [
      s for s in plan
      if len(s.quadrants) == 2 and s.quadrants[0].x == s.quadrants[1].x
    ]
    # First 1x2 vertical should be offset (not starting at y=0)
    if two_quad_vertical:
      first_1x2_v = two_quad_vertical[0]
      assert first_1x2_v.quadrants[0].y >= 1, "First 1x2 vertical should be offset"

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(3) for y in range(8)}
    assert set(all_quadrants) == expected

    # y=0 should be generated as single quadrants (filled last)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    y0_singles = [s for s in single_steps if s.quadrants[0].y == 0]
    assert len(y0_singles) >= 1, "y=0 positions should be filled as singles"

  def test_depth_3_all_2x1_bridges_offset_horizontal(self) -> None:
    """
    Verify that in depth-3 horizontal strip with left neighbor,
    the 2x1 bridges in the close row are also offset.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # With BOTTOM edge, close row is y=2 (closest to bottom)
    close_row_y = 2

    # Find 2x1 tiles in the close row
    close_row_2x1 = [
      s for s in plan
      if len(s.quadrants) == 2
      and all(q.y == close_row_y for q in s.quadrants)
      and s.quadrants[0].x + 1 == s.quadrants[1].x  # horizontal pair
    ]

    # First 2x1 in close row should start at x=1, not x=0
    if close_row_2x1:
      first_bridge = close_row_2x1[0]
      assert first_bridge.quadrants[0].x >= 1, (
        f"First 2x1 bridge in close row should be offset, "
        f"but starts at x={first_bridge.quadrants[0].x}"
      )

  def test_depth_3_all_1x2_bridges_offset_vertical(self) -> None:
    """
    Verify that in depth-3 vertical strip with top neighbor,
    the 1x2 bridges in the close column are also offset.
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # With LEFT edge, close column is x=2 (farthest from left)
    close_col_x = 2

    # Find 1x2 tiles in the close column
    close_col_1x2 = [
      s for s in plan
      if len(s.quadrants) == 2
      and all(q.x == close_col_x for q in s.quadrants)
      and s.quadrants[0].y + 1 == s.quadrants[1].y  # vertical pair
    ]

    # First 1x2 in close column should start at y=1, not y=0
    if close_col_1x2:
      first_bridge = close_col_1x2[0]
      assert first_bridge.quadrants[0].y >= 1, (
        f"First 1x2 bridge in close column should be offset, "
        f"but starts at y={first_bridge.quadrants[0].y}"
      )

  def test_create_strip_plan_passes_generated(self) -> None:
    """Test that create_strip_plan passes generated set correctly."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated = {Point(-1, 0)}  # Left neighbor
    plan = create_strip_plan(bounds, Edge.BOTTOM, generated)

    # Should be offset due to left neighbor
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]


# =============================================================================
# Task Example Tests (from 019_strip_plan.md)
# =============================================================================


class TestTaskExamples:
  def test_depth_1_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(10,0), depth=1

    Expected pattern:
    Step 1: (0,0),(1,0)
    Step 2: (3,0),(4,0)
    Step 3: (6,0),(7,0)
    Step 4: (9,0),(10,0)
    Step 5: (2,0)
    Step 6: (5,0)
    Step 7: (8,0)
    """
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Verify first 4 steps are 2x1 tiles
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]
    assert plan[3].quadrants == [Point(9, 0), Point(10, 0)]

    # Verify remaining steps are single quadrants
    assert plan[4].quadrants == [Point(2, 0)]
    assert plan[5].quadrants == [Point(5, 0)]
    assert plan[6].quadrants == [Point(8, 0)]

    # Total should be 7 steps
    assert len(plan) == 7

  def test_depth_3_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(7,2), depth=3

    This tests the general pattern of:
    1. 2x2 tiles away from edge
    2. 1x2 bridges between 2x2s
    3. 2x1 bridges back to edge
    4. Single quadrant fills
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Collect all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 24 quadrants (8 x 3)
    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

    # First steps should be 2x2 tiles
    four_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    assert len(four_quad_steps) >= 2, "Should have at least 2 2x2 tiles"

```

All seam avoidance tests pass. Let me run the full test suite to confirm everything works:

All **51 tests pass**. The depth-3 strip algorithm already correctly applies the seam avoidance rule. I've added 2 new tests to explicitly verify this:

### New Tests Added

1. **`test_depth_3_all_2x1_bridges_offset_horizontal`** - Verifies that in horizontal depth-3 strips with a left neighbor, the 2×1 bridges in the close row are also offset (start at x=1, not x=0)

2. **`test_depth_3_all_1x2_bridges_offset_vertical`** - Verifies that in vertical depth-3 strips with a top neighbor, the 1×2 bridges in the close column are also offset (start at y=1, not y=0)

### How Depth-3 Seam Avoidance Works

In `_generate_3_row_chunk_horizontal`:
- Line 532-533 checks for left neighbors and sets `x_start_offset`
- Line 541 uses `x_start_offset` for 2×2 tiles
- Line 571 uses `x_start_offset` for 2×1 bridges in the close row
- Lines 558-568 and 583-585 fill all remaining gaps (including x=0) as single quadrants or 1×2 vertical tiles

The same pattern is applied in `_generate_3_col_chunk_vertical` for vertical strips.

---

**User**

OK - remember though that a 2x2 tile for a 3 deep strip MUST be 1 quadrant away from all edges

---

**Cursor**

```plaintext
Step 1 - (0,0),(1,0),(0,1),(1,1)
S S x x x x x x
S S x x x x x x
x x x x x x x x
G G G G G G G G
```

```python
y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
y_far_2 = y_rows[1]
y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]
```

You're right - let me verify and fix the depth-3 implementation. The 2×2 tiles need to maintain a 1-quadrant gap from:
1. The generation edge (the row/column closest to generated content)
2. Any perpendicular edge with generated neighbors

Let me check the current implementation:

I see the current implementation places 2×2 tiles in the two rows farthest from the generation edge (y=0,1) with row 2 as the gap. Let me add explicit tests to verify the "1 quadrant gap from all edges" requirement:

```python
"""
Tests for make_strip_plan.py

These tests verify the strip generation planning algorithms without
needing a database connection.
"""

import pytest

from isometric_nyc.e2e_generation.make_strip_plan import (
  Edge,
  GenerationStep,
  Point,
  StepStatus,
  StripBounds,
  create_depth_1_plan,
  create_depth_2_plan,
  create_depth_3_plus_plan,
  create_strip_plan,
  find_generation_edge,
  get_exterior_neighbors,
  is_edge_fully_generated,
)

# =============================================================================
# Point Tests
# =============================================================================


class TestPoint:
  def test_str(self) -> None:
    p = Point(3, 5)
    assert str(p) == "(3,5)"

  def test_add(self) -> None:
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = p1 + p2
    assert result == Point(4, 6)

  def test_from_string_simple(self) -> None:
    p = Point.from_string("3,5")
    assert p == Point(3, 5)

  def test_from_string_with_parens(self) -> None:
    p = Point.from_string("(3,5)")
    assert p == Point(3, 5)

  def test_from_string_with_spaces(self) -> None:
    p = Point.from_string("( 3 , 5 )")
    assert p == Point(3, 5)

  def test_from_string_negative(self) -> None:
    p = Point.from_string("-3,-5")
    assert p == Point(-3, -5)

  def test_from_string_invalid(self) -> None:
    with pytest.raises(ValueError):
      Point.from_string("invalid")


# =============================================================================
# StripBounds Tests
# =============================================================================


class TestStripBounds:
  def test_width_height(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert bounds.width == 11
    assert bounds.height == 3

  def test_is_horizontal(self) -> None:
    # Horizontal strip (wider than tall)
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.is_horizontal is True

    # Vertical strip (taller than wide)
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.is_horizontal is False

    # Square (treated as horizontal)
    sq_bounds = StripBounds(Point(0, 0), Point(5, 5))
    assert sq_bounds.is_horizontal is True

  def test_depth(self) -> None:
    # Horizontal strip: depth = height
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.depth == 3

    # Vertical strip: depth = width
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.depth == 3

  def test_length(self) -> None:
    # Horizontal strip: length = width
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.length == 11

    # Vertical strip: length = height
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.length == 11

  def test_all_points(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 1))
    points = bounds.all_points()
    expected = [
      Point(0, 0),
      Point(1, 0),
      Point(2, 0),
      Point(0, 1),
      Point(1, 1),
      Point(2, 1),
    ]
    assert points == expected


# =============================================================================
# Edge Detection Tests
# =============================================================================


class TestGetExteriorNeighbors:
  def test_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.TOP)
    assert neighbors == [Point(0, -1), Point(1, -1), Point(2, -1)]

  def test_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.BOTTOM)
    assert neighbors == [Point(0, 1), Point(1, 1), Point(2, 1)]

  def test_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.LEFT)
    assert neighbors == [Point(-1, 0), Point(-1, 1), Point(-1, 2)]

  def test_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.RIGHT)
    assert neighbors == [Point(1, 0), Point(1, 1), Point(1, 2)]


class TestIsEdgeFullyGenerated:
  def test_fully_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is True

  def test_partially_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(2, -1)}  # Missing Point(1, -1)
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False

  def test_not_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated: set[Point] = set()
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False


class TestFindGenerationEdge:
  def test_horizontal_strip_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants below the strip
    generated = {Point(x, 1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.BOTTOM

  def test_horizontal_strip_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants above the strip
    generated = {Point(x, -1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.TOP

  def test_vertical_strip_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the left
    generated = {Point(-1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.LEFT

  def test_vertical_strip_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the right
    generated = {Point(1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.RIGHT

  def test_no_valid_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    generated: set[Point] = set()  # Nothing generated
    edge = find_generation_edge(bounds, generated)
    assert edge is None


# =============================================================================
# Depth 1 Plan Tests
# =============================================================================


class TestDepth1Plan:
  def test_horizontal_depth_1_simple(self) -> None:
    """Test a simple 11-wide depth-1 horizontal strip."""
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Should have 2x1 tiles first, then single fills
    # Pattern: SS.SS.SS.SS (positions 0-1, 3-4, 6-7, 9-10)
    # Then fills: positions 2, 5, 8

    # Extract all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 11 positions
    expected = {Point(x, 0) for x in range(11)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_1_pattern(self) -> None:
    """Verify the 2x1 + gap + fill pattern."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Phase 1: 2x1 tiles at positions 0-1, 3-4, 6-7
    # Phase 2: Single tiles at positions 2, 5

    # First step should be 2x1 at (0,0)
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

    # Second step should be 2x1 at (3,0)
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]

    # Third step should be 2x1 at (6,0)
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]

    # Fourth step should be single at (2,0)
    assert plan[3].quadrants == [Point(2, 0)]

    # Fifth step should be single at (5,0)
    assert plan[4].quadrants == [Point(5, 0)]

  def test_vertical_depth_1(self) -> None:
    """Test a vertical depth-1 strip."""
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    plan = create_depth_1_plan(bounds, Edge.LEFT)

    # Should have 2-quadrant tiles first, then singles
    # Pattern: SS.SS.SS.SS (y positions 0-1, 3-4, 6-7)
    # Then fills: positions 2, 5

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected


# =============================================================================
# Depth 2 Plan Tests
# =============================================================================


class TestDepth2Plan:
  def test_horizontal_depth_2_coverage(self) -> None:
    """Test that depth-2 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_2_order(self) -> None:
    """Test that depth-2 processes row closest to edge first."""
    bounds = StripBounds(Point(0, 0), Point(4, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    # With BOTTOM edge, should process y=1 first, then y=0
    # First steps should all be at y=1
    first_row_steps = []
    for step in plan:
      if all(q.y == 1 for q in step.quadrants):
        first_row_steps.append(step)
      elif len(first_row_steps) > 0:
        break  # Found a different row, stop

    assert len(first_row_steps) > 0
    assert all(q.y == 1 for step in first_row_steps for q in step.quadrants)


# =============================================================================
# Depth 3+ Plan Tests
# =============================================================================


class TestDepth3PlusPlan:
  def test_horizontal_depth_3_coverage(self) -> None:
    """Test that depth-3 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

  def test_depth_3_has_2x2_tiles(self) -> None:
    """Test that depth-3 plan uses 2x2 tiles."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Should have at least one 2x2 tile
    has_2x2 = any(len(step.quadrants) == 4 for step in plan)
    assert has_2x2, "Depth-3 plan should contain 2x2 tiles"

  def test_depth_4_coverage(self) -> None:
    """Test that depth-4 (3+1) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 3))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(4)}
    assert set(all_quadrants) == expected

  def test_depth_5_coverage(self) -> None:
    """Test that depth-5 (3+2) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 4))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(5)}
    assert set(all_quadrants) == expected

  def test_depth_6_coverage(self) -> None:
    """Test that depth-6 (3+3) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 5))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(6)}
    assert set(all_quadrants) == expected


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateStripPlan:
  def test_depth_1_uses_depth_1_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 1 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_2_uses_depth_2_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 1))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 2 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_3_uses_depth_3_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 3 should have some 4-quadrant tiles
    has_4_quad = any(len(step.quadrants) == 4 for step in plan)
    assert has_4_quad


# =============================================================================
# GenerationStep Serialization Tests
# =============================================================================


class TestGenerationStepSerialization:
  def test_to_dict(self) -> None:
    step = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0)],
      status=StepStatus.PENDING,
    )
    d = step.to_dict()
    assert d["quadrants"] == "(0,0),(1,0)"
    assert d["status"] == "pending"

  def test_from_dict(self) -> None:
    d = {
      "quadrants": "(0,0),(1,0)",
      "status": "done",
    }
    step = GenerationStep.from_dict(d)
    assert step.quadrants == [Point(0, 0), Point(1, 0)]
    assert step.status == StepStatus.DONE

  def test_roundtrip(self) -> None:
    original = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)],
      status=StepStatus.PENDING,
    )
    d = original.to_dict()
    restored = GenerationStep.from_dict(d)
    assert restored.quadrants == original.quadrants
    assert restored.status == original.status


# =============================================================================
# Seam Avoidance Tests
# =============================================================================


class TestSeamAvoidance:
  """Tests for seam avoidance when there's a generated quadrant at strip ends."""

  def test_depth_1_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip with generated quadrant to the left.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    # Generated quadrant at (-1, 0) - to the left of the strip
    generated = {Point(-1, 0)}
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # With left neighbor, should start at x=1 instead of x=0
    # So first 2x1 should be at (1,0),(2,0), not (0,0),(1,0)
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]

    # The gap at x=0 should be filled last
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should still cover all positions
    expected = {Point(x, 0) for x in range(8)}
    assert set(all_quadrants) == expected

    # x=0 should be generated as a single quadrant (last step)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    assert Point(0, 0) in [q for s in single_steps for q in s.quadrants]

  def test_depth_1_horizontal_no_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip without left neighbor.
    Should NOT offset start.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated: set[Point] = set()  # No neighbors
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # Without left neighbor, should start at x=0
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

  def test_depth_1_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-1 vertical strip with generated quadrant above.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    # Generated quadrant at (0, -1) - above the strip
    generated = {Point(0, -1)}
    plan = create_depth_1_plan(bounds, Edge.LEFT, generated)

    # With top neighbor, should start at y=1 instead of y=0
    assert plan[0].quadrants == [Point(0, 1), Point(0, 2)]

    # Should still cover all positions
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected

  def test_depth_2_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-2 horizontal strip with left neighbor.
    Both rows should be offset.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    # Generated quadrants to the left of both rows
    generated = {Point(-1, 0), Point(-1, 1)}
    plan = create_depth_2_plan(bounds, Edge.BOTTOM, generated)

    # First 2x1 in each row should be offset
    # (With BOTTOM edge, row y=1 is processed first)
    # First step should start at x=1, not x=0
    assert plan[0].quadrants[0].x == 1

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_depth_3_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-3 horizontal strip with left neighbor.
    All generation steps should respect the offset:
    - 2x2 tiles should start at x=1
    - 2x1 bridges in close row should start at x=1
    - x=0 should be filled last as single quadrants
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    # Generated quadrants to the left
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # First 2x2 should be offset to start at x=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.x >= 1 for q in first_4_quad.quadrants), "2x2 tiles should be offset"

    # All 2x2 tiles should not touch x=0
    all_4_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    for step in all_4_quad_steps:
      assert all(q.x >= 1 for q in step.quadrants), f"2x2 at x=0 would create seam: {step.quadrants}"

    # Find 2x1 horizontal tiles (2 quadrants in same row)
    two_quad_horizontal = [
      s for s in plan
      if len(s.quadrants) == 2 and s.quadrants[0].y == s.quadrants[1].y
    ]
    # First 2x1 horizontal should be offset (not starting at x=0)
    if two_quad_horizontal:
      first_2x1_h = two_quad_horizontal[0]
      assert first_2x1_h.quadrants[0].x >= 1, "First 2x1 horizontal should be offset"

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

    # x=0 should be generated as single quadrants (filled last)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    x0_singles = [s for s in single_steps if s.quadrants[0].x == 0]
    assert len(x0_singles) >= 1, "x=0 positions should be filled as singles"

  def test_depth_3_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-3 vertical strip with top neighbor.
    All generation steps should respect the offset.
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    # Generated quadrants above
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # First 2x2 should be offset to start at y=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.y >= 1 for q in first_4_quad.quadrants), "2x2 tiles should be offset"

    # All 2x2 tiles should not touch y=0
    all_4_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    for step in all_4_quad_steps:
      assert all(q.y >= 1 for q in step.quadrants), f"2x2 at y=0 would create seam: {step.quadrants}"

    # Find 1x2 vertical tiles (2 quadrants in same column)
    two_quad_vertical = [
      s for s in plan
      if len(s.quadrants) == 2 and s.quadrants[0].x == s.quadrants[1].x
    ]
    # First 1x2 vertical should be offset (not starting at y=0)
    if two_quad_vertical:
      first_1x2_v = two_quad_vertical[0]
      assert first_1x2_v.quadrants[0].y >= 1, "First 1x2 vertical should be offset"

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(3) for y in range(8)}
    assert set(all_quadrants) == expected

    # y=0 should be generated as single quadrants (filled last)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    y0_singles = [s for s in single_steps if s.quadrants[0].y == 0]
    assert len(y0_singles) >= 1, "y=0 positions should be filled as singles"

  def test_depth_3_all_2x1_bridges_offset_horizontal(self) -> None:
    """
    Verify that in depth-3 horizontal strip with left neighbor,
    the 2x1 bridges in the close row are also offset.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # With BOTTOM edge, close row is y=2 (closest to bottom)
    close_row_y = 2

    # Find 2x1 tiles in the close row
    close_row_2x1 = [
      s for s in plan
      if len(s.quadrants) == 2
      and all(q.y == close_row_y for q in s.quadrants)
      and s.quadrants[0].x + 1 == s.quadrants[1].x  # horizontal pair
    ]

    # First 2x1 in close row should start at x=1, not x=0
    if close_row_2x1:
      first_bridge = close_row_2x1[0]
      assert first_bridge.quadrants[0].x >= 1, (
        f"First 2x1 bridge in close row should be offset, "
        f"but starts at x={first_bridge.quadrants[0].x}"
      )

  def test_depth_3_all_1x2_bridges_offset_vertical(self) -> None:
    """
    Verify that in depth-3 vertical strip with top neighbor,
    the 1x2 bridges in the close column are also offset.
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # With LEFT edge, close column is x=2 (farthest from left)
    close_col_x = 2

    # Find 1x2 tiles in the close column
    close_col_1x2 = [
      s for s in plan
      if len(s.quadrants) == 2
      and all(q.x == close_col_x for q in s.quadrants)
      and s.quadrants[0].y + 1 == s.quadrants[1].y  # vertical pair
    ]

    # First 1x2 in close column should start at y=1, not y=0
    if close_col_1x2:
      first_bridge = close_col_1x2[0]
      assert first_bridge.quadrants[0].y >= 1, (
        f"First 1x2 bridge in close column should be offset, "
        f"but starts at y={first_bridge.quadrants[0].y}"
      )

  def test_depth_3_2x2_one_away_from_generation_edge_horizontal(self) -> None:
    """
    Verify that 2x2 tiles are placed in the rows FARTHEST from the generation
    edge, leaving a 1-row gap (the "close" row) between 2x2 and generated content.

    For BOTTOM edge with strip rows 0,1,2 and generated at row 3:
    - 2x2 tiles should be at rows 0,1 (farthest from edge)
    - Row 2 should be the bridge row (1 away from generated)
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should only be in rows 0 and 1 (NOT in row 2 which is close to edge)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.y in [0, 1], (
          f"2x2 tile at y={q.y} violates gap requirement - "
          f"should only be in rows 0,1 (farthest from generation edge)"
        )

  def test_depth_3_2x2_one_away_from_generation_edge_vertical(self) -> None:
    """
    Verify that 2x2 tiles are placed in the columns FARTHEST from the generation
    edge, leaving a 1-column gap between 2x2 and generated content.

    For LEFT edge with strip columns 0,1,2 and generated at column -1:
    - 2x2 tiles should be at columns 1,2 (farthest from edge)
    - Column 0 should be the bridge column (1 away from generated)
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should only be in columns 1 and 2 (NOT in column 0 which is close to edge)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.x in [1, 2], (
          f"2x2 tile at x={q.x} violates gap requirement - "
          f"should only be in columns 1,2 (farthest from generation edge)"
        )

  def test_depth_3_2x2_one_away_from_left_neighbor(self) -> None:
    """
    Verify that when there's a generated neighbor to the left,
    2x2 tiles maintain a 1-quadrant gap from that edge too.

    With generated at x=-1, strip from x=0 to x=7:
    - x=0 should be the gap column (filled later as 1x2 bridges)
    - 2x2 tiles should start at x=1 minimum
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should NOT be at x=0 (gap column next to left neighbor)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.x >= 1, (
          f"2x2 tile at x={q.x} violates gap requirement - "
          f"should be 1 away from left neighbor at x=-1"
        )

    # x=0 should be filled as 1x2 vertical bridges or singles, not 2x2
    x0_steps = [s for s in plan if any(q.x == 0 for q in s.quadrants)]
    for step in x0_steps:
      assert len(step.quadrants) <= 2, (
        f"x=0 should be filled as 1x2 or single, not {len(step.quadrants)}-quad tile"
      )

  def test_depth_3_2x2_one_away_from_top_neighbor(self) -> None:
    """
    Verify that when there's a generated neighbor above,
    2x2 tiles maintain a 1-quadrant gap from that edge too.

    With generated at y=-1, strip from y=0 to y=7:
    - y=0 should be the gap row (filled later as 2x1 bridges)
    - 2x2 tiles should start at y=1 minimum
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should NOT be at y=0 (gap row next to top neighbor)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.y >= 1, (
          f"2x2 tile at y={q.y} violates gap requirement - "
          f"should be 1 away from top neighbor at y=-1"
        )

    # y=0 should be filled as 2x1 horizontal bridges or singles, not 2x2
    y0_steps = [s for s in plan if any(q.y == 0 for q in s.quadrants)]
    for step in y0_steps:
      assert len(step.quadrants) <= 2, (
        f"y=0 should be filled as 2x1 or single, not {len(step.quadrants)}-quad tile"
      )

  def test_create_strip_plan_passes_generated(self) -> None:
    """Test that create_strip_plan passes generated set correctly."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated = {Point(-1, 0)}  # Left neighbor
    plan = create_strip_plan(bounds, Edge.BOTTOM, generated)

    # Should be offset due to left neighbor
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]


# =============================================================================
# Task Example Tests (from 019_strip_plan.md)
# =============================================================================


class TestTaskExamples:
  def test_depth_1_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(10,0), depth=1

    Expected pattern:
    Step 1: (0,0),(1,0)
    Step 2: (3,0),(4,0)
    Step 3: (6,0),(7,0)
    Step 4: (9,0),(10,0)
    Step 5: (2,0)
    Step 6: (5,0)
    Step 7: (8,0)
    """
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Verify first 4 steps are 2x1 tiles
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]
    assert plan[3].quadrants == [Point(9, 0), Point(10, 0)]

    # Verify remaining steps are single quadrants
    assert plan[4].quadrants == [Point(2, 0)]
    assert plan[5].quadrants == [Point(5, 0)]
    assert plan[6].quadrants == [Point(8, 0)]

    # Total should be 7 steps
    assert len(plan) == 7

  def test_depth_3_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(7,2), depth=3

    This tests the general pattern of:
    1. 2x2 tiles away from edge
    2. 1x2 bridges between 2x2s
    3. 2x1 bridges back to edge
    4. Single quadrant fills
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Collect all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 24 quadrants (8 x 3)
    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

    # First steps should be 2x2 tiles
    four_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    assert len(four_quad_steps) >= 2, "Should have at least 2 2x2 tiles"

```

```python
def _create_vertical_depth_3_plus_plan(  ...
  # Determine x positions based on generation edge
  if generation_edge == Edge.RIGHT:
    # Generated region is to the right, work from right-to-left    x_positions = list(range(bounds.bottom_right.x, bounds.top_left.x - 1, -1))  else:    # Generated region is to the left, work from left-to-right
    x_positions = list(range(bounds.top_left.x, bounds.bottom_right.x + 1))
```

I found the bug! The logic for determining which columns are "close" vs "far" is inverted for the vertical strip with LEFT edge. Let me fix this:

```python
x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]
x_far_2 = x_cols[1]
x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]
```

```python
# x_cols[0] = closest to edge
# x_cols[2] = farthest from edge this is wrong! Looking at how x_cols is created in `_create_vertical_depth_3_plus_plan`:
if generation_edge == Edge.RIGHT:
    # Generated region is to the right, work from right-to-left
    x_positions = list(range(bounds.bottom_right.x, bounds.top_left.x - 1, -1))
else:
    # Generated region is to the left, work from left-to-right
    x_positions = list(range(bounds.top_left.x, bounds.bottom_right.x + 1)) LEFT edge with bounds (0,0) to (2,7):
- x_positions = range(0, 3) = [0, 1, 2]

So x_cols = [0, 1, 2] where:- x_cols[0] = 0 = leftmost, which is CLOSEST to the LEFT edge (at x=-1)
- x_cols[2] = 2 = rightmost, which is FARTHEST from the LEFT edgeWait, but the comment says "x_cols should be ordered from closest to farthest from the generation edge". So for LEFT edge:
- x_cols[0] should be closest (x=0)
- x_cols[2] should be farthest (x=2)

That matches what x_positions gives us! So the comment is correct.


I'm investigating the x_far_1 assignment, which seems problematic. For the LEFT edge, x_far_1 = x_cols[0] = 0, but this is actually the closest point, not the farthest. The right-to-left x_positions suggests the indexing might be counterintuitive. I need to carefully trace how x_far_1 is determined based on the generation edge.

Checking the x_positions range confirms the indexing pattern. For the RIGHT edge, x_cols[0] is 2 (rightmost), while x_cols[2] is 0 (leftmost). Similarly, for the LEFT edge, x_cols[0] is 0 (leftmost), and x_cols[2] is 2 (rightmost). This means x_cols[0] consistently represents the point closest to the generation edge.

The original comment holds true: x_cols[0] is always closest to the generation edge, x_cols[2] always farthest.

The code selects far and close columns based on the generation edge, ensuring correct column selection depending on whether generation starts from the right or left side of the grid.

For RIGHT edge, the column selection works correctly:
- Farthest column is x_cols[2] = 0
- Middle column is x_cols[1] = 1
- Closest column is x_cols[0] = 2

For LEFT edge, the current implementation has an inverted logic problem. The fix requires swapping the column selections to match the right edge pattern, ensuring consistent column indexing across both edge scenarios.

The proposed solution simplifies column selection by always choosing x_cols[2] as the farthest column and x_cols[1] as the middle column, with x_close dynamically adjusted based on the generation edge.
```

The bug is in the logic for determining "far" vs "close" columns in `_generate_3_col_chunk_vertical`. The `x_cols` array is already ordered from closest to farthest from the generation edge, so:
- `x_cols[0]` is always closest to the edge
- `x_cols[2]` is always farthest from the edge

The current logic is backwards for the LEFT edge. Let me fix it:

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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
  """
  steps: list[GenerationStep] = []
  generated = generated or set()

  # y_rows[0] = closest to edge (will be done last)
  # y_rows[1] = middle row
  # y_rows[2] = farthest from edge (will be done first with 2x2)

  # For the pattern, we need y1 and y2 for the 2x2 tiles (farthest two rows)
  # and y3 for the bridge row (closest to edge)
  y_far_1 = y_rows[2] if generation_edge == Edge.BOTTOM else y_rows[0]
  y_far_2 = y_rows[1]
  y_close = y_rows[0] if generation_edge == Edge.BOTTOM else y_rows[2]

  # The 2x2 tiles should use the two rows farthest from edge
  # Ensure y_2x2_top < y_2x2_bottom
  y_2x2_top = min(y_far_1, y_far_2)
  y_2x2_bottom = max(y_far_1, y_far_2)

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Track covered x positions for each row type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles (gaps in 2x2 rows)
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
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Track covered y positions for each column type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles (gaps in 2x2 cols)
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
  y = y_start_offset
  while y + 1 <= y_end:
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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

Now let me also verify the horizontal version is correct:

```plaintext
 me trace through this for BOTTOM edge with bounds (0,0) to (7,2):- y_positions = [2, 1, 0] (from closest to farthest from bottom at y=3)- y_rows = [2, 1, 0]- y_far_1 = y_rows[2] = 0 (for BOTTOM edge)- y_far_2 = y_rows[1] = 1- y_close = y_rows[0] = 2 (for BOTTOM edge)Wait, that doesn't follow the same pattern. Let me think about this more carefully.For y_positions with BOTTOM edge:
- range(bounds.bottom_right.y, bounds.top_left.y - 1, -1)
- range(2, -1, -1) = [2, 1, 0]So y_rows = [2, 1, 0] where:
- y_rows[0] = 2 = bottom row, which is CLOSEST to the BOTTOM edge (at y=3)
- y_rows[2] = 0 = top row, which is FARTHEST from the BOTTOM edgeThe comment says y_rows is "ordered from closest to farthest from the generation edge". So:
- y_rows[0] = closest- y_rows[2] = farthestFor BOTTOM edge:- y_far_1 = y_rows[2] = 0 (farthest from bottom) ✓
- y_far_2 = y_rows[1] = 1 (middle) ✓- y_close = y_rows[0] = 2 (closest to bottom) ✓


I'm analyzing the TOP edge scenario, noting the different row arrangement and indexing compared to the BOTTOM edge.

The TOP edge generates rows from 0 to 2, with a distinct spatial relationship. The row indices shift, creating a new pattern of proximity to the generation edge. This requires careful mapping of row positions and understanding how the edge changes the row selection logic.

Specifically, the current approach incorrectly assigns the closest and farthest rows. I need to rethink how y_rows are mapped, ensuring the row indices accurately represent their distance from the TOP edge.

The horizontal version suggests a similar complexity in row positioning that requires precise indexing and edge-specific handling.

I'll modify the code to consistently define row positions: the first row always represents the closest to the edge, the middle row is intermediate, and the last row represents the farthest point.

The test for the bottom edge worked correctly, revealing this issue is specific to the top edge's implementation. I'll update the horizontal generation logic to match the vertical approach, ensuring uniform row selection across different edge orientations.
```

I see the horizontal version has a similar issue for the TOP edge. Let me fix it to use the same consistent pattern:

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Track covered x positions for each row type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles (gaps in 2x2 rows)
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
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Track covered y positions for each column type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles (gaps in 2x2 cols)
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
  y = y_start_offset
  while y + 1 <= y_end:
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
    help="Top-left corner of the strip (x,y)",
  )
  parser.add_argument(
    "--br",
    type=str,
    required=True,
    help="Bottom-right corner of the strip (x,y)",
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

```

All seam avoidance tests pass. Let me run the full test suite:

All **55 tests pass**. Let me format the code:

Done! I fixed a bug in the depth-3 algorithm where 2×2 tiles were being placed in the wrong rows/columns.

## Bug Fix Summary

The issue was in how `_generate_3_row_chunk_horizontal` and `_generate_3_col_chunk_vertical` determined which rows/columns were "close" vs "far" from the generation edge.

### The Problem

The original code had conditional logic that was inverted for certain edges:
```python
# OLD (buggy) - vertical
x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]  # Wrong for LEFT
x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]  # Wrong for LEFT
```

### The Fix

Since `y_rows` and `x_cols` are already ordered from closest to farthest from the generation edge, the logic simplifies to:
```python
# NEW (correct) - both horizontal and vertical
y_close = y_rows[0]   # Always closest to edge - becomes bridge row
y_far_1 = y_rows[1]   # Middle - part of 2x2
y_far_2 = y_rows[2]   # Farthest - part of 2x2
```

### Depth-3 Gap Requirements (Now Verified by Tests)

For depth-3 strips, 2×2 tiles are now correctly placed **1 quadrant away from ALL edges**:

1. **Generation edge**: 2×2 tiles use the 2 rows/columns FARTHEST from the generation edge, leaving the CLOSEST row/column as a gap
2. **Perpendicular edges**: If there's a generated neighbor at the start of the strip, 2×2 tiles are offset by 1, leaving the first row/column as a gap

### New Tests Added

- `test_depth_3_2x2_one_away_from_generation_edge_horizontal` - verifies 2×2 tiles don't touch the bridge row
- `test_depth_3_2x2_one_away_from_generation_edge_vertical` - same for vertical strips
- `test_depth_3_2x2_one_away_from_left_neighbor` - verifies gap when there's a left neighbor
- `test_depth_3_2x2_one_away_from_top_neighbor` - verifies gap when there's a top neighbor

```python
# OLD (buggy) - vertical
x_far_1 = x_cols[2] if generation_edge == Edge.RIGHT else x_cols[0]  # Wrong for LEFT
x_close = x_cols[0] if generation_edge == Edge.RIGHT else x_cols[2]  # Wrong for LEFT
```

```python
# NEW (correct) - both horizontal and vertical
y_close = y_rows[0]   # Always closest to edge - becomes bridge row
y_far_1 = y_rows[1]   # Middle - part of 2x2
y_far_2 = y_rows[2]   # Farthest - part of 2x2
```

---

**User**

When I run the command `@zsh (69-74) `, I get the following error: @zsh (69-74) 

---

**Cursor**

```plaintext
make_strip_plan.py: error: argument --tl: expected one argument
```

The error is because argparse interprets `-12,-1` as a flag (starting with `-`) rather than a value. This is a common issue with negative numbers in command-line arguments.

Let me fix the script to handle negative coordinates properly:

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0
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

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Track covered x positions for each row type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles (gaps in 2x2 rows)
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
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Track covered y positions for each column type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles (gaps in 2x2 cols)
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
  y = y_start_offset
  while y + 1 <= y_end:
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

```

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0

  # For negative coordinates, use = and quotes:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/v01 \\
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

  # Check if there's a generated quadrant to the left of the strip start
  # If so, offset by 1 to avoid a seam
  has_left_neighbor = any(Point(x_start - 1, y) in generated for y in y_rows)
  x_start_offset = x_start + 1 if has_left_neighbor else x_start

  # Track covered x positions for each row type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  # Pattern: SS.SS.SS... (each SS is a 2x2 tile)
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Step 2: Generate 1x2 vertical bridges between 2x2 tiles (gaps in 2x2 rows)
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
  x = x_start_offset
  while x + 1 <= x_end:
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

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Track covered y positions for each column type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles (gaps in 2x2 cols)
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
  y = y_start_offset
  while y + 1 <= y_end:
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

```

The issue is that argparse interprets `-12,-1` as a flag because it starts with `-`. For negative coordinates, you need to use `=` and quotes:

```bash
uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \
  generations/v01 \
  --tl='-12,-1' \
  --br='-10,7'
```

I've updated the script's help text and documentation to show this syntax. Let me verify it works:

```shellscript
uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \
  generations/v01 \
  --tl='-12,-1' \
  --br='-10,7'
```

---

**User**

OK - just encountered another bug - when creating a plan for a 3 deep horizontal generation, you didn't check the right edge - a 2x2 generation would touch already generated tiles. For all edges of the rectangle, any 2x1, 1x2 can only touch one previously generated edge, and a 2x2 can't touch any previously generated edge

---

**Cursor**

You're right! The depth-3 algorithm needs to check for generated neighbors on ALL edges of the strip, not just the left/top. Let me fix this:

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0

  # For negative coordinates, use = and quotes:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/v01 \\
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

  # Check if there's a generated quadrant above the strip start
  # If so, offset by 1 to avoid a seam
  has_top_neighbor = any(Point(x, y_start - 1) in generated for x in x_cols)
  y_start_offset = y_start + 1 if has_top_neighbor else y_start

  # Track covered y positions for each column type
  covered_2x2: set[int] = set()
  covered_close: set[int] = set()

  # Step 1: Generate 2x2 quadrants with gaps
  y = y_start_offset
  while y + 1 <= y_end:
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

  # Step 2: Generate 2x1 horizontal bridges between 2x2 tiles (gaps in 2x2 cols)
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
  y = y_start_offset
  while y + 1 <= y_end:
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

```

Now let me update the vertical version too:

```python
"""
Strip plan generation script.

Creates a JSON file with generation steps for generating a strip of quadrants.

The algorithm:
1. Find the "generation edge" - the edge where all exterior neighbors are generated
2. Determine direction of progress along the edge
3. Generate quadrants using an efficient pattern based on strip depth

Usage:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    <generation_dir> \\
    --tl <x>,<y> \\
    --br <x>,<y>

Example:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/test_generation \\
    --tl 0,0 \\
    --br 10,0

  # For negative coordinates, use = and quotes:
  uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \\
    generations/v01 \\
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

```

Now let me add tests for the right and bottom edge checks:

```python
"""
Tests for make_strip_plan.py

These tests verify the strip generation planning algorithms without
needing a database connection.
"""

import pytest

from isometric_nyc.e2e_generation.make_strip_plan import (
  Edge,
  GenerationStep,
  Point,
  StepStatus,
  StripBounds,
  create_depth_1_plan,
  create_depth_2_plan,
  create_depth_3_plus_plan,
  create_strip_plan,
  find_generation_edge,
  get_exterior_neighbors,
  is_edge_fully_generated,
)

# =============================================================================
# Point Tests
# =============================================================================


class TestPoint:
  def test_str(self) -> None:
    p = Point(3, 5)
    assert str(p) == "(3,5)"

  def test_add(self) -> None:
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = p1 + p2
    assert result == Point(4, 6)

  def test_from_string_simple(self) -> None:
    p = Point.from_string("3,5")
    assert p == Point(3, 5)

  def test_from_string_with_parens(self) -> None:
    p = Point.from_string("(3,5)")
    assert p == Point(3, 5)

  def test_from_string_with_spaces(self) -> None:
    p = Point.from_string("( 3 , 5 )")
    assert p == Point(3, 5)

  def test_from_string_negative(self) -> None:
    p = Point.from_string("-3,-5")
    assert p == Point(-3, -5)

  def test_from_string_invalid(self) -> None:
    with pytest.raises(ValueError):
      Point.from_string("invalid")


# =============================================================================
# StripBounds Tests
# =============================================================================


class TestStripBounds:
  def test_width_height(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert bounds.width == 11
    assert bounds.height == 3

  def test_is_horizontal(self) -> None:
    # Horizontal strip (wider than tall)
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.is_horizontal is True

    # Vertical strip (taller than wide)
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.is_horizontal is False

    # Square (treated as horizontal)
    sq_bounds = StripBounds(Point(0, 0), Point(5, 5))
    assert sq_bounds.is_horizontal is True

  def test_depth(self) -> None:
    # Horizontal strip: depth = height
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.depth == 3

    # Vertical strip: depth = width
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.depth == 3

  def test_length(self) -> None:
    # Horizontal strip: length = width
    h_bounds = StripBounds(Point(0, 0), Point(10, 2))
    assert h_bounds.length == 11

    # Vertical strip: length = height
    v_bounds = StripBounds(Point(0, 0), Point(2, 10))
    assert v_bounds.length == 11

  def test_all_points(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 1))
    points = bounds.all_points()
    expected = [
      Point(0, 0),
      Point(1, 0),
      Point(2, 0),
      Point(0, 1),
      Point(1, 1),
      Point(2, 1),
    ]
    assert points == expected


# =============================================================================
# Edge Detection Tests
# =============================================================================


class TestGetExteriorNeighbors:
  def test_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.TOP)
    assert neighbors == [Point(0, -1), Point(1, -1), Point(2, -1)]

  def test_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    neighbors = get_exterior_neighbors(bounds, Edge.BOTTOM)
    assert neighbors == [Point(0, 1), Point(1, 1), Point(2, 1)]

  def test_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.LEFT)
    assert neighbors == [Point(-1, 0), Point(-1, 1), Point(-1, 2)]

  def test_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 2))
    neighbors = get_exterior_neighbors(bounds, Edge.RIGHT)
    assert neighbors == [Point(1, 0), Point(1, 1), Point(1, 2)]


class TestIsEdgeFullyGenerated:
  def test_fully_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is True

  def test_partially_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated = {Point(0, -1), Point(2, -1)}  # Missing Point(1, -1)
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False

  def test_not_generated_top(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(2, 0))
    generated: set[Point] = set()
    assert is_edge_fully_generated(bounds, Edge.TOP, generated) is False


class TestFindGenerationEdge:
  def test_horizontal_strip_bottom_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants below the strip
    generated = {Point(x, 1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.BOTTOM

  def test_horizontal_strip_top_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    # Generate all quadrants above the strip
    generated = {Point(x, -1) for x in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.TOP

  def test_vertical_strip_left_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the left
    generated = {Point(-1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.LEFT

  def test_vertical_strip_right_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(0, 10))
    # Generate all quadrants to the right
    generated = {Point(1, y) for y in range(11)}
    edge = find_generation_edge(bounds, generated)
    assert edge == Edge.RIGHT

  def test_no_valid_edge(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    generated: set[Point] = set()  # Nothing generated
    edge = find_generation_edge(bounds, generated)
    assert edge is None


# =============================================================================
# Depth 1 Plan Tests
# =============================================================================


class TestDepth1Plan:
  def test_horizontal_depth_1_simple(self) -> None:
    """Test a simple 11-wide depth-1 horizontal strip."""
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Should have 2x1 tiles first, then single fills
    # Pattern: SS.SS.SS.SS (positions 0-1, 3-4, 6-7, 9-10)
    # Then fills: positions 2, 5, 8

    # Extract all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 11 positions
    expected = {Point(x, 0) for x in range(11)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_1_pattern(self) -> None:
    """Verify the 2x1 + gap + fill pattern."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Phase 1: 2x1 tiles at positions 0-1, 3-4, 6-7
    # Phase 2: Single tiles at positions 2, 5

    # First step should be 2x1 at (0,0)
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

    # Second step should be 2x1 at (3,0)
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]

    # Third step should be 2x1 at (6,0)
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]

    # Fourth step should be single at (2,0)
    assert plan[3].quadrants == [Point(2, 0)]

    # Fifth step should be single at (5,0)
    assert plan[4].quadrants == [Point(5, 0)]

  def test_vertical_depth_1(self) -> None:
    """Test a vertical depth-1 strip."""
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    plan = create_depth_1_plan(bounds, Edge.LEFT)

    # Should have 2-quadrant tiles first, then singles
    # Pattern: SS.SS.SS.SS (y positions 0-1, 3-4, 6-7)
    # Then fills: positions 2, 5

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected


# =============================================================================
# Depth 2 Plan Tests
# =============================================================================


class TestDepth2Plan:
  def test_horizontal_depth_2_coverage(self) -> None:
    """Test that depth-2 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_horizontal_depth_2_order(self) -> None:
    """Test that depth-2 processes row closest to edge first."""
    bounds = StripBounds(Point(0, 0), Point(4, 1))
    plan = create_depth_2_plan(bounds, Edge.BOTTOM)

    # With BOTTOM edge, should process y=1 first, then y=0
    # First steps should all be at y=1
    first_row_steps = []
    for step in plan:
      if all(q.y == 1 for q in step.quadrants):
        first_row_steps.append(step)
      elif len(first_row_steps) > 0:
        break  # Found a different row, stop

    assert len(first_row_steps) > 0
    assert all(q.y == 1 for step in first_row_steps for q in step.quadrants)


# =============================================================================
# Depth 3+ Plan Tests
# =============================================================================


class TestDepth3PlusPlan:
  def test_horizontal_depth_3_coverage(self) -> None:
    """Test that depth-3 plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

  def test_depth_3_has_2x2_tiles(self) -> None:
    """Test that depth-3 plan uses 2x2 tiles."""
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Should have at least one 2x2 tile
    has_2x2 = any(len(step.quadrants) == 4 for step in plan)
    assert has_2x2, "Depth-3 plan should contain 2x2 tiles"

  def test_depth_4_coverage(self) -> None:
    """Test that depth-4 (3+1) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 3))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(4)}
    assert set(all_quadrants) == expected

  def test_depth_5_coverage(self) -> None:
    """Test that depth-5 (3+2) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 4))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(5)}
    assert set(all_quadrants) == expected

  def test_depth_6_coverage(self) -> None:
    """Test that depth-6 (3+3) plan covers all quadrants."""
    bounds = StripBounds(Point(0, 0), Point(7, 5))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(6)}
    assert set(all_quadrants) == expected


# =============================================================================
# Integration Tests
# =============================================================================


class TestCreateStripPlan:
  def test_depth_1_uses_depth_1_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 1 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_2_uses_depth_2_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 1))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 2 should not have any 4-quadrant tiles
    assert all(len(step.quadrants) <= 2 for step in plan)

  def test_depth_3_uses_depth_3_plan(self) -> None:
    bounds = StripBounds(Point(0, 0), Point(10, 2))
    plan = create_strip_plan(bounds, Edge.BOTTOM)

    # Depth 3 should have some 4-quadrant tiles
    has_4_quad = any(len(step.quadrants) == 4 for step in plan)
    assert has_4_quad


# =============================================================================
# GenerationStep Serialization Tests
# =============================================================================


class TestGenerationStepSerialization:
  def test_to_dict(self) -> None:
    step = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0)],
      status=StepStatus.PENDING,
    )
    d = step.to_dict()
    assert d["quadrants"] == "(0,0),(1,0)"
    assert d["status"] == "pending"

  def test_from_dict(self) -> None:
    d = {
      "quadrants": "(0,0),(1,0)",
      "status": "done",
    }
    step = GenerationStep.from_dict(d)
    assert step.quadrants == [Point(0, 0), Point(1, 0)]
    assert step.status == StepStatus.DONE

  def test_roundtrip(self) -> None:
    original = GenerationStep(
      quadrants=[Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)],
      status=StepStatus.PENDING,
    )
    d = original.to_dict()
    restored = GenerationStep.from_dict(d)
    assert restored.quadrants == original.quadrants
    assert restored.status == original.status


# =============================================================================
# Seam Avoidance Tests
# =============================================================================


class TestSeamAvoidance:
  """Tests for seam avoidance when there's a generated quadrant at strip ends."""

  def test_depth_1_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip with generated quadrant to the left.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    # Generated quadrant at (-1, 0) - to the left of the strip
    generated = {Point(-1, 0)}
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # With left neighbor, should start at x=1 instead of x=0
    # So first 2x1 should be at (1,0),(2,0), not (0,0),(1,0)
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]

    # The gap at x=0 should be filled last
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should still cover all positions
    expected = {Point(x, 0) for x in range(8)}
    assert set(all_quadrants) == expected

    # x=0 should be generated as a single quadrant (last step)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    assert Point(0, 0) in [q for s in single_steps for q in s.quadrants]

  def test_depth_1_horizontal_no_left_neighbor(self) -> None:
    """
    Test depth-1 horizontal strip without left neighbor.
    Should NOT offset start.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated: set[Point] = set()  # No neighbors
    plan = create_depth_1_plan(bounds, Edge.BOTTOM, generated)

    # Without left neighbor, should start at x=0
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]

  def test_depth_1_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-1 vertical strip with generated quadrant above.
    Should offset start by 1 to avoid seam.
    """
    bounds = StripBounds(Point(0, 0), Point(0, 7))
    # Generated quadrant at (0, -1) - above the strip
    generated = {Point(0, -1)}
    plan = create_depth_1_plan(bounds, Edge.LEFT, generated)

    # With top neighbor, should start at y=1 instead of y=0
    assert plan[0].quadrants == [Point(0, 1), Point(0, 2)]

    # Should still cover all positions
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(0, y) for y in range(8)}
    assert set(all_quadrants) == expected

  def test_depth_2_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-2 horizontal strip with left neighbor.
    Both rows should be offset.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 1))
    # Generated quadrants to the left of both rows
    generated = {Point(-1, 0), Point(-1, 1)}
    plan = create_depth_2_plan(bounds, Edge.BOTTOM, generated)

    # First 2x1 in each row should be offset
    # (With BOTTOM edge, row y=1 is processed first)
    # First step should start at x=1, not x=0
    assert plan[0].quadrants[0].x == 1

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(2)}
    assert set(all_quadrants) == expected

  def test_depth_3_horizontal_with_left_neighbor(self) -> None:
    """
    Test depth-3 horizontal strip with left neighbor.
    All generation steps should respect the offset:
    - 2x2 tiles should start at x=1
    - 2x1 bridges in close row should start at x=1
    - x=0 should be filled last as single quadrants
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    # Generated quadrants to the left
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # First 2x2 should be offset to start at x=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.x >= 1 for q in first_4_quad.quadrants), "2x2 tiles should be offset"

    # All 2x2 tiles should not touch x=0
    all_4_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    for step in all_4_quad_steps:
      assert all(q.x >= 1 for q in step.quadrants), (
        f"2x2 at x=0 would create seam: {step.quadrants}"
      )

    # Find 2x1 horizontal tiles (2 quadrants in same row)
    two_quad_horizontal = [
      s for s in plan if len(s.quadrants) == 2 and s.quadrants[0].y == s.quadrants[1].y
    ]
    # First 2x1 horizontal should be offset (not starting at x=0)
    if two_quad_horizontal:
      first_2x1_h = two_quad_horizontal[0]
      assert first_2x1_h.quadrants[0].x >= 1, "First 2x1 horizontal should be offset"

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

    # x=0 should be generated as single quadrants (filled last)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    x0_singles = [s for s in single_steps if s.quadrants[0].x == 0]
    assert len(x0_singles) >= 1, "x=0 positions should be filled as singles"

  def test_depth_3_vertical_with_top_neighbor(self) -> None:
    """
    Test depth-3 vertical strip with top neighbor.
    All generation steps should respect the offset.
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    # Generated quadrants above
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # First 2x2 should be offset to start at y=1
    first_4_quad = [s for s in plan if len(s.quadrants) == 4][0]
    assert all(q.y >= 1 for q in first_4_quad.quadrants), "2x2 tiles should be offset"

    # All 2x2 tiles should not touch y=0
    all_4_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    for step in all_4_quad_steps:
      assert all(q.y >= 1 for q in step.quadrants), (
        f"2x2 at y=0 would create seam: {step.quadrants}"
      )

    # Find 1x2 vertical tiles (2 quadrants in same column)
    two_quad_vertical = [
      s for s in plan if len(s.quadrants) == 2 and s.quadrants[0].x == s.quadrants[1].x
    ]
    # First 1x2 vertical should be offset (not starting at y=0)
    if two_quad_vertical:
      first_1x2_v = two_quad_vertical[0]
      assert first_1x2_v.quadrants[0].y >= 1, "First 1x2 vertical should be offset"

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(3) for y in range(8)}
    assert set(all_quadrants) == expected

    # y=0 should be generated as single quadrants (filled last)
    single_steps = [s for s in plan if len(s.quadrants) == 1]
    y0_singles = [s for s in single_steps if s.quadrants[0].y == 0]
    assert len(y0_singles) >= 1, "y=0 positions should be filled as singles"

  def test_depth_3_all_2x1_bridges_offset_horizontal(self) -> None:
    """
    Verify that in depth-3 horizontal strip with left neighbor,
    the 2x1 bridges in the close row are also offset.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # With BOTTOM edge, close row is y=2 (closest to bottom)
    close_row_y = 2

    # Find 2x1 tiles in the close row
    close_row_2x1 = [
      s
      for s in plan
      if len(s.quadrants) == 2
      and all(q.y == close_row_y for q in s.quadrants)
      and s.quadrants[0].x + 1 == s.quadrants[1].x  # horizontal pair
    ]

    # First 2x1 in close row should start at x=1, not x=0
    if close_row_2x1:
      first_bridge = close_row_2x1[0]
      assert first_bridge.quadrants[0].x >= 1, (
        f"First 2x1 bridge in close row should be offset, "
        f"but starts at x={first_bridge.quadrants[0].x}"
      )

  def test_depth_3_all_1x2_bridges_offset_vertical(self) -> None:
    """
    Verify that in depth-3 vertical strip with top neighbor,
    the 1x2 bridges in the close column are also offset.
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # With LEFT edge, close column is x=2 (farthest from left)
    close_col_x = 2

    # Find 1x2 tiles in the close column
    close_col_1x2 = [
      s
      for s in plan
      if len(s.quadrants) == 2
      and all(q.x == close_col_x for q in s.quadrants)
      and s.quadrants[0].y + 1 == s.quadrants[1].y  # vertical pair
    ]

    # First 1x2 in close column should start at y=1, not y=0
    if close_col_1x2:
      first_bridge = close_col_1x2[0]
      assert first_bridge.quadrants[0].y >= 1, (
        f"First 1x2 bridge in close column should be offset, "
        f"but starts at y={first_bridge.quadrants[0].y}"
      )

  def test_depth_3_2x2_one_away_from_generation_edge_horizontal(self) -> None:
    """
    Verify that 2x2 tiles are placed in the rows FARTHEST from the generation
    edge, leaving a 1-row gap (the "close" row) between 2x2 and generated content.

    For BOTTOM edge with strip rows 0,1,2 and generated at row 3:
    - 2x2 tiles should be at rows 0,1 (farthest from edge)
    - Row 2 should be the bridge row (1 away from generated)
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should only be in rows 0 and 1 (NOT in row 2 which is close to edge)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.y in [0, 1], (
          f"2x2 tile at y={q.y} violates gap requirement - "
          f"should only be in rows 0,1 (farthest from generation edge)"
        )

  def test_depth_3_2x2_one_away_from_generation_edge_vertical(self) -> None:
    """
    Verify that 2x2 tiles are placed in the columns FARTHEST from the generation
    edge, leaving a 1-column gap between 2x2 and generated content.

    For LEFT edge with strip columns 0,1,2 and generated at column -1:
    - 2x2 tiles should be at columns 1,2 (farthest from edge)
    - Column 0 should be the bridge column (1 away from generated)
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should only be in columns 1 and 2 (NOT in column 0 which is close to edge)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.x in [1, 2], (
          f"2x2 tile at x={q.x} violates gap requirement - "
          f"should only be in columns 1,2 (farthest from generation edge)"
        )

  def test_depth_3_2x2_one_away_from_left_neighbor(self) -> None:
    """
    Verify that when there's a generated neighbor to the left,
    2x2 tiles maintain a 1-quadrant gap from that edge too.

    With generated at x=-1, strip from x=0 to x=7:
    - x=0 should be the gap column (filled later as 1x2 bridges)
    - 2x2 tiles should start at x=1 minimum
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    generated = {Point(-1, 0), Point(-1, 1), Point(-1, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should NOT be at x=0 (gap column next to left neighbor)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.x >= 1, (
          f"2x2 tile at x={q.x} violates gap requirement - "
          f"should be 1 away from left neighbor at x=-1"
        )

    # x=0 should be filled as 1x2 vertical bridges or singles, not 2x2
    x0_steps = [s for s in plan if any(q.x == 0 for q in s.quadrants)]
    for step in x0_steps:
      assert len(step.quadrants) <= 2, (
        f"x=0 should be filled as 1x2 or single, not {len(step.quadrants)}-quad tile"
      )

  def test_depth_3_2x2_one_away_from_top_neighbor(self) -> None:
    """
    Verify that when there's a generated neighbor above,
    2x2 tiles maintain a 1-quadrant gap from that edge too.

    With generated at y=-1, strip from y=0 to y=7:
    - y=0 should be the gap row (filled later as 2x1 bridges)
    - 2x2 tiles should start at y=1 minimum
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should NOT be at y=0 (gap row next to top neighbor)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.y >= 1, (
          f"2x2 tile at y={q.y} violates gap requirement - "
          f"should be 1 away from top neighbor at y=-1"
        )

    # y=0 should be filled as 2x1 horizontal bridges or singles, not 2x2
    y0_steps = [s for s in plan if any(q.y == 0 for q in s.quadrants)]
    for step in y0_steps:
      assert len(step.quadrants) <= 2, (
        f"y=0 should be filled as 2x1 or single, not {len(step.quadrants)}-quad tile"
      )

  def test_depth_3_2x2_one_away_from_right_neighbor(self) -> None:
    """
    Verify that when there's a generated neighbor to the right,
    2x2 tiles maintain a 1-quadrant gap from that edge too.

    With generated at x=8, strip from x=0 to x=7:
    - x=7 should be the gap column (filled later as 1x2 bridges)
    - 2x2 tiles should end at x=6 maximum
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    generated = {Point(8, 0), Point(8, 1), Point(8, 2)}
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should NOT be at x=7 (gap column next to right neighbor)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.x <= 6, (
          f"2x2 tile at x={q.x} violates gap requirement - "
          f"should be 1 away from right neighbor at x=8"
        )

    # x=7 should be filled as 1x2 vertical bridges or singles, not 2x2
    x7_steps = [s for s in plan if any(q.x == 7 for q in s.quadrants)]
    for step in x7_steps:
      assert len(step.quadrants) <= 2, (
        f"x=7 should be filled as 1x2 or single, not {len(step.quadrants)}-quad tile"
      )

  def test_depth_3_2x2_one_away_from_bottom_neighbor(self) -> None:
    """
    Verify that when there's a generated neighbor below,
    2x2 tiles maintain a 1-quadrant gap from that edge too.

    With generated at y=8, strip from y=0 to y=7:
    - y=7 should be the gap row (filled later as 2x1 bridges)
    - 2x2 tiles should end at y=6 maximum
    """
    bounds = StripBounds(Point(0, 0), Point(2, 7))
    generated = {Point(0, 8), Point(1, 8), Point(2, 8)}
    plan = create_depth_3_plus_plan(bounds, Edge.LEFT, generated)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should NOT be at y=7 (gap row next to bottom neighbor)
    for step in all_2x2:
      for q in step.quadrants:
        assert q.y <= 6, (
          f"2x2 tile at y={q.y} violates gap requirement - "
          f"should be 1 away from bottom neighbor at y=8"
        )

    # y=7 should be filled as 2x1 horizontal bridges or singles, not 2x2
    y7_steps = [s for s in plan if any(q.y == 7 for q in s.quadrants)]
    for step in y7_steps:
      assert len(step.quadrants) <= 2, (
        f"y=7 should be filled as 2x1 or single, not {len(step.quadrants)}-quad tile"
      )

  def test_depth_3_2x2_one_away_from_both_left_and_right(self) -> None:
    """
    Verify that when there are generated neighbors on BOTH left and right,
    2x2 tiles maintain a 1-quadrant gap from BOTH edges.
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    # Neighbors on both left (x=-1) and right (x=8)
    generated = {
      Point(-1, 0), Point(-1, 1), Point(-1, 2),
      Point(8, 0), Point(8, 1), Point(8, 2),
    }
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM, generated)

    # Get all 2x2 tiles
    all_2x2 = [s for s in plan if len(s.quadrants) == 4]

    # 2x2 tiles should NOT be at x=0 or x=7
    for step in all_2x2:
      for q in step.quadrants:
        assert 1 <= q.x <= 6, (
          f"2x2 tile at x={q.x} violates gap requirement - "
          f"should be 1 away from both left (x=-1) and right (x=8) neighbors"
        )

    # Coverage should still be complete
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

  def test_create_strip_plan_passes_generated(self) -> None:
    """Test that create_strip_plan passes generated set correctly."""
    bounds = StripBounds(Point(0, 0), Point(7, 0))
    generated = {Point(-1, 0)}  # Left neighbor
    plan = create_strip_plan(bounds, Edge.BOTTOM, generated)

    # Should be offset due to left neighbor
    assert plan[0].quadrants == [Point(1, 0), Point(2, 0)]


# =============================================================================
# Task Example Tests (from 019_strip_plan.md)
# =============================================================================


class TestTaskExamples:
  def test_depth_1_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(10,0), depth=1

    Expected pattern:
    Step 1: (0,0),(1,0)
    Step 2: (3,0),(4,0)
    Step 3: (6,0),(7,0)
    Step 4: (9,0),(10,0)
    Step 5: (2,0)
    Step 6: (5,0)
    Step 7: (8,0)
    """
    bounds = StripBounds(Point(0, 0), Point(10, 0))
    plan = create_depth_1_plan(bounds, Edge.BOTTOM)

    # Verify first 4 steps are 2x1 tiles
    assert plan[0].quadrants == [Point(0, 0), Point(1, 0)]
    assert plan[1].quadrants == [Point(3, 0), Point(4, 0)]
    assert plan[2].quadrants == [Point(6, 0), Point(7, 0)]
    assert plan[3].quadrants == [Point(9, 0), Point(10, 0)]

    # Verify remaining steps are single quadrants
    assert plan[4].quadrants == [Point(2, 0)]
    assert plan[5].quadrants == [Point(5, 0)]
    assert plan[6].quadrants == [Point(8, 0)]

    # Total should be 7 steps
    assert len(plan) == 7

  def test_depth_3_example_from_task(self) -> None:
    """
    Test the example from the task file:
    tl=(0,0) br=(7,2), depth=3

    This tests the general pattern of:
    1. 2x2 tiles away from edge
    2. 1x2 bridges between 2x2s
    3. 2x1 bridges back to edge
    4. Single quadrant fills
    """
    bounds = StripBounds(Point(0, 0), Point(7, 2))
    plan = create_depth_3_plus_plan(bounds, Edge.BOTTOM)

    # Collect all quadrants
    all_quadrants: list[Point] = []
    for step in plan:
      all_quadrants.extend(step.quadrants)

    # Should cover all 24 quadrants (8 x 3)
    expected = {Point(x, y) for x in range(8) for y in range(3)}
    assert set(all_quadrants) == expected

    # First steps should be 2x2 tiles
    four_quad_steps = [s for s in plan if len(s.quadrants) == 4]
    assert len(four_quad_steps) >= 2, "Should have at least 2 2x2 tiles"

```

All seam avoidance tests pass. Let me run the full test suite:

All **58 tests pass**. Let me format the code:

Done! I've fixed the depth-3 algorithm to check for generated neighbors on **ALL edges** of the strip.

## Summary of Fixes

### Bug Fixed
The depth-3 algorithm was only checking for neighbors on the **left** edge (for horizontal strips) and **top** edge (for vertical strips). It wasn't checking for neighbors on the **right** or **bottom** edges, which could cause 2×2 tiles to touch already-generated content.

### Changes Made

**For horizontal depth-3 strips (`_generate_3_row_chunk_horizontal`):**
- Now checks for both `has_left_neighbor` AND `has_right_neighbor`
- 2×2 tiles are constrained to `[x_2x2_start, x_2x2_end]` where:
  - `x_2x2_start = x_start + 1` if left neighbor exists
  - `x_2x2_end = x_end - 1` if right neighbor exists

**For vertical depth-3 strips (`_generate_3_col_chunk_vertical`):**
- Now checks for both `has_top_neighbor` AND `has_bottom_neighbor`
- 2×2 tiles are constrained to `[y_2x2_start, y_2x2_end]` where:
  - `y_2x2_start = y_start + 1` if top neighbor exists
  - `y_2x2_end = y_end - 1` if bottom neighbor exists

### Rules Now Enforced

1. **2×2 tiles**: Cannot touch ANY previously generated edge (1-quadrant gap from all sides)
2. **2×1 / 1×2 tiles**: Can only touch ONE previously generated edge
3. **Single quadrants**: Used to fill gaps at edges

### New Tests Added
- `test_depth_3_2x2_one_away_from_right_neighbor`
- `test_depth_3_2x2_one_away_from_bottom_neighbor`
- `test_depth_3_2x2_one_away_from_both_left_and_right`

All **58 tests pass**.

### Command Line Usage
For negative coordinates, use `=` and quotes:
```bash
uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \
  generations/v01 \
  --tl='-12,-1' \
  --br='-10,7'
```

```shellscript
uv run python src/isometric_nyc/e2e_generation/make_strip_plan.py \
  generations/v01 \
  --tl='-12,-1' \
  --br='-10,7'
```

