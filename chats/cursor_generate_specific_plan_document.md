# Generate specific plan document
_Exported on 12/10/2025 at 13:55:07 EST from Cursor (2.1.39)_

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

