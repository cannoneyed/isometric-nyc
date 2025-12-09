# Implement automatic generation script
_Exported on 12/9/2025 at 08:47:24 EST from Cursor (2.1.39)_

---

**User**

Implement the logic / script described in @tasks/015_automatic_generation.md 

---

**Cursor**

I'll implement the automated generation script described in the task. Let me first explore the existing codebase to understand the data structures and generation logic.

Now let me look at how the database can be queried for all generated quadrants:

```sql
SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL
```

Now I have all the information I need. Let me create the automated generation script:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.generate_tile_omni import generate_tile
        from isometric_nyc.e2e_generation.shared import WEB_DIR, start_web_server

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            for step in steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # For each step, we need to generate the tile
                # The tile position is determined by the quadrants
                quadrants = step.quadrants

                if len(quadrants) == 4:
                    # 2x2 tile - find top-left
                    min_x = min(q.x for q in quadrants)
                    min_y = min(q.y for q in quadrants)
                    tile_x, tile_y = min_x, min_y
                elif len(quadrants) == 2:
                    # 1x2 or 2x1 tile
                    # The generation system works on 2x2 tiles, so we need
                    # to handle this specially
                    min_x = min(q.x for q in quadrants)
                    min_y = min(q.y for q in quadrants)
                    tile_x, tile_y = min_x, min_y
                else:
                    # Single quadrant
                    tile_x, tile_y = quadrants[0].x, quadrants[0].y

                try:
                    result = generate_tile(
                        generation_dir,
                        tile_x,
                        tile_y,
                        port=args.port,
                        overwrite=False,
                    )
                    if result:
                        print(f"✅ Step {step.step_number} complete")
                    else:
                        print(f"⏭️  Step {step.step_number} skipped (already generated)")
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    raise

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())

```

The script has been created with no linting errors. Let me test the dry-run mode to make sure it works correctly:

The shell is parsing the commas. Let me quote the arguments:

Excellent! The script is working! Let me see the rest of the output:

The dry run mode is working well. Let me also test with a larger bounding box to see the spiral expansion in action:

The script is working well. However, I noticed that the generation integration with `generate_tile_omni.py` might not work perfectly for all tile sizes since that script expects 2x2 tiles. Let me review the task description again and look at how the actual generation works for smaller tiles:

I see. The `view_generations.py` has the more flexible generation logic using `TemplateBuilder` and `InfillRegion`. The `generate_tile_omni.py` is a more specific script for 2x2 tiles. Let me update the automatic_generation.py script to use the more flexible approach from `view_generations.py`:

Now let me look at the imports and helper functions I'll need to reuse:

Now I have a complete understanding of the codebase. The script works, but I need to update the execution part to use the more flexible template-based generation. Let me update the automatic_generation.py to better integrate with the existing generation pipeline:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            for step in steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

Now I need to add the `run_generation_step` function that handles the flexible template-based generation:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Step Execution
# =============================================================================


def run_generation_step(
    conn: sqlite3.Connection,
    config: dict,
    quadrant_tuples: list[tuple[int, int]],
    generation_dir: Path,
    port: int,
) -> dict:
    """
    Run a single generation step for the given quadrants.

    This uses the flexible TemplateBuilder approach that can handle
    1, 2, or 4 quadrant selections.

    Returns dict with success status and message/error.
    """
    import os
    import tempfile

    import requests
    from dotenv import load_dotenv
    from PIL import Image

    from isometric_nyc.e2e_generation.infill_template import (
        QUADRANT_SIZE,
        InfillRegion,
        TemplateBuilder,
        validate_quadrant_selection,
    )
    from isometric_nyc.e2e_generation.shared import (
        get_quadrant_generation as shared_get_quadrant_generation,
        get_quadrant_render as shared_get_quadrant_render,
        image_to_png_bytes,
        png_bytes_to_image,
        save_quadrant_generation,
        upload_to_gcs,
    )

    load_dotenv()

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
        }

    # Helper functions
    def has_generation_in_db(qx: int, qy: int) -> bool:
        gen = shared_get_quadrant_generation(conn, qx, qy)
        return gen is not None

    def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
        """Get render, rendering if it doesn't exist yet."""
        render_bytes = shared_get_quadrant_render(conn, qx, qy)
        if render_bytes:
            return png_bytes_to_image(render_bytes)

        # Need to render
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)
        if render_bytes:
            return png_bytes_to_image(render_bytes)
        return None

    def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
        gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
            return png_bytes_to_image(gen_bytes)
        return None

    # Validate selection with auto-expansion
    is_valid, msg, placement = validate_quadrant_selection(
        quadrant_tuples, has_generation_in_db, allow_expansion=True
    )

    if not is_valid:
        return {"success": False, "error": msg}

    print(f"   ✅ Validation: {msg}")

    # Get primary quadrants (the ones we selected, not padding)
    primary_quadrants = (
        placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
    )
    padding_quadrants = placement.padding_quadrants if placement else []

    if padding_quadrants:
        print(f"   📦 Padding quadrants: {padding_quadrants}")

    # Create the infill region (may be expanded)
    if placement._expanded_region is not None:
        region = placement._expanded_region
    else:
        region = InfillRegion.from_quadrants(quadrant_tuples)

    # Build the template
    print("   🎨 Building template image...")
    builder = TemplateBuilder(
        region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
    )

    result = builder.build(border_width=2, allow_expansion=True)

    if result is None:
        error_msg = builder._last_validation_error or "Failed to build template"
        return {"success": False, "error": error_msg}

    template_image, placement = result

    # Save template to temp file and upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

    try:
        print("   📤 Uploading template to GCS...")
        image_url = upload_to_gcs(template_path, "isometric-nyc-infills")

        print("   🤖 Calling Oxen API...")
        generated_url = call_oxen_api(image_url, api_key)

        print("   📥 Downloading generated image...")
        generated_image = download_image_to_pil(generated_url)

        # Extract quadrants from generated image and save to database
        print("   💾 Saving generated quadrants to database...")

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
                    print(f"      ✓ Saved generation for ({qx}, {qy})")
                    saved_count += 1
                else:
                    print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
            else:
                print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

        return {
            "success": True,
            "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
            "quadrants": primary_quadrants,
        }

    finally:
        # Clean up temp file
        template_path.unlink(missing_ok=True)


def render_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    qx: int,
    qy: int,
    port: int,
) -> bytes | None:
    """Render a single quadrant using the web server."""
    from urllib.parse import urlencode

    from PIL import Image
    from playwright.sync_api import sync_playwright

    from isometric_nyc.e2e_generation.shared import (
        ensure_quadrant_exists,
        image_to_png_bytes,
        save_quadrant_render,
        split_tile_into_quadrants,
    )

    # Ensure the quadrant exists in the database
    quadrant = ensure_quadrant_exists(conn, config, qx, qy)

    # Find the top-left of the tile containing this quadrant
    # Quadrants come in 2x2 tiles, so we need to find which tile this is in
    tile_x = (qx // 2) * 2
    tile_y = (qy // 2) * 2

    # Get the top-left quadrant of the tile
    tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

    # Build URL for rendering
    params = {
        "export": "true",
        "lat": tl_quadrant["lat"],
        "lon": tl_quadrant["lng"],
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
            print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

        screenshot = page.screenshot()

        page.close()
        context.close()
        browser.close()

    # Convert to PIL Image
    from io import BytesIO

    tile_image = Image.open(BytesIO(screenshot))

    # Split into quadrants
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Save all 4 quadrants
    for (dx, dy), quad_img in quadrant_images.items():
        qx_save, qy_save = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)
        save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

    # Return the specific quadrant we wanted
    dx = qx - tile_x
    dy = qy - tile_y
    return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
    """Call the Oxen API to generate pixel art."""
    import requests

    endpoint = "https://hub.oxen.ai/api/images/edit"
    model = "cannoneyed-gentle-gold-antlion"
    prompt = (
        "Fill in the outlined section with the missing pixels corresponding to "
        "the <isometric nyc pixel art> style, removing the border and exactly "
        "following the shape/style/structure of the surrounding image (if present)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input_image": image_url,
        "prompt": prompt,
        "num_inference_steps": 28,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]["url"]
    elif "url" in result:
        return result["url"]
    elif "image_url" in result:
        return result["image_url"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> "Image.Image":
    """Download an image from a URL and return as PIL Image."""
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


# =============================================================================
# Load Generated Quadrants (override to avoid duplicate)
# =============================================================================


def _load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
    """Load all quadrants that have generations from the database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
    )
    return {Point(row[0], row[1]) for row in cursor.fetchall()}


# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            for step in steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

Now I need to fix the duplicate function definition:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Step Execution
# =============================================================================


def run_generation_step(
    conn: sqlite3.Connection,
    config: dict,
    quadrant_tuples: list[tuple[int, int]],
    generation_dir: Path,
    port: int,
) -> dict:
    """
    Run a single generation step for the given quadrants.

    This uses the flexible TemplateBuilder approach that can handle
    1, 2, or 4 quadrant selections.

    Returns dict with success status and message/error.
    """
    import os
    import tempfile

    import requests
    from dotenv import load_dotenv
    from PIL import Image

    from isometric_nyc.e2e_generation.infill_template import (
        QUADRANT_SIZE,
        InfillRegion,
        TemplateBuilder,
        validate_quadrant_selection,
    )
    from isometric_nyc.e2e_generation.shared import (
        get_quadrant_generation as shared_get_quadrant_generation,
        get_quadrant_render as shared_get_quadrant_render,
        image_to_png_bytes,
        png_bytes_to_image,
        save_quadrant_generation,
        upload_to_gcs,
    )

    load_dotenv()

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
        }

    # Helper functions
    def has_generation_in_db(qx: int, qy: int) -> bool:
        gen = shared_get_quadrant_generation(conn, qx, qy)
        return gen is not None

    def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
        """Get render, rendering if it doesn't exist yet."""
        render_bytes = shared_get_quadrant_render(conn, qx, qy)
        if render_bytes:
            return png_bytes_to_image(render_bytes)

        # Need to render
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)
        if render_bytes:
            return png_bytes_to_image(render_bytes)
        return None

    def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
        gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
            return png_bytes_to_image(gen_bytes)
        return None

    # Validate selection with auto-expansion
    is_valid, msg, placement = validate_quadrant_selection(
        quadrant_tuples, has_generation_in_db, allow_expansion=True
    )

    if not is_valid:
        return {"success": False, "error": msg}

    print(f"   ✅ Validation: {msg}")

    # Get primary quadrants (the ones we selected, not padding)
    primary_quadrants = (
        placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
    )
    padding_quadrants = placement.padding_quadrants if placement else []

    if padding_quadrants:
        print(f"   📦 Padding quadrants: {padding_quadrants}")

    # Create the infill region (may be expanded)
    if placement._expanded_region is not None:
        region = placement._expanded_region
    else:
        region = InfillRegion.from_quadrants(quadrant_tuples)

    # Build the template
    print("   🎨 Building template image...")
    builder = TemplateBuilder(
        region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
    )

    result = builder.build(border_width=2, allow_expansion=True)

    if result is None:
        error_msg = builder._last_validation_error or "Failed to build template"
        return {"success": False, "error": error_msg}

    template_image, placement = result

    # Save template to temp file and upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

    try:
        print("   📤 Uploading template to GCS...")
        image_url = upload_to_gcs(template_path, "isometric-nyc-infills")

        print("   🤖 Calling Oxen API...")
        generated_url = call_oxen_api(image_url, api_key)

        print("   📥 Downloading generated image...")
        generated_image = download_image_to_pil(generated_url)

        # Extract quadrants from generated image and save to database
        print("   💾 Saving generated quadrants to database...")

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
                    print(f"      ✓ Saved generation for ({qx}, {qy})")
                    saved_count += 1
                else:
                    print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
            else:
                print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

        return {
            "success": True,
            "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
            "quadrants": primary_quadrants,
        }

    finally:
        # Clean up temp file
        template_path.unlink(missing_ok=True)


def render_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    qx: int,
    qy: int,
    port: int,
) -> bytes | None:
    """Render a single quadrant using the web server."""
    from urllib.parse import urlencode

    from PIL import Image
    from playwright.sync_api import sync_playwright

    from isometric_nyc.e2e_generation.shared import (
        ensure_quadrant_exists,
        image_to_png_bytes,
        save_quadrant_render,
        split_tile_into_quadrants,
    )

    # Ensure the quadrant exists in the database
    quadrant = ensure_quadrant_exists(conn, config, qx, qy)

    # Find the top-left of the tile containing this quadrant
    # Quadrants come in 2x2 tiles, so we need to find which tile this is in
    tile_x = (qx // 2) * 2
    tile_y = (qy // 2) * 2

    # Get the top-left quadrant of the tile
    tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

    # Build URL for rendering
    params = {
        "export": "true",
        "lat": tl_quadrant["lat"],
        "lon": tl_quadrant["lng"],
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
            print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

        screenshot = page.screenshot()

        page.close()
        context.close()
        browser.close()

    # Convert to PIL Image
    from io import BytesIO

    tile_image = Image.open(BytesIO(screenshot))

    # Split into quadrants
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Save all 4 quadrants
    for (dx, dy), quad_img in quadrant_images.items():
        qx_save, qy_save = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)
        save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

    # Return the specific quadrant we wanted
    dx = qx - tile_x
    dy = qy - tile_y
    return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
    """Call the Oxen API to generate pixel art."""
    import requests

    endpoint = "https://hub.oxen.ai/api/images/edit"
    model = "cannoneyed-gentle-gold-antlion"
    prompt = (
        "Fill in the outlined section with the missing pixels corresponding to "
        "the <isometric nyc pixel art> style, removing the border and exactly "
        "following the shape/style/structure of the surrounding image (if present)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input_image": image_url,
        "prompt": prompt,
        "num_inference_steps": 28,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]["url"]
    elif "url" in result:
        return result["url"]
    elif "image_url" in result:
        return result["image_url"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> "Image.Image":
    """Download an image from a URL and return as PIL Image."""
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))




# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            for step in steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

Let me verify the script still works with dry-run:

The script is working. Now let me also add a `--max-steps` option to limit how many steps to execute (useful for testing), and let me add some additional helpful options. Let me also verify the help output:

Let me add a few more useful options including `--max-steps` for testing and `--start-from` to resume from a specific step:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Step Execution
# =============================================================================


def run_generation_step(
    conn: sqlite3.Connection,
    config: dict,
    quadrant_tuples: list[tuple[int, int]],
    generation_dir: Path,
    port: int,
) -> dict:
    """
    Run a single generation step for the given quadrants.

    This uses the flexible TemplateBuilder approach that can handle
    1, 2, or 4 quadrant selections.

    Returns dict with success status and message/error.
    """
    import os
    import tempfile

    import requests
    from dotenv import load_dotenv
    from PIL import Image

    from isometric_nyc.e2e_generation.infill_template import (
        QUADRANT_SIZE,
        InfillRegion,
        TemplateBuilder,
        validate_quadrant_selection,
    )
    from isometric_nyc.e2e_generation.shared import (
        get_quadrant_generation as shared_get_quadrant_generation,
        get_quadrant_render as shared_get_quadrant_render,
        image_to_png_bytes,
        png_bytes_to_image,
        save_quadrant_generation,
        upload_to_gcs,
    )

    load_dotenv()

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
        }

    # Helper functions
    def has_generation_in_db(qx: int, qy: int) -> bool:
        gen = shared_get_quadrant_generation(conn, qx, qy)
        return gen is not None

    def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
        """Get render, rendering if it doesn't exist yet."""
        render_bytes = shared_get_quadrant_render(conn, qx, qy)
        if render_bytes:
            return png_bytes_to_image(render_bytes)

        # Need to render
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)
        if render_bytes:
            return png_bytes_to_image(render_bytes)
        return None

    def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
        gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
            return png_bytes_to_image(gen_bytes)
        return None

    # Validate selection with auto-expansion
    is_valid, msg, placement = validate_quadrant_selection(
        quadrant_tuples, has_generation_in_db, allow_expansion=True
    )

    if not is_valid:
        return {"success": False, "error": msg}

    print(f"   ✅ Validation: {msg}")

    # Get primary quadrants (the ones we selected, not padding)
    primary_quadrants = (
        placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
    )
    padding_quadrants = placement.padding_quadrants if placement else []

    if padding_quadrants:
        print(f"   📦 Padding quadrants: {padding_quadrants}")

    # Create the infill region (may be expanded)
    if placement._expanded_region is not None:
        region = placement._expanded_region
    else:
        region = InfillRegion.from_quadrants(quadrant_tuples)

    # Build the template
    print("   🎨 Building template image...")
    builder = TemplateBuilder(
        region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
    )

    result = builder.build(border_width=2, allow_expansion=True)

    if result is None:
        error_msg = builder._last_validation_error or "Failed to build template"
        return {"success": False, "error": error_msg}

    template_image, placement = result

    # Save template to temp file and upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

    try:
        print("   📤 Uploading template to GCS...")
        image_url = upload_to_gcs(template_path, "isometric-nyc-infills")

        print("   🤖 Calling Oxen API...")
        generated_url = call_oxen_api(image_url, api_key)

        print("   📥 Downloading generated image...")
        generated_image = download_image_to_pil(generated_url)

        # Extract quadrants from generated image and save to database
        print("   💾 Saving generated quadrants to database...")

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
                    print(f"      ✓ Saved generation for ({qx}, {qy})")
                    saved_count += 1
                else:
                    print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
            else:
                print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

        return {
            "success": True,
            "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
            "quadrants": primary_quadrants,
        }

    finally:
        # Clean up temp file
        template_path.unlink(missing_ok=True)


def render_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    qx: int,
    qy: int,
    port: int,
) -> bytes | None:
    """Render a single quadrant using the web server."""
    from urllib.parse import urlencode

    from PIL import Image
    from playwright.sync_api import sync_playwright

    from isometric_nyc.e2e_generation.shared import (
        ensure_quadrant_exists,
        image_to_png_bytes,
        save_quadrant_render,
        split_tile_into_quadrants,
    )

    # Ensure the quadrant exists in the database
    quadrant = ensure_quadrant_exists(conn, config, qx, qy)

    # Find the top-left of the tile containing this quadrant
    # Quadrants come in 2x2 tiles, so we need to find which tile this is in
    tile_x = (qx // 2) * 2
    tile_y = (qy // 2) * 2

    # Get the top-left quadrant of the tile
    tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

    # Build URL for rendering
    params = {
        "export": "true",
        "lat": tl_quadrant["lat"],
        "lon": tl_quadrant["lng"],
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
            print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

        screenshot = page.screenshot()

        page.close()
        context.close()
        browser.close()

    # Convert to PIL Image
    from io import BytesIO

    tile_image = Image.open(BytesIO(screenshot))

    # Split into quadrants
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Save all 4 quadrants
    for (dx, dy), quad_img in quadrant_images.items():
        qx_save, qy_save = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)
        save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

    # Return the specific quadrant we wanted
    dx = qx - tile_x
    dy = qy - tile_y
    return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
    """Call the Oxen API to generate pixel art."""
    import requests

    endpoint = "https://hub.oxen.ai/api/images/edit"
    model = "cannoneyed-gentle-gold-antlion"
    prompt = (
        "Fill in the outlined section with the missing pixels corresponding to "
        "the <isometric nyc pixel art> style, removing the border and exactly "
        "following the shape/style/structure of the surrounding image (if present)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input_image": image_url,
        "prompt": prompt,
        "num_inference_steps": 28,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]["url"]
    elif "url" in result:
        return result["url"]
    elif "image_url" in result:
        return result["image_url"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> "Image.Image":
    """Download an image from a URL and return as PIL Image."""
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))




# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to execute (for testing)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from step N (default: 1)",
    )
    parser.add_argument(
        "--bucket",
        default="isometric-nyc-infills",
        help="GCS bucket name for uploading images",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            for step in steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

Now update the execution section to use these options:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Step Execution
# =============================================================================


def run_generation_step(
    conn: sqlite3.Connection,
    config: dict,
    quadrant_tuples: list[tuple[int, int]],
    generation_dir: Path,
    port: int,
) -> dict:
    """
    Run a single generation step for the given quadrants.

    This uses the flexible TemplateBuilder approach that can handle
    1, 2, or 4 quadrant selections.

    Returns dict with success status and message/error.
    """
    import os
    import tempfile

    import requests
    from dotenv import load_dotenv
    from PIL import Image

    from isometric_nyc.e2e_generation.infill_template import (
        QUADRANT_SIZE,
        InfillRegion,
        TemplateBuilder,
        validate_quadrant_selection,
    )
    from isometric_nyc.e2e_generation.shared import (
        get_quadrant_generation as shared_get_quadrant_generation,
        get_quadrant_render as shared_get_quadrant_render,
        image_to_png_bytes,
        png_bytes_to_image,
        save_quadrant_generation,
        upload_to_gcs,
    )

    load_dotenv()

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
        }

    # Helper functions
    def has_generation_in_db(qx: int, qy: int) -> bool:
        gen = shared_get_quadrant_generation(conn, qx, qy)
        return gen is not None

    def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
        """Get render, rendering if it doesn't exist yet."""
        render_bytes = shared_get_quadrant_render(conn, qx, qy)
        if render_bytes:
            return png_bytes_to_image(render_bytes)

        # Need to render
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)
        if render_bytes:
            return png_bytes_to_image(render_bytes)
        return None

    def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
        gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
            return png_bytes_to_image(gen_bytes)
        return None

    # Validate selection with auto-expansion
    is_valid, msg, placement = validate_quadrant_selection(
        quadrant_tuples, has_generation_in_db, allow_expansion=True
    )

    if not is_valid:
        return {"success": False, "error": msg}

    print(f"   ✅ Validation: {msg}")

    # Get primary quadrants (the ones we selected, not padding)
    primary_quadrants = (
        placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
    )
    padding_quadrants = placement.padding_quadrants if placement else []

    if padding_quadrants:
        print(f"   📦 Padding quadrants: {padding_quadrants}")

    # Create the infill region (may be expanded)
    if placement._expanded_region is not None:
        region = placement._expanded_region
    else:
        region = InfillRegion.from_quadrants(quadrant_tuples)

    # Build the template
    print("   🎨 Building template image...")
    builder = TemplateBuilder(
        region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
    )

    result = builder.build(border_width=2, allow_expansion=True)

    if result is None:
        error_msg = builder._last_validation_error or "Failed to build template"
        return {"success": False, "error": error_msg}

    template_image, placement = result

    # Save template to temp file and upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

    try:
        print("   📤 Uploading template to GCS...")
        image_url = upload_to_gcs(template_path, "isometric-nyc-infills")

        print("   🤖 Calling Oxen API...")
        generated_url = call_oxen_api(image_url, api_key)

        print("   📥 Downloading generated image...")
        generated_image = download_image_to_pil(generated_url)

        # Extract quadrants from generated image and save to database
        print("   💾 Saving generated quadrants to database...")

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
                    print(f"      ✓ Saved generation for ({qx}, {qy})")
                    saved_count += 1
                else:
                    print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
            else:
                print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

        return {
            "success": True,
            "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
            "quadrants": primary_quadrants,
        }

    finally:
        # Clean up temp file
        template_path.unlink(missing_ok=True)


def render_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    qx: int,
    qy: int,
    port: int,
) -> bytes | None:
    """Render a single quadrant using the web server."""
    from urllib.parse import urlencode

    from PIL import Image
    from playwright.sync_api import sync_playwright

    from isometric_nyc.e2e_generation.shared import (
        ensure_quadrant_exists,
        image_to_png_bytes,
        save_quadrant_render,
        split_tile_into_quadrants,
    )

    # Ensure the quadrant exists in the database
    quadrant = ensure_quadrant_exists(conn, config, qx, qy)

    # Find the top-left of the tile containing this quadrant
    # Quadrants come in 2x2 tiles, so we need to find which tile this is in
    tile_x = (qx // 2) * 2
    tile_y = (qy // 2) * 2

    # Get the top-left quadrant of the tile
    tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

    # Build URL for rendering
    params = {
        "export": "true",
        "lat": tl_quadrant["lat"],
        "lon": tl_quadrant["lng"],
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
            print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

        screenshot = page.screenshot()

        page.close()
        context.close()
        browser.close()

    # Convert to PIL Image
    from io import BytesIO

    tile_image = Image.open(BytesIO(screenshot))

    # Split into quadrants
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Save all 4 quadrants
    for (dx, dy), quad_img in quadrant_images.items():
        qx_save, qy_save = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)
        save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

    # Return the specific quadrant we wanted
    dx = qx - tile_x
    dy = qy - tile_y
    return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
    """Call the Oxen API to generate pixel art."""
    import requests

    endpoint = "https://hub.oxen.ai/api/images/edit"
    model = "cannoneyed-gentle-gold-antlion"
    prompt = (
        "Fill in the outlined section with the missing pixels corresponding to "
        "the <isometric nyc pixel art> style, removing the border and exactly "
        "following the shape/style/structure of the surrounding image (if present)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input_image": image_url,
        "prompt": prompt,
        "num_inference_steps": 28,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]["url"]
    elif "url" in result:
        return result["url"]
    elif "image_url" in result:
        return result["image_url"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> "Image.Image":
    """Download an image from a URL and return as PIL Image."""
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))




# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to execute (for testing)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from step N (default: 1)",
    )
    parser.add_argument(
        "--bucket",
        default="isometric-nyc-infills",
        help="GCS bucket name for uploading images",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            # Filter steps based on --start-from and --max-steps
            filtered_steps = [s for s in steps if s.step_number >= args.start_from]
            if args.max_steps is not None:
                filtered_steps = filtered_steps[: args.max_steps]

            if not filtered_steps:
                print("No steps to execute (check --start-from and --max-steps)")
                return 0

            print(f"   Executing steps {filtered_steps[0].step_number} to {filtered_steps[-1].step_number}")

            for step in filtered_steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

Let me also update the `run_generation_step` to use the bucket argument:

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Step Execution
# =============================================================================


def run_generation_step(
    conn: sqlite3.Connection,
    config: dict,
    quadrant_tuples: list[tuple[int, int]],
    generation_dir: Path,
    port: int,
    bucket_name: str = "isometric-nyc-infills",
) -> dict:
    """
    Run a single generation step for the given quadrants.

    This uses the flexible TemplateBuilder approach that can handle
    1, 2, or 4 quadrant selections.

    Returns dict with success status and message/error.
    """
    import os
    import tempfile

    import requests
    from dotenv import load_dotenv
    from PIL import Image

    from isometric_nyc.e2e_generation.infill_template import (
        QUADRANT_SIZE,
        InfillRegion,
        TemplateBuilder,
        validate_quadrant_selection,
    )
    from isometric_nyc.e2e_generation.shared import (
        get_quadrant_generation as shared_get_quadrant_generation,
        get_quadrant_render as shared_get_quadrant_render,
        image_to_png_bytes,
        png_bytes_to_image,
        save_quadrant_generation,
        upload_to_gcs,
    )

    load_dotenv()

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
        }

    # Helper functions
    def has_generation_in_db(qx: int, qy: int) -> bool:
        gen = shared_get_quadrant_generation(conn, qx, qy)
        return gen is not None

    def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
        """Get render, rendering if it doesn't exist yet."""
        render_bytes = shared_get_quadrant_render(conn, qx, qy)
        if render_bytes:
            return png_bytes_to_image(render_bytes)

        # Need to render
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)
        if render_bytes:
            return png_bytes_to_image(render_bytes)
        return None

    def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
        gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
            return png_bytes_to_image(gen_bytes)
        return None

    # Validate selection with auto-expansion
    is_valid, msg, placement = validate_quadrant_selection(
        quadrant_tuples, has_generation_in_db, allow_expansion=True
    )

    if not is_valid:
        return {"success": False, "error": msg}

    print(f"   ✅ Validation: {msg}")

    # Get primary quadrants (the ones we selected, not padding)
    primary_quadrants = (
        placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
    )
    padding_quadrants = placement.padding_quadrants if placement else []

    if padding_quadrants:
        print(f"   📦 Padding quadrants: {padding_quadrants}")

    # Create the infill region (may be expanded)
    if placement._expanded_region is not None:
        region = placement._expanded_region
    else:
        region = InfillRegion.from_quadrants(quadrant_tuples)

    # Build the template
    print("   🎨 Building template image...")
    builder = TemplateBuilder(
        region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
    )

    result = builder.build(border_width=2, allow_expansion=True)

    if result is None:
        error_msg = builder._last_validation_error or "Failed to build template"
        return {"success": False, "error": error_msg}

    template_image, placement = result

    # Save template to temp file and upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

    try:
        print("   📤 Uploading template to GCS...")
        image_url = upload_to_gcs(template_path, "isometric-nyc-infills")

        print("   🤖 Calling Oxen API...")
        generated_url = call_oxen_api(image_url, api_key)

        print("   📥 Downloading generated image...")
        generated_image = download_image_to_pil(generated_url)

        # Extract quadrants from generated image and save to database
        print("   💾 Saving generated quadrants to database...")

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
                    print(f"      ✓ Saved generation for ({qx}, {qy})")
                    saved_count += 1
                else:
                    print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
            else:
                print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

        return {
            "success": True,
            "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
            "quadrants": primary_quadrants,
        }

    finally:
        # Clean up temp file
        template_path.unlink(missing_ok=True)


def render_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    qx: int,
    qy: int,
    port: int,
) -> bytes | None:
    """Render a single quadrant using the web server."""
    from urllib.parse import urlencode

    from PIL import Image
    from playwright.sync_api import sync_playwright

    from isometric_nyc.e2e_generation.shared import (
        ensure_quadrant_exists,
        image_to_png_bytes,
        save_quadrant_render,
        split_tile_into_quadrants,
    )

    # Ensure the quadrant exists in the database
    quadrant = ensure_quadrant_exists(conn, config, qx, qy)

    # Find the top-left of the tile containing this quadrant
    # Quadrants come in 2x2 tiles, so we need to find which tile this is in
    tile_x = (qx // 2) * 2
    tile_y = (qy // 2) * 2

    # Get the top-left quadrant of the tile
    tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

    # Build URL for rendering
    params = {
        "export": "true",
        "lat": tl_quadrant["lat"],
        "lon": tl_quadrant["lng"],
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
            print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

        screenshot = page.screenshot()

        page.close()
        context.close()
        browser.close()

    # Convert to PIL Image
    from io import BytesIO

    tile_image = Image.open(BytesIO(screenshot))

    # Split into quadrants
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Save all 4 quadrants
    for (dx, dy), quad_img in quadrant_images.items():
        qx_save, qy_save = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)
        save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

    # Return the specific quadrant we wanted
    dx = qx - tile_x
    dy = qy - tile_y
    return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
    """Call the Oxen API to generate pixel art."""
    import requests

    endpoint = "https://hub.oxen.ai/api/images/edit"
    model = "cannoneyed-gentle-gold-antlion"
    prompt = (
        "Fill in the outlined section with the missing pixels corresponding to "
        "the <isometric nyc pixel art> style, removing the border and exactly "
        "following the shape/style/structure of the surrounding image (if present)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input_image": image_url,
        "prompt": prompt,
        "num_inference_steps": 28,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]["url"]
    elif "url" in result:
        return result["url"]
    elif "image_url" in result:
        return result["image_url"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> "Image.Image":
    """Download an image from a URL and return as PIL Image."""
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))




# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to execute (for testing)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from step N (default: 1)",
    )
    parser.add_argument(
        "--bucket",
        default="isometric-nyc-infills",
        help="GCS bucket name for uploading images",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            # Filter steps based on --start-from and --max-steps
            filtered_steps = [s for s in steps if s.step_number >= args.start_from]
            if args.max_steps is not None:
                filtered_steps = filtered_steps[: args.max_steps]

            if not filtered_steps:
                print("No steps to execute (check --start-from and --max-steps)")
                return 0

            print(f"   Executing steps {filtered_steps[0].step_number} to {filtered_steps[-1].step_number}")

            for step in filtered_steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Step Execution
# =============================================================================


def run_generation_step(
    conn: sqlite3.Connection,
    config: dict,
    quadrant_tuples: list[tuple[int, int]],
    generation_dir: Path,
    port: int,
    bucket_name: str = "isometric-nyc-infills",
) -> dict:
    """
    Run a single generation step for the given quadrants.

    This uses the flexible TemplateBuilder approach that can handle
    1, 2, or 4 quadrant selections.

    Returns dict with success status and message/error.
    """
    import os
    import tempfile

    import requests
    from dotenv import load_dotenv
    from PIL import Image

    from isometric_nyc.e2e_generation.infill_template import (
        QUADRANT_SIZE,
        InfillRegion,
        TemplateBuilder,
        validate_quadrant_selection,
    )
    from isometric_nyc.e2e_generation.shared import (
        get_quadrant_generation as shared_get_quadrant_generation,
        get_quadrant_render as shared_get_quadrant_render,
        image_to_png_bytes,
        png_bytes_to_image,
        save_quadrant_generation,
        upload_to_gcs,
    )

    load_dotenv()

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
        }

    # Helper functions
    def has_generation_in_db(qx: int, qy: int) -> bool:
        gen = shared_get_quadrant_generation(conn, qx, qy)
        return gen is not None

    def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
        """Get render, rendering if it doesn't exist yet."""
        render_bytes = shared_get_quadrant_render(conn, qx, qy)
        if render_bytes:
            return png_bytes_to_image(render_bytes)

        # Need to render
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)
        if render_bytes:
            return png_bytes_to_image(render_bytes)
        return None

    def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
        gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
            return png_bytes_to_image(gen_bytes)
        return None

    # Validate selection with auto-expansion
    is_valid, msg, placement = validate_quadrant_selection(
        quadrant_tuples, has_generation_in_db, allow_expansion=True
    )

    if not is_valid:
        return {"success": False, "error": msg}

    print(f"   ✅ Validation: {msg}")

    # Get primary quadrants (the ones we selected, not padding)
    primary_quadrants = (
        placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
    )
    padding_quadrants = placement.padding_quadrants if placement else []

    if padding_quadrants:
        print(f"   📦 Padding quadrants: {padding_quadrants}")

    # Create the infill region (may be expanded)
    if placement._expanded_region is not None:
        region = placement._expanded_region
    else:
        region = InfillRegion.from_quadrants(quadrant_tuples)

    # Build the template
    print("   🎨 Building template image...")
    builder = TemplateBuilder(
        region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
    )

    result = builder.build(border_width=2, allow_expansion=True)

    if result is None:
        error_msg = builder._last_validation_error or "Failed to build template"
        return {"success": False, "error": error_msg}

    template_image, placement = result

    # Save template to temp file and upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

    try:
        print("   📤 Uploading template to GCS...")
        image_url = upload_to_gcs(template_path, bucket_name)

        print("   🤖 Calling Oxen API...")
        generated_url = call_oxen_api(image_url, api_key)

        print("   📥 Downloading generated image...")
        generated_image = download_image_to_pil(generated_url)

        # Extract quadrants from generated image and save to database
        print("   💾 Saving generated quadrants to database...")

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
                    print(f"      ✓ Saved generation for ({qx}, {qy})")
                    saved_count += 1
                else:
                    print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
            else:
                print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

        return {
            "success": True,
            "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
            "quadrants": primary_quadrants,
        }

    finally:
        # Clean up temp file
        template_path.unlink(missing_ok=True)


def render_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    qx: int,
    qy: int,
    port: int,
) -> bytes | None:
    """Render a single quadrant using the web server."""
    from urllib.parse import urlencode

    from PIL import Image
    from playwright.sync_api import sync_playwright

    from isometric_nyc.e2e_generation.shared import (
        ensure_quadrant_exists,
        image_to_png_bytes,
        save_quadrant_render,
        split_tile_into_quadrants,
    )

    # Ensure the quadrant exists in the database
    quadrant = ensure_quadrant_exists(conn, config, qx, qy)

    # Find the top-left of the tile containing this quadrant
    # Quadrants come in 2x2 tiles, so we need to find which tile this is in
    tile_x = (qx // 2) * 2
    tile_y = (qy // 2) * 2

    # Get the top-left quadrant of the tile
    tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

    # Build URL for rendering
    params = {
        "export": "true",
        "lat": tl_quadrant["lat"],
        "lon": tl_quadrant["lng"],
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
            print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

        screenshot = page.screenshot()

        page.close()
        context.close()
        browser.close()

    # Convert to PIL Image
    from io import BytesIO

    tile_image = Image.open(BytesIO(screenshot))

    # Split into quadrants
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Save all 4 quadrants
    for (dx, dy), quad_img in quadrant_images.items():
        qx_save, qy_save = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)
        save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

    # Return the specific quadrant we wanted
    dx = qx - tile_x
    dy = qy - tile_y
    return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
    """Call the Oxen API to generate pixel art."""
    import requests

    endpoint = "https://hub.oxen.ai/api/images/edit"
    model = "cannoneyed-gentle-gold-antlion"
    prompt = (
        "Fill in the outlined section with the missing pixels corresponding to "
        "the <isometric nyc pixel art> style, removing the border and exactly "
        "following the shape/style/structure of the surrounding image (if present)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input_image": image_url,
        "prompt": prompt,
        "num_inference_steps": 28,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]["url"]
    elif "url" in result:
        return result["url"]
    elif "image_url" in result:
        return result["image_url"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> "Image.Image":
    """Download an image from a URL and return as PIL Image."""
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))




# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to execute (for testing)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from step N (default: 1)",
    )
    parser.add_argument(
        "--bucket",
        default="isometric-nyc-infills",
        help="GCS bucket name for uploading images",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            # Filter steps based on --start-from and --max-steps
            filtered_steps = [s for s in steps if s.step_number >= args.start_from]
            if args.max_steps is not None:
                filtered_steps = filtered_steps[: args.max_steps]

            if not filtered_steps:
                print("No steps to execute (check --start-from and --max-steps)")
                return 0

            print(f"   Executing steps {filtered_steps[0].step_number} to {filtered_steps[-1].step_number}")

            for step in filtered_steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

```python
"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles offset from center
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y> \\
    [--dry-run]

Example:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20 \\
    --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
    """State of a quadrant in the grid."""

    EMPTY = "empty"  # Not generated
    GENERATED = "generated"  # Already has generation
    SELECTED = "selected"  # Selected for generation in current step


@dataclass(frozen=True)
class Point:
    """A 2D point representing a quadrant coordinate."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class BoundingBox:
    """Bounding box defined by top-left and bottom-right corners."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> int:
        return self.bottom_right.x - self.top_left.x + 1

    @property
    def height(self) -> int:
        return self.bottom_right.y - self.top_left.y + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, p: Point) -> bool:
        return (
            self.top_left.x <= p.x <= self.bottom_right.x
            and self.top_left.y <= p.y <= self.bottom_right.y
        )

    def all_points(self) -> list[Point]:
        """Return all points within the bounding box."""
        return [
            Point(x, y)
            for y in range(self.top_left.y, self.bottom_right.y + 1)
            for x in range(self.top_left.x, self.bottom_right.x + 1)
        ]


@dataclass
class GenerationStep:
    """A single step in the generation plan."""

    step_number: int
    quadrants: list[Point]  # List of quadrant positions to generate
    description: str


# =============================================================================
# Grid Class
# =============================================================================


class QuadrantGrid:
    """
    In-memory grid of quadrant states.

    This class manages the state of all quadrants within a bounding box
    and helps construct generation plans.
    """

    def __init__(self, bounds: BoundingBox):
        self.bounds = bounds
        self._states: dict[Point, QuadrantState] = {}

        # Initialize all quadrants as empty
        for p in bounds.all_points():
            self._states[p] = QuadrantState.EMPTY

    def get_state(self, p: Point) -> QuadrantState:
        """Get the state of a quadrant."""
        return self._states.get(p, QuadrantState.EMPTY)

    def set_state(self, p: Point, state: QuadrantState) -> None:
        """Set the state of a quadrant."""
        if p in self._states:
            self._states[p] = state

    def is_generated(self, p: Point) -> bool:
        """Check if a quadrant has been generated."""
        return self.get_state(p) == QuadrantState.GENERATED

    def mark_generated(self, p: Point) -> None:
        """Mark a quadrant as generated."""
        self.set_state(p, QuadrantState.GENERATED)

    def mark_multiple_generated(self, points: list[Point]) -> None:
        """Mark multiple quadrants as generated."""
        for p in points:
            self.mark_generated(p)

    def get_all_generated(self) -> list[Point]:
        """Get all generated quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.GENERATED]

    def get_all_empty(self) -> list[Point]:
        """Get all empty (not generated) quadrant positions."""
        return [p for p, state in self._states.items() if state == QuadrantState.EMPTY]

    def get_generated_bounds(self) -> BoundingBox | None:
        """Get the bounding box of all generated quadrants."""
        generated = self.get_all_generated()
        if not generated:
            return None

        min_x = min(p.x for p in generated)
        max_x = max(p.x for p in generated)
        min_y = min(p.y for p in generated)
        max_y = max(p.y for p in generated)

        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def has_generated_neighbor(self, p: Point) -> bool:
        """Check if a quadrant has any generated neighbors (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return any(self.is_generated(n) for n in neighbors)

    def count_generated_neighbors(self, p: Point) -> int:
        """Count how many generated neighbors a quadrant has (4-connected)."""
        neighbors = [
            Point(p.x - 1, p.y),
            Point(p.x + 1, p.y),
            Point(p.x, p.y - 1),
            Point(p.x, p.y + 1),
        ]
        return sum(1 for n in neighbors if self.is_generated(n))

    def visualize(
        self,
        highlight: list[Point] | None = None,
        step_number: int | None = None,
    ) -> str:
        """
        Create an ASCII visualization of the grid.

        Legend:
          G = Generated
          . = Empty
          S = Selected (highlighted)
        """
        lines = []
        if step_number is not None:
            lines.append(f"Step {step_number}:")
        else:
            lines.append("Current state:")

        highlight_set = set(highlight or [])

        # Header with x coordinates
        x_range = range(self.bounds.top_left.x, self.bounds.bottom_right.x + 1)
        header = "   " + " ".join(f"{x:2d}" for x in x_range)
        lines.append(header)
        lines.append("   " + "-" * (len(x_range) * 3 - 1))

        for y in range(self.bounds.top_left.y, self.bounds.bottom_right.y + 1):
            row = f"{y:2d}|"
            for x in x_range:
                p = Point(x, y)
                if p in highlight_set:
                    char = " S"
                elif self.is_generated(p):
                    char = " G"
                else:
                    char = " ."
                row += char + " "
            lines.append(row)

        return "\n".join(lines)


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
# Generation Step Execution
# =============================================================================


def run_generation_step(
    conn: sqlite3.Connection,
    config: dict,
    quadrant_tuples: list[tuple[int, int]],
    generation_dir: Path,
    port: int,
    bucket_name: str = "isometric-nyc-infills",
) -> dict:
    """
    Run a single generation step for the given quadrants.

    This uses the flexible TemplateBuilder approach that can handle
    1, 2, or 4 quadrant selections.

    Returns dict with success status and message/error.
    """
    import os
    import tempfile

    import requests
    from dotenv import load_dotenv
    from PIL import Image

    from isometric_nyc.e2e_generation.infill_template import (
        QUADRANT_SIZE,
        InfillRegion,
        TemplateBuilder,
        validate_quadrant_selection,
    )
    from isometric_nyc.e2e_generation.shared import (
        get_quadrant_generation as shared_get_quadrant_generation,
        get_quadrant_render as shared_get_quadrant_render,
        image_to_png_bytes,
        png_bytes_to_image,
        save_quadrant_generation,
        upload_to_gcs,
    )

    load_dotenv()

    # Check for API key
    api_key = os.getenv("OXEN_OMNI_v04_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "OXEN_OMNI_v04_API_KEY environment variable not set",
        }

    # Helper functions
    def has_generation_in_db(qx: int, qy: int) -> bool:
        gen = shared_get_quadrant_generation(conn, qx, qy)
        return gen is not None

    def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
        """Get render, rendering if it doesn't exist yet."""
        render_bytes = shared_get_quadrant_render(conn, qx, qy)
        if render_bytes:
            return png_bytes_to_image(render_bytes)

        # Need to render
        print(f"   📦 Rendering quadrant ({qx}, {qy})...")
        render_bytes = render_quadrant(conn, config, qx, qy, port)
        if render_bytes:
            return png_bytes_to_image(render_bytes)
        return None

    def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
        gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
        if gen_bytes:
            return png_bytes_to_image(gen_bytes)
        return None

    # Validate selection with auto-expansion
    is_valid, msg, placement = validate_quadrant_selection(
        quadrant_tuples, has_generation_in_db, allow_expansion=True
    )

    if not is_valid:
        return {"success": False, "error": msg}

    print(f"   ✅ Validation: {msg}")

    # Get primary quadrants (the ones we selected, not padding)
    primary_quadrants = (
        placement.primary_quadrants if placement.primary_quadrants else quadrant_tuples
    )
    padding_quadrants = placement.padding_quadrants if placement else []

    if padding_quadrants:
        print(f"   📦 Padding quadrants: {padding_quadrants}")

    # Create the infill region (may be expanded)
    if placement._expanded_region is not None:
        region = placement._expanded_region
    else:
        region = InfillRegion.from_quadrants(quadrant_tuples)

    # Build the template
    print("   🎨 Building template image...")
    builder = TemplateBuilder(
        region, has_generation_in_db, get_render_from_db_with_render, get_generation_from_db
    )

    result = builder.build(border_width=2, allow_expansion=True)

    if result is None:
        error_msg = builder._last_validation_error or "Failed to build template"
        return {"success": False, "error": error_msg}

    template_image, placement = result

    # Save template to temp file and upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

    try:
        print("   📤 Uploading template to GCS...")
        image_url = upload_to_gcs(template_path, bucket_name)

        print("   🤖 Calling Oxen API...")
        generated_url = call_oxen_api(image_url, api_key)

        print("   📥 Downloading generated image...")
        generated_image = download_image_to_pil(generated_url)

        # Extract quadrants from generated image and save to database
        print("   💾 Saving generated quadrants to database...")

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
                    print(f"      ✓ Saved generation for ({qx}, {qy})")
                    saved_count += 1
                else:
                    print(f"      ⚠️ Failed to save generation for ({qx}, {qy})")
            else:
                print(f"      ⏭️ Skipped padding quadrant ({qx}, {qy})")

        return {
            "success": True,
            "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
            "quadrants": primary_quadrants,
        }

    finally:
        # Clean up temp file
        template_path.unlink(missing_ok=True)


def render_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    qx: int,
    qy: int,
    port: int,
) -> bytes | None:
    """Render a single quadrant using the web server."""
    from urllib.parse import urlencode

    from PIL import Image
    from playwright.sync_api import sync_playwright

    from isometric_nyc.e2e_generation.shared import (
        ensure_quadrant_exists,
        image_to_png_bytes,
        save_quadrant_render,
        split_tile_into_quadrants,
    )

    # Ensure the quadrant exists in the database
    quadrant = ensure_quadrant_exists(conn, config, qx, qy)

    # Find the top-left of the tile containing this quadrant
    # Quadrants come in 2x2 tiles, so we need to find which tile this is in
    tile_x = (qx // 2) * 2
    tile_y = (qy // 2) * 2

    # Get the top-left quadrant of the tile
    tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

    # Build URL for rendering
    params = {
        "export": "true",
        "lat": tl_quadrant["lat"],
        "lon": tl_quadrant["lng"],
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
            print("      ⚠️ Timeout waiting for tiles, continuing anyway...")

        screenshot = page.screenshot()

        page.close()
        context.close()
        browser.close()

    # Convert to PIL Image
    from io import BytesIO

    tile_image = Image.open(BytesIO(screenshot))

    # Split into quadrants
    quadrant_images = split_tile_into_quadrants(tile_image)

    # Save all 4 quadrants
    for (dx, dy), quad_img in quadrant_images.items():
        qx_save, qy_save = tile_x + dx, tile_y + dy
        png_bytes = image_to_png_bytes(quad_img)
        save_quadrant_render(conn, config, qx_save, qy_save, png_bytes)

    # Return the specific quadrant we wanted
    dx = qx - tile_x
    dy = qy - tile_y
    return image_to_png_bytes(quadrant_images[(dx, dy)])


def call_oxen_api(image_url: str, api_key: str) -> str:
    """Call the Oxen API to generate pixel art."""
    import requests

    endpoint = "https://hub.oxen.ai/api/images/edit"
    model = "cannoneyed-gentle-gold-antlion"
    prompt = (
        "Fill in the outlined section with the missing pixels corresponding to "
        "the <isometric nyc pixel art> style, removing the border and exactly "
        "following the shape/style/structure of the surrounding image (if present)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input_image": image_url,
        "prompt": prompt,
        "num_inference_steps": 28,
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()

    if "images" in result and len(result["images"]) > 0:
        return result["images"][0]["url"]
    elif "url" in result:
        return result["url"]
    elif "image_url" in result:
        return result["image_url"]
    elif "output" in result:
        return result["output"]
    else:
        raise ValueError(f"Unexpected API response format: {result}")


def download_image_to_pil(url: str) -> "Image.Image":
    """Download an image from a URL and return as PIL Image."""
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))




# =============================================================================
# Generation Plan Algorithm
# =============================================================================


def find_interior_gaps(grid: QuadrantGrid) -> list[Point]:
    """
    Find empty quadrants that are inside the generated bounds.

    These need to be filled before expanding outward.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    gaps = []
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y + 1):
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x + 1):
            p = Point(x, y)
            if not grid.is_generated(p) and grid.bounds.contains(p):
                gaps.append(p)

    return gaps


def can_generate_2x2(grid: QuadrantGrid, top_left: Point) -> bool:
    """
    Check if a 2x2 tile starting at top_left can be generated.

    A 2x2 tile can be generated if:
    - All 4 quadrants are empty (not already generated)
    - At least one quadrant has a generated neighbor outside the 2x2
    """
    quadrants = [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]

    # All must be empty
    if any(grid.is_generated(q) for q in quadrants):
        return False

    # All must be within bounds
    if not all(grid.bounds.contains(q) for q in quadrants):
        return False

    # At least one must have a generated neighbor (outside the 2x2)
    quadrant_set = set(quadrants)
    for q in quadrants:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in quadrant_set and grid.is_generated(n):
                return True

    return False


def can_generate_1x2_horizontal(grid: QuadrantGrid, left: Point) -> bool:
    """Check if a 1x2 horizontal tile can be generated."""
    right = Point(left.x + 1, left.y)

    if grid.is_generated(left) or grid.is_generated(right):
        return False
    if not grid.bounds.contains(left) or not grid.bounds.contains(right):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {left, right}
    for q in [left, right]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
    """Check if a 2x1 vertical tile can be generated."""
    bottom = Point(top.x, top.y + 1)

    if grid.is_generated(top) or grid.is_generated(bottom):
        return False
    if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
        return False

    # At least one must have generated neighbor outside the pair
    pair = {top, bottom}
    for q in [top, bottom]:
        neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
        ]
        for n in neighbors:
            if n not in pair and grid.is_generated(n):
                return True

    return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
    """Check if a single quadrant can be generated."""
    if grid.is_generated(p):
        return False
    if not grid.bounds.contains(p):
        return False
    return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(grid: QuadrantGrid, direction: str) -> list[Point]:
    """
    Find all valid 2x2 tile positions along a direction from the generated region.

    Returns list of top-left corners for valid 2x2 tiles.
    """
    gen_bounds = grid.get_generated_bounds()
    if gen_bounds is None:
        return []

    valid_positions = []

    if direction == "top":
        # Look for 2x2 tiles above the current bounds
        y = gen_bounds.top_left.y - 2
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "bottom":
        # Look for 2x2 tiles below the current bounds
        y = gen_bounds.bottom_right.y + 1
        for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "left":
        # Look for 2x2 tiles to the left of current bounds
        x = gen_bounds.top_left.x - 2
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    elif direction == "right":
        # Look for 2x2 tiles to the right of current bounds
        x = gen_bounds.bottom_right.x + 1
        for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
            tl = Point(x, y)
            if can_generate_2x2(grid, tl):
                valid_positions.append(tl)

    return valid_positions


def get_2x2_quadrants(top_left: Point) -> list[Point]:
    """Get all 4 quadrants for a 2x2 tile starting at top_left."""
    return [
        top_left,
        Point(top_left.x + 1, top_left.y),
        Point(top_left.x, top_left.y + 1),
        Point(top_left.x + 1, top_left.y + 1),
    ]


def get_1x2_quadrants(left: Point) -> list[Point]:
    """Get both quadrants for a 1x2 horizontal tile."""
    return [left, Point(left.x + 1, left.y)]


def get_2x1_quadrants(top: Point) -> list[Point]:
    """Get both quadrants for a 2x1 vertical tile."""
    return [top, Point(top.x, top.y + 1)]


def create_generation_plan(grid: QuadrantGrid) -> list[GenerationStep]:
    """
    Create an optimal generation plan to fill all empty quadrants.

    Strategy:
    1. Fill any interior gaps first (within current generated bounds)
    2. Expand outward in spiral pattern (top, right, bottom, left)
    3. For each direction:
       a. Generate 2x2 tiles offset from the edge
       b. Bridge the offset tiles to the center
       c. Bridge the offset tiles to each other
       d. Fill remaining single-quadrant gaps
    """
    steps: list[GenerationStep] = []
    step_num = 1

    # Phase 1: Fill interior gaps
    interior_gaps = find_interior_gaps(grid)
    if interior_gaps:
        # Try to fill gaps efficiently using largest possible tiles
        gap_steps = fill_gaps_efficiently(grid, interior_gaps)
        for quadrants, desc in gap_steps:
            steps.append(GenerationStep(step_num, quadrants, f"Interior fill: {desc}"))
            grid.mark_multiple_generated(quadrants)
            step_num += 1

    # Phase 2: Spiral expansion
    directions = ["top", "right", "bottom", "left"]
    direction_idx = 0
    max_iterations = 1000  # Safety limit

    while grid.get_all_empty() and max_iterations > 0:
        max_iterations -= 1
        made_progress = False

        # Try all 4 directions in order
        for _ in range(4):
            direction = directions[direction_idx]
            direction_idx = (direction_idx + 1) % 4

            # Step A: Generate offset 2x2 tiles in this direction
            offset_2x2 = find_best_2x2_tiles(grid, direction)
            for tl in offset_2x2:
                quadrants = get_2x2_quadrants(tl)
                steps.append(
                    GenerationStep(
                        step_num, quadrants, f"Offset 2x2 ({direction}): {tl}"
                    )
                )
                grid.mark_multiple_generated(quadrants)
                step_num += 1
                made_progress = True

        # Step B: Fill remaining gaps (bridges and single quadrants)
        remaining_empty = grid.get_all_empty()
        if remaining_empty:
            # Sort by how many generated neighbors they have (more = better)
            remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

            for p in remaining_empty:
                if grid.is_generated(p):
                    continue

                # Try 2x2 first
                for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                    tl = Point(p.x + dx, p.y + dy)
                    if can_generate_2x2(grid, tl):
                        quadrants = get_2x2_quadrants(tl)
                        steps.append(
                            GenerationStep(step_num, quadrants, f"Bridge 2x2: {tl}")
                        )
                        grid.mark_multiple_generated(quadrants)
                        step_num += 1
                        made_progress = True
                        break
                else:
                    # Try 1x2 horizontal
                    for dx in [0, -1]:
                        left = Point(p.x + dx, p.y)
                        if can_generate_1x2_horizontal(grid, left):
                            quadrants = get_1x2_quadrants(left)
                            steps.append(
                                GenerationStep(
                                    step_num, quadrants, f"Bridge 1x2 horizontal: {left}"
                                )
                            )
                            grid.mark_multiple_generated(quadrants)
                            step_num += 1
                            made_progress = True
                            break
                    else:
                        # Try 2x1 vertical
                        for dy in [0, -1]:
                            top = Point(p.x, p.y + dy)
                            if can_generate_2x1_vertical(grid, top):
                                quadrants = get_2x1_quadrants(top)
                                steps.append(
                                    GenerationStep(
                                        step_num,
                                        quadrants,
                                        f"Bridge 2x1 vertical: {top}",
                                    )
                                )
                                grid.mark_multiple_generated(quadrants)
                                step_num += 1
                                made_progress = True
                                break
                        else:
                            # Single quadrant as last resort
                            if can_generate_single(grid, p):
                                steps.append(
                                    GenerationStep(
                                        step_num, [p], f"Single quadrant: {p}"
                                    )
                                )
                                grid.mark_generated(p)
                                step_num += 1
                                made_progress = True

        if not made_progress:
            # Check if we have disconnected empty regions
            empty = grid.get_all_empty()
            if empty:
                # Find an empty quadrant adjacent to the generated region
                for p in empty:
                    if grid.has_generated_neighbor(p):
                        # Generate it as a single
                        steps.append(
                            GenerationStep(
                                step_num, [p], f"Single quadrant (fallback): {p}"
                            )
                        )
                        grid.mark_generated(p)
                        step_num += 1
                        made_progress = True
                        break

                if not made_progress:
                    # Truly stuck - there may be disconnected regions
                    print(
                        f"Warning: {len(empty)} quadrants cannot be reached from generated region"
                    )
                    break

    return steps


def fill_gaps_efficiently(
    grid: QuadrantGrid, gaps: list[Point]
) -> list[tuple[list[Point], str]]:
    """
    Fill interior gaps using the largest possible tile sizes.

    Returns list of (quadrants, description) tuples.
    """
    result = []
    gap_set = set(gaps)

    while gap_set:
        # Try to find a 2x2 that fills gaps
        found = False
        for p in list(gap_set):
            # Try all possible 2x2 positions that include this gap
            for dx, dy in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                tl = Point(p.x + dx, p.y + dy)
                quadrants = get_2x2_quadrants(tl)

                # Check if this is valid: all empty, at least one neighbor
                all_empty = all(not grid.is_generated(q) for q in quadrants)
                all_in_bounds = all(grid.bounds.contains(q) for q in quadrants)

                if all_empty and all_in_bounds:
                    # Check for generated neighbors
                    quad_set = set(quadrants)
                    has_neighbor = False
                    for q in quadrants:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in quad_set and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append((quadrants, f"2x2 at {tl}"))
                        grid.mark_multiple_generated(quadrants)
                        gap_set -= set(quadrants)
                        found = True
                        break

            if found:
                break

        if not found:
            # Fall back to smaller tiles
            for p in list(gap_set):
                if p not in gap_set:
                    continue

                # Try 1x2 horizontal
                right = Point(p.x + 1, p.y)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(right)
                    and grid.bounds.contains(right)
                ):
                    pair = {p, right}
                    has_neighbor = False
                    for q in [p, right]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, right], f"1x2 at {p}"))
                        grid.mark_multiple_generated([p, right])
                        gap_set -= {p, right}
                        found = True
                        break

                # Try 2x1 vertical
                bottom = Point(p.x, p.y + 1)
                if (
                    not grid.is_generated(p)
                    and not grid.is_generated(bottom)
                    and grid.bounds.contains(bottom)
                ):
                    pair = {p, bottom}
                    has_neighbor = False
                    for q in [p, bottom]:
                        neighbors = [
                            Point(q.x - 1, q.y),
                            Point(q.x + 1, q.y),
                            Point(q.x, q.y - 1),
                            Point(q.x, q.y + 1),
                        ]
                        for n in neighbors:
                            if n not in pair and grid.is_generated(n):
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break

                    if has_neighbor:
                        result.append(([p, bottom], f"2x1 at {p}"))
                        grid.mark_multiple_generated([p, bottom])
                        gap_set -= {p, bottom}
                        found = True
                        break

                # Single quadrant
                if grid.has_generated_neighbor(p):
                    result.append(([p], f"single at {p}"))
                    grid.mark_generated(p)
                    gap_set.discard(p)
                    found = True
                    break

            if not found:
                # Stuck - remaining gaps are unreachable
                print(f"Warning: {len(gap_set)} interior gaps unreachable")
                break

    return result


# =============================================================================
# Main Script
# =============================================================================


def parse_coordinate(s: str) -> Point:
    """Parse a coordinate string like '10,20' or '10, 20' into a Point."""
    parts = s.strip().replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {s}")
    return Point(int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Automatically generate tiles to fill a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "--top-left",
        type=str,
        required=True,
        help="Top-left corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--bottom-right",
        type=str,
        required=True,
        help="Bottom-right corner of bounding box (x,y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generation plan without executing",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII visualization of each step",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Web server port (default: 5173)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Don't start web server (assume it's already running)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to execute (for testing)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from step N (default: 1)",
    )
    parser.add_argument(
        "--bucket",
        default="isometric-nyc-infills",
        help="GCS bucket name for uploading images",
    )

    args = parser.parse_args()

    # Parse coordinates
    try:
        top_left = parse_coordinate(args.top_left)
        bottom_right = parse_coordinate(args.bottom_right)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate bounding box
    if top_left.x > bottom_right.x or top_left.y > bottom_right.y:
        print("❌ Error: top-left must be above and to the left of bottom-right")
        return 1

    bounds = BoundingBox(top_left, bottom_right)
    print(f"📦 Bounding box: {top_left} to {bottom_right}")
    print(f"   Size: {bounds.width} x {bounds.height} = {bounds.area} quadrants")

    # Load database
    generation_dir = args.generation_dir.resolve()
    db_path = generation_dir / "quadrants.db"

    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)

    try:
        # Load current state
        generated = load_generated_quadrants(conn)
        print(f"\n📊 Currently generated: {len(generated)} quadrants")

        # Create grid
        grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                grid.mark_generated(p)

        generated_in_bounds = len(grid.get_all_generated())
        empty_in_bounds = len(grid.get_all_empty())
        print(f"   Within bounds: {generated_in_bounds} generated, {empty_in_bounds} empty")

        if empty_in_bounds == 0:
            print("\n✅ All quadrants in bounding box are already generated!")
            return 0

        # Show initial state
        if args.visualize:
            print("\n" + grid.visualize())

        # Create plan
        print("\n🔧 Creating generation plan...")

        # Create a copy of the grid for planning (original stays at current state)
        plan_grid = QuadrantGrid(bounds)
        for p in generated:
            if bounds.contains(p):
                plan_grid.mark_generated(p)

        steps = create_generation_plan(plan_grid)

        print(f"\n📋 Generation plan: {len(steps)} steps")

        # Show plan summary
        total_quadrants = sum(len(step.quadrants) for step in steps)
        by_size = {}
        for step in steps:
            size = len(step.quadrants)
            by_size[size] = by_size.get(size, 0) + 1

        print(f"   Total quadrants to generate: {total_quadrants}")
        print(f"   Steps by tile size:")
        for size in sorted(by_size.keys(), reverse=True):
            label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
            print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

        if args.dry_run:
            print("\n🔍 DRY RUN - Generation plan details:\n")

            # Reset grid for visualization
            viz_grid = QuadrantGrid(bounds)
            for p in generated:
                if bounds.contains(p):
                    viz_grid.mark_generated(p)

            for step in steps:
                coords = ", ".join(str(q) for q in step.quadrants)
                print(f"Step {step.step_number}: {step.description}")
                print(f"  Quadrants: [{coords}]")

                if args.visualize:
                    print(viz_grid.visualize(step.quadrants, step.step_number))
                    viz_grid.mark_multiple_generated(step.quadrants)
                    print()

            print("=" * 60)
            print("DRY RUN complete. Use without --dry-run to execute.")
            print("=" * 60)
            return 0

        # Execute plan
        print("\n🚀 Executing generation plan...")

        # Import here to avoid circular imports
        from isometric_nyc.e2e_generation.shared import (
            WEB_DIR,
            get_generation_config,
            get_quadrant_generation as shared_get_quadrant_generation,
            get_quadrant_render as shared_get_quadrant_render,
            image_to_png_bytes,
            png_bytes_to_image,
            save_quadrant_generation,
            start_web_server,
            upload_to_gcs,
        )

        web_server = None

        try:
            if not args.no_start_server:
                web_server = start_web_server(WEB_DIR, args.port)

            config = get_generation_config(conn)

            # Filter steps based on --start-from and --max-steps
            filtered_steps = [s for s in steps if s.step_number >= args.start_from]
            if args.max_steps is not None:
                filtered_steps = filtered_steps[: args.max_steps]

            if not filtered_steps:
                print("No steps to execute (check --start-from and --max-steps)")
                return 0

            print(f"   Executing steps {filtered_steps[0].step_number} to {filtered_steps[-1].step_number}")

            for step in filtered_steps:
                print(f"\n{'=' * 60}")
                print(f"Step {step.step_number}/{len(steps)}: {step.description}")
                print("=" * 60)

                # Convert Points to tuples for the generation API
                quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

                try:
                    result = run_generation_step(
                        conn,
                        config,
                        quadrant_tuples,
                        generation_dir,
                        args.port,
                        args.bucket,
                    )
                    if result.get("success"):
                        print(f"✅ Step {step.step_number} complete: {result.get('message')}")
                    else:
                        print(f"❌ Step {step.step_number} failed: {result.get('error')}")
                        # Continue with next step instead of raising
                except Exception as e:
                    print(f"❌ Step {step.step_number} failed: {e}")
                    # Continue with next step instead of raising

        finally:
            if web_server:
                print("\n🛑 Stopping web server...")
                web_server.terminate()
                web_server.wait()

        print(f"\n{'=' * 60}")
        print("✅ Generation complete!")
        print("=" * 60)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())


```

Let me run the help and verify all the new options are there:

The script is complete with all the features. Let me also test the visualization with a smaller bounding box to show the full output:

Excellent! The script is working perfectly. The visualization shows clearly how the algorithm:

1. Finds empty quadrants at the bottom
2. Generates them efficiently using 2x2 tiles (4 steps for 16 quadrants)
3. Finishes with a 2x1 vertical tile for the remaining 2 quadrants

Let me read the final file to ensure everything is correct:

The script is complete. Let me provide a summary of what was implemented:

## Summary

I've implemented the automatic generation script as described in the task file. Here's what it does:

### Key Features

1. **Dry-run mode** (`--dry-run`): Shows the complete generation plan without executing any API calls
2. **ASCII visualization** (`--visualize`): Shows a grid visualization of each step with `G` for generated, `S` for selected, and `.` for empty
3. **Flexible bounding box**: Specify any rectangular region to fill using `--top-left` and `--bottom-right`
4. **Resume capability**: Use `--start-from N` to start from step N (useful if a previous run was interrupted)
5. **Step limiting**: Use `--max-steps N` to only execute N steps (useful for testing)

### Algorithm

The script implements the optimal generation strategy from the task:

1. **Phase 1 - Fill interior gaps**: First fills any holes inside the current generated region using the largest possible tiles (2x2 > 1x2/2x1 > single)

2. **Phase 2 - Spiral expansion**: Expands outward from the center in a spiral pattern (top, right, bottom, left):
   - Generates 2x2 tiles offset from the current edge
   - Bridges the offset tiles back to the center
   - Fills remaining gaps with smaller tiles

3. **Tile size priority**: Always prefers larger tiles (more efficient):
   - 2x2 tiles (4 quadrants) - most efficient
   - 1x2 or 2x1 tiles (2 quadrants)
   - Single quadrants - last resort

### Usage Examples

```bash
# Dry run to see the plan
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="-10,-10" \
  --bottom-right="20,20" \
  --dry-run

# With visualization
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="0,0" \
  --bottom-right="10,10" \
  --dry-run --visualize

# Execute the plan
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="-5,-5" \
  --bottom-right="15,15"

# Resume from step 50
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="-5,-5" \
  --bottom-right="15,15" \
  --start-from=50
```

```shellscript
# Dry run to see the plan
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="-10,-10" \
  --bottom-right="20,20" \
  --dry-run

# With visualization
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="0,0" \
  --bottom-right="10,10" \
  --dry-run --visualize

# Execute the plan
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="-5,-5" \
  --bottom-right="15,15"

# Resume from step 50
uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \
  generations/test_generation \
  --top-left="-5,-5" \
  --bottom-right="15,15" \
  --start-from=50
```

