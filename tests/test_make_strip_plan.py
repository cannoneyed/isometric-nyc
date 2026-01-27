"""
Tests for make_strip_plan.py

These tests verify the strip generation planning algorithms without
needing a database connection.
"""

import pytest

from isometric_nyc.generation.make_strip_plan import (
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
      Point(-1, 0),
      Point(-1, 1),
      Point(-1, 2),
      Point(8, 0),
      Point(8, 1),
      Point(8, 2),
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
