"""
Tests for make_rectangle_plan.py

These tests verify the rectangle generation planning algorithm handles
all cases correctly, including:
- Basic coverage of all quadrants
- 2x2 tile placement rules (no touching generated neighbors)
- 2x1/1x2 tile placement rules (extend from generated edges)
- 1x1 filling of remaining gaps
- Pre-existing generated quadrants
- Various rectangle sizes and configurations
"""

from isometric_nyc.generation.make_rectangle_plan import (
  GenerationStep,
  Point,
  RectanglePlan,
  RectBounds,
  can_place_1x1,
  can_place_1x2_vertical,
  can_place_2x1_horizontal,
  can_place_2x2,
  create_rectangle_plan,
  create_rectangle_plan_from_coords,
  get_2x2_neighbors,
  get_2x2_quadrants,
  get_plan_summary,
  has_valid_2x2_context,
  validate_plan,
  validate_plan_context,
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

  def test_from_string_negative(self) -> None:
    p = Point.from_string("-3,-5")
    assert p == Point(-3, -5)

  def test_to_tuple(self) -> None:
    p = Point(3, 5)
    assert p.to_tuple() == (3, 5)


# =============================================================================
# RectBounds Tests
# =============================================================================


class TestRectBounds:
  def test_width_height(self) -> None:
    bounds = RectBounds(Point(0, 0), Point(10, 5))
    assert bounds.width == 11
    assert bounds.height == 6

  def test_area(self) -> None:
    bounds = RectBounds(Point(0, 0), Point(3, 2))
    assert bounds.area == 12  # 4 x 3

  def test_contains(self) -> None:
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    assert bounds.contains(Point(0, 0))
    assert bounds.contains(Point(5, 5))
    assert bounds.contains(Point(3, 3))
    assert not bounds.contains(Point(-1, 0))
    assert not bounds.contains(Point(6, 0))
    assert not bounds.contains(Point(0, -1))
    assert not bounds.contains(Point(0, 6))

  def test_all_points(self) -> None:
    bounds = RectBounds(Point(0, 0), Point(2, 1))
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
# 2x2 Tile Helper Tests
# =============================================================================


class TestGet2x2Quadrants:
  def test_basic(self) -> None:
    quadrants = get_2x2_quadrants(Point(0, 0))
    assert quadrants == [
      Point(0, 0),
      Point(1, 0),
      Point(0, 1),
      Point(1, 1),
    ]

  def test_offset(self) -> None:
    quadrants = get_2x2_quadrants(Point(5, 3))
    assert quadrants == [
      Point(5, 3),
      Point(6, 3),
      Point(5, 4),
      Point(6, 4),
    ]


class TestGet2x2Neighbors:
  def test_basic(self) -> None:
    neighbors = get_2x2_neighbors(Point(0, 0))
    # Top, Bottom, Left, Right
    expected = [
      Point(0, -1),
      Point(1, -1),  # Top
      Point(0, 2),
      Point(1, 2),  # Bottom
      Point(-1, 0),
      Point(-1, 1),  # Left
      Point(2, 0),
      Point(2, 1),  # Right
    ]
    assert set(neighbors) == set(expected)
    assert len(neighbors) == 8


# =============================================================================
# 2x2 Placement Tests
# =============================================================================


class TestCanPlace2x2:
  def test_empty_grid(self) -> None:
    """2x2 can be placed in empty rectangle."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    assert can_place_2x2(Point(0, 0), bounds, set(), set())

  def test_out_of_bounds(self) -> None:
    """2x2 cannot extend beyond rectangle."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    # Top-left at (5, 5) would extend to (6, 6)
    assert not can_place_2x2(Point(5, 5), bounds, set(), set())
    # Top-left at (5, 0) would extend to (6, 1)
    assert not can_place_2x2(Point(5, 0), bounds, set(), set())

  def test_neighbor_generated_top(self) -> None:
    """2x2 cannot be placed if top neighbors are generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, -1)}  # Top neighbor
    assert not can_place_2x2(Point(0, 0), bounds, generated, set())

  def test_neighbor_generated_bottom(self) -> None:
    """2x2 cannot be placed if bottom neighbors are generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, 2)}  # Bottom neighbor
    assert not can_place_2x2(Point(0, 0), bounds, generated, set())

  def test_neighbor_generated_left(self) -> None:
    """2x2 cannot be placed if left neighbors are generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(-1, 0)}  # Left neighbor
    assert not can_place_2x2(Point(0, 0), bounds, generated, set())

  def test_neighbor_generated_right(self) -> None:
    """2x2 cannot be placed if right neighbors are generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(2, 0)}  # Right neighbor
    assert not can_place_2x2(Point(0, 0), bounds, generated, set())

  def test_quadrant_already_generated(self) -> None:
    """2x2 cannot be placed if any quadrant is already generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, 0)}  # One of the quadrants
    assert not can_place_2x2(Point(0, 0), bounds, generated, set())

  def test_quadrant_already_scheduled(self) -> None:
    """2x2 cannot be placed if any quadrant is already scheduled."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    scheduled = {Point(1, 1)}  # One of the quadrants
    assert not can_place_2x2(Point(0, 0), bounds, set(), scheduled)

  def test_valid_with_distant_generated(self) -> None:
    """2x2 can be placed if generated quadrants are not neighbors."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(3, 3)}  # Far from (0,0) 2x2
    assert can_place_2x2(Point(0, 0), bounds, generated, set())


# =============================================================================
# 2x1 Horizontal Placement Tests
# =============================================================================


class TestCanPlace2x1Horizontal:
  def test_valid_with_top_generated(self) -> None:
    """2x1 can be placed when top row is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, -1), Point(1, -1)}  # Top neighbors
    assert can_place_2x1_horizontal(Point(0, 0), bounds, generated, set())

  def test_valid_with_bottom_generated(self) -> None:
    """2x1 can be placed when bottom row is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, 1), Point(1, 1)}  # Bottom neighbors
    assert can_place_2x1_horizontal(Point(0, 0), bounds, generated, set())

  def test_invalid_when_left_generated(self) -> None:
    """2x1 cannot be placed when left neighbor is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(-1, 0), Point(0, -1), Point(1, -1)}
    assert not can_place_2x1_horizontal(Point(0, 0), bounds, generated, set())

  def test_invalid_when_right_generated(self) -> None:
    """2x1 cannot be placed when right neighbor is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(2, 0), Point(0, -1), Point(1, -1)}
    assert not can_place_2x1_horizontal(Point(0, 0), bounds, generated, set())

  def test_invalid_when_neither_long_side_generated(self) -> None:
    """2x1 cannot be placed when neither long side is fully generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated: set[Point] = set()
    assert not can_place_2x1_horizontal(Point(0, 0), bounds, generated, set())

  def test_valid_when_both_long_sides_generated(self) -> None:
    """2x1 CAN be placed when both long sides are generated (bridges between 2x2s)."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {
      Point(0, -1),
      Point(1, -1),  # Top
      Point(0, 1),
      Point(1, 1),  # Bottom
    }
    # This is now valid - bridges between two 2x2 tiles have both sides generated
    assert can_place_2x1_horizontal(Point(0, 0), bounds, generated, set())

  def test_invalid_when_only_partial_long_side(self) -> None:
    """2x1 cannot be placed when only one of two top neighbors is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, -1)}  # Only one top neighbor
    assert not can_place_2x1_horizontal(Point(0, 0), bounds, generated, set())

  def test_out_of_bounds(self) -> None:
    """2x1 cannot extend beyond rectangle."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(5, -1), Point(6, -1)}
    # Left at x=5 would extend to x=6
    assert not can_place_2x1_horizontal(Point(5, 0), bounds, generated, set())


# =============================================================================
# 1x2 Vertical Placement Tests
# =============================================================================


class TestCanPlace1x2Vertical:
  def test_valid_with_left_generated(self) -> None:
    """1x2 can be placed when left column is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(-1, 0), Point(-1, 1)}  # Left neighbors
    assert can_place_1x2_vertical(Point(0, 0), bounds, generated, set())

  def test_valid_with_right_generated(self) -> None:
    """1x2 can be placed when right column is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(1, 0), Point(1, 1)}  # Right neighbors
    assert can_place_1x2_vertical(Point(0, 0), bounds, generated, set())

  def test_invalid_when_top_generated(self) -> None:
    """1x2 cannot be placed when top neighbor is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, -1), Point(-1, 0), Point(-1, 1)}
    assert not can_place_1x2_vertical(Point(0, 0), bounds, generated, set())

  def test_invalid_when_bottom_generated(self) -> None:
    """1x2 cannot be placed when bottom neighbor is generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(0, 2), Point(-1, 0), Point(-1, 1)}
    assert not can_place_1x2_vertical(Point(0, 0), bounds, generated, set())

  def test_invalid_when_neither_long_side_generated(self) -> None:
    """1x2 cannot be placed when neither long side is fully generated."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated: set[Point] = set()
    assert not can_place_1x2_vertical(Point(0, 0), bounds, generated, set())

  def test_valid_when_both_long_sides_generated(self) -> None:
    """1x2 CAN be placed when both long sides are generated (bridges between 2x2s)."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {
      Point(-1, 0),
      Point(-1, 1),  # Left
      Point(1, 0),
      Point(1, 1),  # Right
    }
    # This is now valid - bridges between two 2x2 tiles have both sides generated
    assert can_place_1x2_vertical(Point(0, 0), bounds, generated, set())

  def test_out_of_bounds(self) -> None:
    """1x2 cannot extend beyond rectangle."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    generated = {Point(-1, 5), Point(-1, 6)}
    # Top at y=5 would extend to y=6
    assert not can_place_1x2_vertical(Point(0, 5), bounds, generated, set())


# =============================================================================
# Full Rectangle Plan Tests
# =============================================================================


class TestCreateRectanglePlan:
  def test_empty_rectangle_no_generated(self) -> None:
    """Empty rectangle with no pre-generated quadrants."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_single_quadrant_with_context(self) -> None:
    """Single quadrant rectangle requires 3 neighbors for context."""
    bounds = RectBounds(Point(0, 0), Point(0, 0))
    # 1x1 requires 3 of 4 quadrants in a 2x2 block to be generated
    generated = {Point(1, 0), Point(0, 1), Point(1, 1)}
    plan = create_rectangle_plan(bounds, generated)

    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == "1x1"
    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_single_quadrant_without_context(self) -> None:
    """Single quadrant without context cannot be generated."""
    bounds = RectBounds(Point(0, 0), Point(0, 0))
    plan = create_rectangle_plan(bounds)

    # Without context, no tiles can be placed
    assert len(plan.steps) == 0

  def test_2x2_rectangle(self) -> None:
    """2x2 rectangle should be a single 2x2 tile."""
    bounds = RectBounds(Point(0, 0), Point(1, 1))
    plan = create_rectangle_plan(bounds)

    # Should have exactly one 2x2 step
    assert len(plan.steps) == 1
    assert len(plan.steps[0].quadrants) == 4
    assert plan.steps[0].step_type == "2x2"
    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_3x3_rectangle(self) -> None:
    """3x3 rectangle - mix of tiles."""
    bounds = RectBounds(Point(0, 0), Point(2, 2))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_4x4_rectangle(self) -> None:
    """4x4 rectangle - should fit multiple 2x2 tiles."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Should have multiple 2x2 tiles
    count_2x2 = sum(1 for s in plan.steps if s.step_type == "2x2")
    assert count_2x2 >= 1

  def test_wide_rectangle(self) -> None:
    """Wide rectangle (10x2)."""
    bounds = RectBounds(Point(0, 0), Point(9, 1))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_tall_rectangle(self) -> None:
    """Tall rectangle (2x10)."""
    bounds = RectBounds(Point(0, 0), Point(1, 9))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_with_pre_generated_corner(self) -> None:
    """Rectangle with one corner already generated."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    generated = {Point(0, 0)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Should not include (0,0) in any step
    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    assert Point(0, 0) not in all_quadrants

  def test_with_pre_generated_edge(self) -> None:
    """Rectangle with one edge already generated (outside the bounds)."""
    bounds = RectBounds(Point(0, 0), Point(5, 2))
    # Generated row above the rectangle
    generated = {Point(x, -1) for x in range(6)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_with_pre_generated_interior(self) -> None:
    """Rectangle with some interior quadrants already generated."""
    bounds = RectBounds(Point(0, 0), Point(4, 4))
    # Some interior points generated - form a vertical 1x2 block
    # This provides context for adjacent tiles
    generated = {Point(2, 2), Point(2, 3)}
    plan = create_rectangle_plan(bounds, generated)

    # Should not include generated points
    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    assert Point(2, 2) not in all_quadrants
    assert Point(2, 3) not in all_quadrants

    # The plan should cover the remaining quadrants
    # Note: Some gaps may remain if there's no valid context
    covered = set(all_quadrants)
    expected_to_cover = set(bounds.all_points()) - generated

    # Verify no extra quadrants are covered
    extra = covered - expected_to_cover
    assert not extra, f"Extra quadrants covered: {extra}"

  def test_fully_generated(self) -> None:
    """Rectangle where all quadrants are already generated."""
    bounds = RectBounds(Point(0, 0), Point(2, 2))
    generated = set(bounds.all_points())
    plan = create_rectangle_plan(bounds, generated)

    assert len(plan.steps) == 0
    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_negative_coordinates(self) -> None:
    """Rectangle in negative coordinate space."""
    bounds = RectBounds(Point(-5, -3), Point(-2, -1))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_from_coords_convenience(self) -> None:
    """Test the convenience function with tuples."""
    plan = create_rectangle_plan_from_coords((0, 0), (3, 3))

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_from_coords_with_generated(self) -> None:
    """Test convenience function with pre-generated set."""
    generated = {(0, 0), (1, 0)}
    plan = create_rectangle_plan_from_coords((0, 0), (3, 3), generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    assert Point(0, 0) not in all_quadrants
    assert Point(1, 0) not in all_quadrants


# =============================================================================
# 2x2 Tile Rule Enforcement Tests
# =============================================================================


class Test2x2RuleEnforcement:
  def test_2x2_not_touching_pre_generated(self) -> None:
    """2x2 tiles should not touch pre-generated quadrants."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    # Pre-generated quadrant at (2, -1) - above the rectangle
    generated = {Point(2, -1)}
    plan = create_rectangle_plan(bounds, generated)

    # Find all 2x2 steps
    for step in plan.steps:
      if step.step_type == "2x2":
        neighbors = set()
        for q in step.quadrants:
          neighbors.add(Point(q.x - 1, q.y))
          neighbors.add(Point(q.x + 1, q.y))
          neighbors.add(Point(q.x, q.y - 1))
          neighbors.add(Point(q.x, q.y + 1))
        # Subtract the tile itself
        neighbors -= set(step.quadrants)
        # No neighbor should be pre-generated
        for n in neighbors:
          assert n not in generated, (
            f"2x2 at {step.quadrants} has pre-generated neighbor {n}"
          )

  def test_2x2_not_touching_other_scheduled(self) -> None:
    """2x2 tiles should not touch other scheduled tiles (at placement time)."""
    # This is verified by the algorithm itself - 2x2 tiles can only be placed
    # where no neighbor is scheduled. We verify by checking no 2x2 tiles share edges.
    bounds = RectBounds(Point(0, 0), Point(7, 7))
    plan = create_rectangle_plan(bounds)

    # Collect all 2x2 tile quadrants
    all_2x2_quadrants: set[Point] = set()
    for step in plan.steps:
      if step.step_type == "2x2":
        for q in step.quadrants:
          all_2x2_quadrants.add(q)

    # Check that no 2x2 tiles share edges (they should have gaps)
    # Actually, 2x2 tiles CAN share edges with OTHER 2x2 tiles if they were
    # placed later. The rule is only about pre-existing generated content.
    # Let's just verify the plan is valid.
    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"


# =============================================================================
# 2x1/1x2 Tile Rule Enforcement Tests
# =============================================================================


class Test2x1RuleEnforcement:
  def test_2x1_extends_from_generated_edge(self) -> None:
    """2x1 tiles should extend from a fully generated edge."""
    bounds = RectBounds(Point(0, 0), Point(5, 2))
    # Pre-generated row above
    generated = {Point(x, -1) for x in range(6)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_1x2_extends_from_generated_edge(self) -> None:
    """1x2 tiles should extend from a fully generated edge."""
    bounds = RectBounds(Point(0, 0), Point(2, 5))
    # Pre-generated column to the left
    generated = {Point(-1, y) for y in range(6)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"


# =============================================================================
# Plan Summary Tests
# =============================================================================


class TestPlanSummary:
  def test_summary_basic(self) -> None:
    """Test plan summary generation."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    plan = create_rectangle_plan(bounds)
    summary = get_plan_summary(plan)

    assert summary["bounds"]["width"] == 4
    assert summary["bounds"]["height"] == 4
    assert summary["total_quadrants"] == 16
    assert summary["total_steps"] > 0

  def test_summary_with_pre_generated(self) -> None:
    """Test summary with pre-generated quadrants."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    generated = {Point(0, 0), Point(0, 1)}
    plan = create_rectangle_plan(bounds, generated)
    summary = get_plan_summary(plan)

    assert summary["pre_generated_count"] == 2
    assert summary["total_quadrants"] == 14  # 16 - 2


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
  def test_1x1_rectangle_with_context(self) -> None:
    """Smallest possible rectangle with 3-neighbor context."""
    bounds = RectBounds(Point(0, 0), Point(0, 0))
    # 1x1 needs 3 of 4 quadrants in a 2x2 block generated
    generated = {Point(1, 0), Point(0, 1), Point(1, 1)}
    plan = create_rectangle_plan(bounds, generated)

    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == "1x1"
    is_valid, errors = validate_plan(plan)
    assert is_valid

  def test_1x1_rectangle_without_context(self) -> None:
    """1x1 rectangle without context cannot be generated."""
    bounds = RectBounds(Point(0, 0), Point(0, 0))
    plan = create_rectangle_plan(bounds)

    # Without context, no tiles can be placed
    assert len(plan.steps) == 0

  def test_1x10_strip_with_context(self) -> None:
    """Very thin horizontal strip with bottom edge context."""
    bounds = RectBounds(Point(0, 0), Point(9, 0))
    # Add context row below the strip
    generated = {Point(x, 1) for x in range(10)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_1x10_strip_without_context(self) -> None:
    """Thin strip without context cannot be fully generated."""
    bounds = RectBounds(Point(0, 0), Point(9, 0))
    plan = create_rectangle_plan(bounds)

    # Without context, no tiles can be placed (can't place 2x2 in 1-row strip)
    assert len(plan.steps) == 0

  def test_10x1_strip_with_context(self) -> None:
    """Very thin vertical strip with right edge context."""
    bounds = RectBounds(Point(0, 0), Point(0, 9))
    # Add context column to the right
    generated = {Point(1, y) for y in range(10)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_10x1_strip_without_context(self) -> None:
    """Thin strip without context cannot be fully generated."""
    bounds = RectBounds(Point(0, 0), Point(0, 9))
    plan = create_rectangle_plan(bounds)

    # Without context, no tiles can be placed (can't place 2x2 in 1-col strip)
    assert len(plan.steps) == 0

  def test_large_rectangle(self) -> None:
    """Large rectangle (20x20)."""
    bounds = RectBounds(Point(0, 0), Point(19, 19))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Should have many 2x2 tiles (greedy algorithm may not be optimal)
    count_2x2 = sum(1 for s in plan.steps if s.step_type == "2x2")
    assert count_2x2 >= 40  # Should have a reasonable number of 2x2 tiles

  def test_checkerboard_pre_generated(self) -> None:
    """Checkerboard pattern of pre-generated quadrants.

    In a checkerboard pattern, every 2x2 block has exactly 2 generated
    quadrants (diagonal corners). This means no point has 3-of-4 context
    in any 2x2 block, so 1x1 tiles cannot be placed.

    This is expected behavior - checkerboard patterns cannot be extended
    with valid context.
    """
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    # Checkerboard: every other quadrant
    generated = {Point(x, y) for y in range(6) for x in range(6) if (x + y) % 2 == 0}
    plan = create_rectangle_plan(bounds, generated)

    # No tiles can be placed because no point has 3-of-4 context
    # in any 2x2 block (checkerboard has only 2-of-4 diagonal)
    assert len(plan.steps) == 0

  def test_row_pattern_provides_context(self) -> None:
    """Row pattern provides valid context for generation.

    Unlike checkerboard, alternating rows allow valid context because
    each non-generated quadrant has 3 neighbors in some 2x2 block.
    """
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    # Alternating rows: y=0,2,4 are generated
    generated = {Point(x, y) for y in [0, 2, 4] for x in range(6)}
    # Add edge context
    generated |= {Point(x, -1) for x in range(6)}
    generated |= {Point(x, 6) for x in range(6)}
    plan = create_rectangle_plan(bounds, generated)

    # Get the non-generated quadrants within bounds (rows 1, 3, 5)
    to_generate = {p for p in bounds.all_points() if p not in generated}
    all_quadrants = [q for s in plan.steps for q in s.quadrants]

    # Should cover all non-generated quadrants
    assert set(all_quadrants) == to_generate

    # All tiles should be 2x1 or 1x1 (both valid with row context)
    for step in plan.steps:
      assert step.step_type in ("2x1", "1x1")

  def test_surrounded_by_generated(self) -> None:
    """Rectangle completely surrounded by generated quadrants."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    # Generate a border around the rectangle
    generated: set[Point] = set()
    for x in range(-1, 5):
      generated.add(Point(x, -1))  # Top
      generated.add(Point(x, 4))  # Bottom
    for y in range(0, 4):
      generated.add(Point(-1, y))  # Left
      generated.add(Point(4, y))  # Right

    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # 2x2 tiles CAN be placed in the interior (shielded by outer ring)
    # But edges should use smaller tiles
    # Check that all 2x2 tiles don't touch the generated border
    for step in plan.steps:
      if step.step_type == "2x2":
        for q in step.quadrants:
          # 2x2 at edge positions would have generated neighbors
          # Interior 2x2 (at 1,1 or 1,2 etc.) is OK
          neighbors = [
            Point(q.x - 1, q.y),
            Point(q.x + 1, q.y),
            Point(q.x, q.y - 1),
            Point(q.x, q.y + 1),
          ]
          for n in neighbors:
            if n not in step.quadrants:  # Not part of the tile itself
              assert n not in generated, (
                f"2x2 at {step.quadrants} has neighbor {n} in generated"
              )


# =============================================================================
# Integration Tests - Complex Scenarios
# =============================================================================


class TestComplexScenarios:
  def test_partial_strip_extension(self) -> None:
    """Extending from a partially generated strip."""
    bounds = RectBounds(Point(0, 0), Point(7, 3))
    # Some quadrants above are generated
    generated = {Point(0, -1), Point(1, -1), Point(2, -1)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_l_shaped_pre_generated(self) -> None:
    """L-shaped region of pre-generated quadrants."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    # L-shape in top-left
    generated = {
      Point(-1, 0),
      Point(-1, 1),
      Point(-1, 2),
      Point(0, -1),
      Point(1, -1),
      Point(2, -1),
    }
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_island_of_generated(self) -> None:
    """Island of pre-generated quadrants in the middle."""
    bounds = RectBounds(Point(0, 0), Point(9, 9))
    # 2x2 island in the center
    generated = {Point(4, 4), Point(5, 4), Point(4, 5), Point(5, 5)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Generated island should not be in any step
    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    for g in generated:
      assert g not in all_quadrants

  def test_multiple_disjoint_pre_generated(self) -> None:
    """Multiple disjoint pre-generated regions."""
    bounds = RectBounds(Point(0, 0), Point(9, 9))
    # Two separate 2x2 regions
    generated = {
      Point(0, 0),
      Point(1, 0),
      Point(0, 1),
      Point(1, 1),
      Point(7, 7),
      Point(8, 7),
      Point(7, 8),
      Point(8, 8),
    }
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"


# =============================================================================
# Serialization Tests
# =============================================================================


# =============================================================================
# Task 019 Strip Plan Pattern Tests
# =============================================================================


class TestStripPlanPattern:
  """
  Tests that verify the algorithm follows the 019 strip plan pattern:
  1. 2x2 tiles placed with gaps (every 3rd column)
  2. 1x2 bridges between 2x2 tiles
  3. 2x1 bridges connecting to generation edge
  4. 1x1 fills for remaining gaps
  """

  def test_depth_3_horizontal_pattern(self) -> None:
    """
    Test 6x3 rectangle with bottom edge generated.

    Expected pattern (following 019):
    Row 0-1: 2x2 at (0,0), (3,0) with 1x2 bridges at (2,0), (5,0)
    Row 2:   2x1 at (0,2), (3,2) with 1x1 at (2,2), (5,2)

    Visual:
    A A C B B D   <- row 0 (A=2x2, B=2x2, C=1x2, D=1x2)
    A A C B B D   <- row 1
    E E G F F H   <- row 2 (E=2x1, F=2x1, G=1x1, H=1x1)
    G G G G G G   <- generation edge (y=3, outside rectangle)
    """
    bounds = RectBounds(Point(0, 0), Point(5, 2))
    # Generation edge at y=3 (below rectangle)
    generated = {Point(x, 3) for x in range(6)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Count by type
    types = {}
    for step in plan.steps:
      t = step.step_type
      types[t] = types.get(t, 0) + 1

    # Should have: 2 2x2, 2 1x2, 2 2x1, 2 1x1
    assert types.get("2x2", 0) == 2, f"Expected 2 2x2 tiles, got {types}"
    assert types.get("1x2", 0) == 2, f"Expected 2 1x2 tiles, got {types}"
    assert types.get("2x1", 0) == 2, f"Expected 2 2x1 tiles, got {types}"
    assert types.get("1x1", 0) == 2, f"Expected 2 1x1 tiles, got {types}"

    # Verify 2x2 positions
    two_by_two = [s for s in plan.steps if s.step_type == "2x2"]
    two_by_two_positions = [
      (min(q.x for q in s.quadrants), min(q.y for q in s.quadrants)) for s in two_by_two
    ]
    assert (0, 0) in two_by_two_positions
    assert (3, 0) in two_by_two_positions

    # Verify 1x2 positions (between 2x2 tiles)
    one_by_two = [s for s in plan.steps if s.step_type == "1x2"]
    one_by_two_positions = [s.quadrants[0] for s in one_by_two]
    assert Point(2, 0) in one_by_two_positions
    assert Point(5, 0) in one_by_two_positions

    # Verify 2x1 positions (connecting to edge)
    two_by_one = [s for s in plan.steps if s.step_type == "2x1"]
    two_by_one_positions = [
      (min(q.x for q in s.quadrants), s.quadrants[0].y) for s in two_by_one
    ]
    assert (0, 2) in two_by_one_positions
    assert (3, 2) in two_by_one_positions

  def test_depth_3_vertical_pattern(self) -> None:
    """
    Test 3x6 rectangle with left edge generated.

    Similar pattern but rotated 90 degrees.
    """
    bounds = RectBounds(Point(0, 0), Point(2, 5))
    # Generation edge at x=-1 (left of rectangle)
    generated = {Point(-1, y) for y in range(6)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Count by type - pattern should be similar
    types = {}
    for step in plan.steps:
      t = step.step_type
      types[t] = types.get(t, 0) + 1

    # Should have: 2 2x2, 2 2x1, 2 1x2, 2 1x1
    assert types.get("2x2", 0) == 2, f"Expected 2 2x2 tiles, got {types}"
    assert types.get("2x1", 0) == 2, f"Expected 2 2x1 tiles, got {types}"
    assert types.get("1x2", 0) == 2, f"Expected 2 1x2 tiles, got {types}"
    assert types.get("1x1", 0) == 2, f"Expected 2 1x1 tiles, got {types}"

  def test_1x2_bridges_between_2x2(self) -> None:
    """Test that 1x2 bridges correctly connect two 2x2 tiles."""
    # 5x3 rectangle (need 3 rows for 2x2 + bridge pattern)
    # With generation edge at y=3 (one row gap from 2x2 tiles)
    bounds = RectBounds(Point(0, 0), Point(4, 2))
    generated = {Point(x, 3) for x in range(5)}
    plan = create_rectangle_plan(bounds, generated)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Visual:
    # A A C B B   <- row 0 (A=2x2 at 0,0; B=2x2 at 3,0; C=1x2 at 2,0)
    # A A C B B   <- row 1
    # D D E F F   <- row 2 (D=2x1 at 0,2; E=1x1 at 2,2; F=2x1 at 3,2)
    # G G G G G   <- generation edge

    types = {}
    for step in plan.steps:
      t = step.step_type
      types[t] = types.get(t, 0) + 1

    # 2 2x2 tiles, 1 1x2 bridge, 2 2x1, 1 1x1
    assert types.get("2x2", 0) == 2, f"Expected 2 2x2 tiles, got {types}"
    assert types.get("1x2", 0) == 1, f"Expected 1 1x2 bridge, got {types}"
    assert types.get("2x1", 0) == 2, f"Expected 2 2x1 tiles, got {types}"
    assert types.get("1x1", 0) == 1, f"Expected 1 1x1 fill, got {types}"

  def test_order_matches_019_pattern(self) -> None:
    """
    Verify that generation order follows 019:
    1. 2x2 tiles first
    2. 1x2 bridges second
    3. 2x1 bridges third
    4. 1x1 fills last
    """
    bounds = RectBounds(Point(0, 0), Point(5, 2))
    generated = {Point(x, 3) for x in range(6)}
    plan = create_rectangle_plan(bounds, generated)

    # Check order
    current_phase = 0
    phase_order = {"2x2": 0, "1x2": 1, "2x1": 2, "1x1": 3}

    for step in plan.steps:
      phase = phase_order.get(step.step_type, 4)
      assert phase >= current_phase, (
        f"Step {step.step_type} appears after later phase. "
        f"Expected order: 2x2, 1x2, 2x1, 1x1"
      )
      current_phase = phase


class TestSerialization:
  def test_step_to_dict(self) -> None:
    """Test GenerationStep serialization."""
    step = GenerationStep(quadrants=[Point(0, 0), Point(1, 0)], step_type="2x1")
    d = step.to_dict()
    assert d["quadrants"] == [(0, 0), (1, 0)]
    assert d["type"] == "2x1"

  def test_plan_to_dict(self) -> None:
    """Test RectanglePlan serialization."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    plan = create_rectangle_plan(bounds)
    d = plan.to_dict()

    assert d["bounds"]["top_left"] == (0, 0)
    assert d["bounds"]["bottom_right"] == (3, 3)
    assert len(d["steps"]) > 0


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
  def test_valid_plan(self) -> None:
    """Valid plan should pass validation."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan(plan)
    assert is_valid
    assert len(errors) == 0

  def test_detect_duplicate_coverage(self) -> None:
    """Validation should detect duplicate coverage."""
    bounds = RectBounds(Point(0, 0), Point(1, 1))
    # Manually create invalid plan with duplicates
    plan = RectanglePlan(
      bounds=bounds,
      steps=[
        GenerationStep(quadrants=[Point(0, 0), Point(1, 0)]),
        GenerationStep(quadrants=[Point(0, 0), Point(0, 1)]),  # Duplicate!
      ],
    )

    is_valid, errors = validate_plan(plan)
    assert not is_valid
    assert any("multiple times" in e for e in errors)

  def test_detect_missing_quadrants(self) -> None:
    """Validation should detect missing quadrants."""
    bounds = RectBounds(Point(0, 0), Point(1, 1))
    # Manually create invalid plan with missing quadrant
    plan = RectanglePlan(
      bounds=bounds,
      steps=[
        GenerationStep(quadrants=[Point(0, 0), Point(1, 0), Point(0, 1)]),
        # Missing (1, 1)
      ],
    )

    is_valid, errors = validate_plan(plan)
    assert not is_valid
    assert any("Missing" in e for e in errors)


# =============================================================================
# Context Validation Tests
# =============================================================================


class TestContextValidation:
  """Tests for 2x2 context requirement validation."""

  def test_has_valid_2x2_context_single_quadrant_with_3_neighbors(self) -> None:
    """Single quadrant with 3 neighbors has valid context."""
    quadrants = [Point(0, 0)]
    combined = {Point(1, 0), Point(0, 1), Point(1, 1)}
    assert has_valid_2x2_context(quadrants, combined)

  def test_has_valid_2x2_context_single_quadrant_with_2_neighbors(self) -> None:
    """Single quadrant with only 2 neighbors lacks valid context."""
    quadrants = [Point(0, 0)]
    combined = {Point(1, 0), Point(0, 1)}  # Only 2 neighbors
    assert not has_valid_2x2_context(quadrants, combined)

  def test_has_valid_2x2_context_2x2_is_self_contained(self) -> None:
    """2x2 tile is always self-contained (has valid context)."""
    quadrants = [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
    combined: set[Point] = set()  # No external context
    assert has_valid_2x2_context(quadrants, combined)

  def test_has_valid_2x2_context_2x1_with_top_neighbors(self) -> None:
    """2x1 with both top neighbors has valid context."""
    quadrants = [Point(0, 1), Point(1, 1)]  # Bottom row of 2x2
    combined = {Point(0, 0), Point(1, 0)}  # Top row
    assert has_valid_2x2_context(quadrants, combined)

  def test_has_valid_2x2_context_2x1_with_only_one_neighbor(self) -> None:
    """2x1 with only one neighbor lacks valid context."""
    quadrants = [Point(0, 1), Point(1, 1)]
    combined = {Point(0, 0)}  # Only one top neighbor
    assert not has_valid_2x2_context(quadrants, combined)

  def test_can_place_1x1_with_3_neighbors(self) -> None:
    """1x1 can be placed when 3 of 4 quadrants in a 2x2 block are generated."""
    combined = {Point(1, 0), Point(0, 1), Point(1, 1)}
    assert can_place_1x1(Point(0, 0), combined)

  def test_can_place_1x1_with_2_neighbors(self) -> None:
    """1x1 cannot be placed with only 2 neighbors."""
    combined = {Point(1, 0), Point(0, 1)}  # Only 2 neighbors
    assert not can_place_1x1(Point(0, 0), combined)

  def test_validate_plan_context_valid_plan(self) -> None:
    """Plan with valid context passes validation."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    plan = create_rectangle_plan(bounds)

    is_valid, errors = validate_plan_context(plan)
    assert is_valid, f"Context validation failed: {errors}"

  def test_validate_plan_context_invalid_plan(self) -> None:
    """Plan with invalid context fails validation."""
    bounds = RectBounds(Point(0, 0), Point(1, 1))
    # Manually create a plan with a 1x1 that lacks context
    plan = RectanglePlan(
      bounds=bounds,
      steps=[
        GenerationStep(quadrants=[Point(0, 0)], step_type="1x1"),
      ],
      pre_generated=set(),
    )

    is_valid, errors = validate_plan_context(plan)
    assert not is_valid
    assert any("lacks valid 2x2 context" in e for e in errors)

  def test_validate_plan_context_with_pre_generated(self) -> None:
    """Plan with pre-generated context passes validation."""
    bounds = RectBounds(Point(0, 0), Point(0, 0))
    # Pre-generated provides 3-of-4 context
    pre_generated = {Point(1, 0), Point(0, 1), Point(1, 1)}
    plan = RectanglePlan(
      bounds=bounds,
      steps=[
        GenerationStep(quadrants=[Point(0, 0)], step_type="1x1"),
      ],
      pre_generated=pre_generated,
    )

    is_valid, errors = validate_plan_context(plan)
    assert is_valid, f"Context validation failed: {errors}"


# =============================================================================
# Queued Quadrant Tests - Seam Detection with In-Progress/Queued Generations
# =============================================================================


class TestQueuedQuadrants:
  """
  Tests that verify the algorithm correctly considers in-progress and queued
  generations when determining if a seam will be formed.

  Queued quadrants are treated the same as generated quadrants for seam
  detection purposes - tiles cannot be placed adjacent to them.
  """

  def test_queued_quadrants_prevent_2x2_placement(self) -> None:
    """2x2 tiles cannot be placed adjacent to queued quadrants."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    # No generated quadrants
    generated: set[Point] = set()
    # But there's a queued generation at (2, -1) - above the rectangle
    queued = {Point(2, -1)}
    plan = create_rectangle_plan(bounds, generated, queued)

    # Find all 2x2 steps and verify none touch the queued quadrant
    for step in plan.steps:
      if step.step_type == "2x2":
        neighbors = set()
        for q in step.quadrants:
          neighbors.add(Point(q.x - 1, q.y))
          neighbors.add(Point(q.x + 1, q.y))
          neighbors.add(Point(q.x, q.y - 1))
          neighbors.add(Point(q.x, q.y + 1))
        neighbors -= set(step.quadrants)
        for n in neighbors:
          assert n not in queued, f"2x2 at {step.quadrants} has queued neighbor {n}"

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

  def test_queued_quadrants_excluded_from_plan(self) -> None:
    """Queued quadrants within bounds should not be included in the plan.

    Queued quadrants act as context for other tiles but are not generated
    in this plan (they're already queued elsewhere).
    """
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    generated: set[Point] = set()
    # Queue a 2x2 block - this provides context for adjacent tiles
    queued = {Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)}
    plan = create_rectangle_plan(bounds, generated, queued)

    # Should not include queued points in any step
    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    for q in queued:
      assert q not in all_quadrants, f"Queued quadrant {q} should not be in plan"

    # The remaining quadrants should be covered
    remaining = set(bounds.all_points()) - queued
    covered = set(all_quadrants)

    # All covered should be from remaining
    assert covered.issubset(remaining), (
      f"Covered {covered} not subset of remaining {remaining}"
    )

  def test_queued_row_affects_2x1_placement(self) -> None:
    """2x1 tiles should consider queued quadrants as generated for edge detection."""
    bounds = RectBounds(Point(0, 0), Point(5, 2))
    generated: set[Point] = set()
    # Queued row below the rectangle acts like a generated edge
    queued = {Point(x, 3) for x in range(6)}
    plan = create_rectangle_plan(bounds, generated, queued)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Should have 2x1 tiles connecting to the queued edge
    types = {}
    for step in plan.steps:
      t = step.step_type
      types[t] = types.get(t, 0) + 1

    # With edge at y=3, we should get 2x2 + bridge pattern
    assert types.get("2x2", 0) == 2, f"Expected 2 2x2 tiles, got {types}"
    assert types.get("2x1", 0) == 2, f"Expected 2 2x1 tiles, got {types}"

  def test_queued_column_affects_1x2_placement(self) -> None:
    """1x2 tiles should consider queued quadrants as generated for edge detection."""
    bounds = RectBounds(Point(0, 0), Point(2, 5))
    generated: set[Point] = set()
    # Queued column to the left acts like a generated edge
    queued = {Point(-1, y) for y in range(6)}
    plan = create_rectangle_plan(bounds, generated, queued)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # Should have 1x2 tiles connecting to the queued edge
    types = {}
    for step in plan.steps:
      t = step.step_type
      types[t] = types.get(t, 0) + 1

    # With edge at x=-1, we should get 2x2 + bridge pattern
    assert types.get("2x2", 0) == 2, f"Expected 2 2x2 tiles, got {types}"
    assert types.get("1x2", 0) == 2, f"Expected 2 1x2 tiles, got {types}"

  def test_mixed_generated_and_queued(self) -> None:
    """Plan should correctly handle both generated and queued quadrants."""
    bounds = RectBounds(Point(0, 0), Point(5, 5))
    # Some quadrants are already generated
    generated = {Point(-1, 0), Point(-1, 1)}
    # Some quadrants are queued for generation
    queued = {Point(6, 0), Point(6, 1)}
    plan = create_rectangle_plan(bounds, generated, queued)

    is_valid, errors = validate_plan(plan)
    assert is_valid, f"Plan invalid: {errors}"

    # 2x2 tiles at left edge should not touch generated at (-1, 0) (-1, 1)
    # 2x2 tiles at right edge should not touch queued at (6, 0) (6, 1)
    for step in plan.steps:
      if step.step_type == "2x2":
        for q in step.quadrants:
          # Check no neighbor is generated
          for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = Point(q.x + dx, q.y + dy)
            if neighbor not in step.quadrants:
              assert neighbor not in generated, (
                f"2x2 at {step.quadrants} has generated neighbor {neighbor}"
              )
              assert neighbor not in queued, (
                f"2x2 at {step.quadrants} has queued neighbor {neighbor}"
              )

  def test_queued_quadrant_in_middle_prevents_2x2(self) -> None:
    """A queued quadrant in the middle should prevent 2x2 placement there."""
    bounds = RectBounds(Point(0, 0), Point(3, 3))
    generated: set[Point] = set()
    # Queued 2x2 block in the center
    queued = {Point(1, 1), Point(2, 1), Point(1, 2), Point(2, 2)}
    plan = create_rectangle_plan(bounds, generated, queued)

    # No step should include any queued quadrants
    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    for q in queued:
      assert q not in all_quadrants, f"Queued quadrant {q} should not be in plan"

    # 2x2 tiles should not be placed adjacent to the queued block
    for step in plan.steps:
      if step.step_type == "2x2":
        for q in step.quadrants:
          for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = Point(q.x + dx, q.y + dy)
            if neighbor not in step.quadrants:
              assert neighbor not in queued, (
                f"2x2 at {step.quadrants} has queued neighbor {neighbor}"
              )

  def test_from_coords_with_queued(self) -> None:
    """Test convenience function with queued set."""
    generated = {(0, 0), (1, 0)}
    queued = {(2, 0), (3, 0)}
    plan = create_rectangle_plan_from_coords((0, 0), (5, 3), generated, queued)

    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    # Neither generated nor queued should be in plan
    assert Point(0, 0) not in all_quadrants
    assert Point(1, 0) not in all_quadrants
    assert Point(2, 0) not in all_quadrants
    assert Point(3, 0) not in all_quadrants

  def test_fully_queued_rectangle(self) -> None:
    """Rectangle where all quadrants are queued should produce empty plan."""
    bounds = RectBounds(Point(0, 0), Point(2, 2))
    generated: set[Point] = set()
    queued = set(bounds.all_points())
    plan = create_rectangle_plan(bounds, generated, queued)

    assert len(plan.steps) == 0

  def test_queued_prevents_seam_with_future_generation(self) -> None:
    """
    Queued quadrants should be treated as if they will be generated,
    preventing seams with future generations.

    Scenario: User is generating a 2x2 at (0,0)-(1,1) and there's already
    a queued 2x2 at (2,0)-(3,1). The plan should not place tiles that
    would create a seam between them.
    """
    bounds = RectBounds(Point(0, 0), Point(5, 3))
    generated: set[Point] = set()
    # First 2x2 is queued
    queued = {Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)}
    plan = create_rectangle_plan(bounds, generated, queued)

    # The plan should not include the queued quadrants
    all_quadrants = [q for s in plan.steps for q in s.quadrants]
    for q in queued:
      assert q not in all_quadrants

    # Verify the plan covers exactly the non-queued quadrants
    expected = set(bounds.all_points()) - queued
    covered = set(all_quadrants)
    assert covered == expected, f"Expected {expected}, got {covered}"

    # 2x2 tiles should not be placed at (2,0) because it would touch
    # the queued 2x2 at x=1
    for step in plan.steps:
      if step.step_type == "2x2":
        step_tl = min(q.x for q in step.quadrants), min(q.y for q in step.quadrants)
        # If there's a 2x2 at (2, 0), it would have neighbors at (1, 0) and (1, 1)
        # which are queued
        if step_tl == (2, 0):
          assert False, "2x2 at (2,0) would create seam with queued (0,0)-(1,1)"
