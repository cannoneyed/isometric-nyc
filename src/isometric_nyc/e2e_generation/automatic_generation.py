"""
Automatic generation script for expanding tile coverage.

This script automatically generates tiles in an optimal order to expand
coverage from an existing generated region outward to fill a bounding box.

The algorithm:
1. Find the current generated "shape" (rectangle with possible holes)
2. Fill in any missing interior quadrants to create a solid rectangle
3. Expand outward in a spiral pattern (top, right, bottom, left)
4. Use an efficient generation pattern:
   - Generate four-quadrant tiles OFFSET from center (with a gap to avoid seams)
   - Bridge offset tiles to the center rectangle
   - Bridge offset tiles to each other
   - Fill remaining single-quadrant gaps

Usage:
  # Generate a plan (does not execute):
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --top-left <x>,<y> \\
    --bottom-right <x>,<y>

  # Execute an existing plan:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    <generation_dir> \\
    --plan-json <path_to_plan.json>

Example:
  # Create plan:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --top-left -10,-10 \\
    --bottom-right 20,20

  # Execute plan:
  uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\
    generations/test_generation \\
    --plan-json generations/test_generation/automatic_plan.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Data Structures
# =============================================================================


class QuadrantState(Enum):
  """State of a quadrant in the grid."""

  EMPTY = "empty"  # Not generated
  GENERATED = "generated"  # Already has generation
  SELECTED = "selected"  # Selected for generation in current step


class StepStatus(Enum):
  """Status of a generation step."""

  PENDING = "pending"
  GENERATED = "generated"
  ERROR = "error"


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
  status: StepStatus = StepStatus.PENDING
  error_message: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    return {
      "step_number": self.step_number,
      "quadrants": [{"x": q.x, "y": q.y} for q in self.quadrants],
      "description": self.description,
      "status": self.status.value,
      "error_message": self.error_message,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> GenerationStep:
    """Create from JSON dict."""
    return cls(
      step_number=data["step_number"],
      quadrants=[Point(q["x"], q["y"]) for q in data["quadrants"]],
      description=data["description"],
      status=StepStatus(data.get("status", "pending")),
      error_message=data.get("error_message"),
    )


@dataclass
class GenerationPlan:
  """A complete generation plan with metadata."""

  created_at: str
  bounds: BoundingBox
  steps: list[GenerationStep]
  generation_dir: str

  def to_dict(self) -> dict[str, Any]:
    """Convert to JSON-serializable dict."""
    return {
      "created_at": self.created_at,
      "bounds": {
        "top_left": {"x": self.bounds.top_left.x, "y": self.bounds.top_left.y},
        "bottom_right": {
          "x": self.bounds.bottom_right.x,
          "y": self.bounds.bottom_right.y,
        },
      },
      "generation_dir": self.generation_dir,
      "steps": [step.to_dict() for step in self.steps],
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> GenerationPlan:
    """Create from JSON dict."""
    bounds = BoundingBox(
      top_left=Point(data["bounds"]["top_left"]["x"], data["bounds"]["top_left"]["y"]),
      bottom_right=Point(
        data["bounds"]["bottom_right"]["x"], data["bounds"]["bottom_right"]["y"]
      ),
    )
    return cls(
      created_at=data["created_at"],
      bounds=bounds,
      generation_dir=data["generation_dir"],
      steps=[GenerationStep.from_dict(s) for s in data["steps"]],
    )

  def save(self, path: Path) -> None:
    """Save plan to JSON file."""
    with open(path, "w") as f:
      json.dump(self.to_dict(), f, indent=2)
    print(f"📄 Saved plan to {path}")

  @classmethod
  def load(cls, path: Path) -> GenerationPlan:
    """Load plan from JSON file."""
    with open(path) as f:
      data = json.load(f)
    return cls.from_dict(data)

  def update_step_status(
    self, step_number: int, status: StepStatus, error_message: str | None = None
  ) -> None:
    """Update the status of a specific step."""
    for step in self.steps:
      if step.step_number == step_number:
        step.status = status
        step.error_message = error_message
        break

  def get_pending_steps(self) -> list[GenerationStep]:
    """Get all steps that are still pending."""
    return [s for s in self.steps if s.status == StepStatus.PENDING]

  def get_summary(self) -> dict[str, int]:
    """Get count of steps by status."""
    summary: dict[str, int] = {"pending": 0, "generated": 0, "error": 0}
    for step in self.steps:
      summary[step.status.value] += 1
    return summary


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
  )
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_render as shared_get_quadrant_render,
  )
  from isometric_nyc.e2e_generation.shared import (
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
  from isometric_nyc.e2e_generation.shared import (
    build_tile_render_url,
    ensure_quadrant_exists,
    image_to_png_bytes,
    render_url_to_bytes,
    save_quadrant_render,
    split_tile_into_quadrants,
  )

  # Ensure the quadrant exists in the database
  ensure_quadrant_exists(conn, config, qx, qy)

  # Find the top-left of the tile containing this quadrant
  # Quadrants come in 2x2 tiles, so we need to find which tile this is in
  tile_x = (qx // 2) * 2
  tile_y = (qy // 2) * 2

  # Get the top-left quadrant of the tile
  tl_quadrant = ensure_quadrant_exists(conn, config, tile_x, tile_y)

  # Build URL and render using shared utilities
  url = build_tile_render_url(
    port=port,
    lat=tl_quadrant["lat"],
    lng=tl_quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    azimuth=config["camera_azimuth_degrees"],
    elevation=config["camera_elevation_degrees"],
    view_height=config.get("view_height_meters", 200),
  )

  screenshot = render_url_to_bytes(url, config["width_px"], config["height_px"])

  # Convert to PIL Image
  from io import BytesIO

  from PIL import Image

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


def download_image_to_pil(
  url: str,
  max_retries: int = 3,
  retry_delay: float = 10.0,
):
  """
  Download an image from a URL and return as PIL Image.

  Includes retry logic for transient errors (e.g., 403 Forbidden when
  the image is not yet available).

  Args:
      url: URL of the image to download
      max_retries: Maximum number of retry attempts (default: 3)
      retry_delay: Seconds to wait between retries (default: 10.0)

  Returns:
      PIL Image object

  Raises:
      requests.HTTPError: If all retry attempts fail
  """
  import time
  from io import BytesIO

  import requests
  from PIL import Image

  last_error = None

  for attempt in range(1, max_retries + 1):
    try:
      response = requests.get(url, timeout=120)
      response.raise_for_status()
      return Image.open(BytesIO(response.content))
    except requests.exceptions.HTTPError as e:
      last_error = e
      if attempt < max_retries:
        print(f"   ⚠️  Download failed (attempt {attempt}/{max_retries}): {e}")
        print(f"   ⏳ Waiting {retry_delay}s before retrying...")
        time.sleep(retry_delay)
      else:
        print(f"   ❌ Download failed after {max_retries} attempts: {e}")
    except requests.exceptions.RequestException as e:
      last_error = e
      if attempt < max_retries:
        print(f"   ⚠️  Download error (attempt {attempt}/{max_retries}): {e}")
        print(f"   ⏳ Waiting {retry_delay}s before retrying...")
        time.sleep(retry_delay)
      else:
        print(f"   ❌ Download failed after {max_retries} attempts: {e}")

  # If we get here, all retries failed
  if last_error:
    raise last_error
  raise RuntimeError("Download failed with no error captured")


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


def can_generate_2x2(
  grid: QuadrantGrid, top_left: Point, require_gap: bool = True
) -> bool:
  """
  Check if a 2x2 tile starting at top_left can be generated.

  A 2x2 tile can be generated if:
  - All 4 quadrants are empty (not already generated)
  - All 4 quadrants are within bounds
  - If require_gap is True (default): NO quadrant has a direct neighbor that's generated
    (to avoid seams), but there is generated content within 2 tiles
  - If require_gap is False: At least one quadrant has a generated neighbor

  The gap requirement prevents seams at tile boundaries. 2x2 tiles with gaps are
  bridged back using 1x2/2x1 tiles which handle seams better.
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

  quadrant_set = set(quadrants)

  if require_gap:
    # Check that NO quadrant has a direct generated neighbor (to avoid seams)
    for q in quadrants:
      direct_neighbors = [
        Point(q.x - 1, q.y),
        Point(q.x + 1, q.y),
        Point(q.x, q.y - 1),
        Point(q.x, q.y + 1),
      ]
      for n in direct_neighbors:
        if n not in quadrant_set and grid.is_generated(n):
          # Has direct neighbor - would create seam
          return False

    # Check that there IS generated content within 2 tiles (so we can bridge later)
    for q in quadrants:
      # Check neighbors at distance 2
      two_away = [
        Point(q.x - 2, q.y),
        Point(q.x + 2, q.y),
        Point(q.x, q.y - 2),
        Point(q.x, q.y + 2),
        Point(q.x - 1, q.y - 1),
        Point(q.x + 1, q.y - 1),
        Point(q.x - 1, q.y + 1),
        Point(q.x + 1, q.y + 1),
      ]
      for n in two_away:
        if n not in quadrant_set and grid.is_generated(n):
          return True
    return False
  else:
    # Original behavior: at least one must have a generated neighbor (outside the 2x2)
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
  """
  Check if a 1x2 horizontal tile can be generated.

  Layout: [left][right]

  IMPORTANT: Generated neighbors can ONLY be on the LONG sides (top/bottom),
  NOT on the SHORT sides (left end of left, right end of right).
  This prevents seams since short-side pixels aren't included in the template.
  """
  right = Point(left.x + 1, left.y)

  if grid.is_generated(left) or grid.is_generated(right):
    return False
  if not grid.bounds.contains(left) or not grid.bounds.contains(right):
    return False

  # SHORT sides (ends) - these CANNOT have generated neighbors
  short_side_neighbors = [
    Point(left.x - 1, left.y),  # Left of left quadrant
    Point(right.x + 1, right.y),  # Right of right quadrant
  ]
  for n in short_side_neighbors:
    if grid.is_generated(n):
      return False  # Would create seam

  # LONG sides (top/bottom) - at least one must have generated neighbor
  long_side_neighbors = [
    Point(left.x, left.y - 1),  # Above left
    Point(left.x, left.y + 1),  # Below left
    Point(right.x, right.y - 1),  # Above right
    Point(right.x, right.y + 1),  # Below right
  ]
  for n in long_side_neighbors:
    if grid.is_generated(n):
      return True

  return False


def can_generate_2x1_vertical(grid: QuadrantGrid, top: Point) -> bool:
  """
  Check if a 2x1 vertical tile can be generated.

  Layout:
    [top]
    [bottom]

  IMPORTANT: Generated neighbors can ONLY be on the LONG sides (left/right),
  NOT on the SHORT sides (above top, below bottom).
  This prevents seams since short-side pixels aren't included in the template.
  """
  bottom = Point(top.x, top.y + 1)

  if grid.is_generated(top) or grid.is_generated(bottom):
    return False
  if not grid.bounds.contains(top) or not grid.bounds.contains(bottom):
    return False

  # SHORT sides (ends) - these CANNOT have generated neighbors
  short_side_neighbors = [
    Point(top.x, top.y - 1),  # Above top quadrant
    Point(bottom.x, bottom.y + 1),  # Below bottom quadrant
  ]
  for n in short_side_neighbors:
    if grid.is_generated(n):
      return False  # Would create seam

  # LONG sides (left/right) - at least one must have generated neighbor
  long_side_neighbors = [
    Point(top.x - 1, top.y),  # Left of top
    Point(top.x + 1, top.y),  # Right of top
    Point(bottom.x - 1, bottom.y),  # Left of bottom
    Point(bottom.x + 1, bottom.y),  # Right of bottom
  ]
  for n in long_side_neighbors:
    if grid.is_generated(n):
      return True

  return False


def can_generate_single(grid: QuadrantGrid, p: Point) -> bool:
  """Check if a single quadrant can be generated."""
  if grid.is_generated(p):
    return False
  if not grid.bounds.contains(p):
    return False
  return grid.has_generated_neighbor(p)


def find_best_2x2_tiles(
  grid: QuadrantGrid, direction: str, require_gap: bool = True
) -> list[Point]:
  """
  Find all valid 2x2 tile positions along a direction from the generated region.

  Args:
      grid: The quadrant grid
      direction: One of "top", "bottom", "left", "right"
      require_gap: If True, 2x2 tiles must have a gap from existing content

  Returns list of top-left corners for valid 2x2 tiles.
  """
  gen_bounds = grid.get_generated_bounds()
  if gen_bounds is None:
    return []

  valid_positions = []

  # When require_gap is True, we look for tiles 2 rows/cols away (with 1 row/col gap)
  # When require_gap is False, we look for tiles 1 row/col away (directly adjacent)
  offset = 2 if require_gap else 1

  if direction == "top":
    # Look for 2x2 tiles above the current bounds
    y = gen_bounds.top_left.y - offset - 1  # -1 because 2x2 tile has height 2
    for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
        valid_positions.append(tl)

  elif direction == "bottom":
    # Look for 2x2 tiles below the current bounds
    y = gen_bounds.bottom_right.y + offset
    for x in range(gen_bounds.top_left.x, gen_bounds.bottom_right.x, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
        valid_positions.append(tl)

  elif direction == "left":
    # Look for 2x2 tiles to the left of current bounds
    x = gen_bounds.top_left.x - offset - 1  # -1 because 2x2 tile has width 2
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
        valid_positions.append(tl)

  elif direction == "right":
    # Look for 2x2 tiles to the right of current bounds
    x = gen_bounds.bottom_right.x + offset
    for y in range(gen_bounds.top_left.y, gen_bounds.bottom_right.y, 2):
      tl = Point(x, y)
      if can_generate_2x2(grid, tl, require_gap=require_gap):
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
          GenerationStep(step_num, quadrants, f"Offset 2x2 ({direction}): {tl}")
        )
        grid.mark_multiple_generated(quadrants)
        step_num += 1
        made_progress = True

    # Step B: Fill remaining gaps (bridges and single quadrants)
    # IMPORTANT: 2x2 tiles can NEVER touch existing generated content.
    # Only use 1x2, 2x1, or single tiles for bridging.
    remaining_empty = grid.get_all_empty()
    if remaining_empty:
      # Sort by how many generated neighbors they have (more = better)
      remaining_empty.sort(key=lambda p: -grid.count_generated_neighbors(p))

      for p in remaining_empty:
        if grid.is_generated(p):
          continue

        # Try 1x2 horizontal
        for dx in [0, -1]:
          left = Point(p.x + dx, p.y)
          if can_generate_1x2_horizontal(grid, left):
            quadrants = get_1x2_quadrants(left)
            steps.append(
              GenerationStep(step_num, quadrants, f"Bridge 1x2 horizontal: {left}")
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
              steps.append(GenerationStep(step_num, [p], f"Single quadrant: {p}"))
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
              GenerationStep(step_num, [p], f"Single quadrant (fallback): {p}")
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
  Fill interior gaps using 1x2, 2x1, or single tiles ONLY.

  IMPORTANT:
  - 2x2 tiles can NEVER touch existing generated content.
  - 1x2 tiles can only have generated neighbors on LONG sides (top/bottom)
  - 2x1 tiles can only have generated neighbors on LONG sides (left/right)
  - Interior gaps are often surrounded by generated content, so many
    1x2/2x1 configurations won't be valid due to short-side constraints.

  Returns list of (quadrants, description) tuples.
  """
  result = []
  gap_set = set(gaps)

  while gap_set:
    found = False

    # Sort gaps by how many generated neighbors they have (prioritize well-connected gaps)
    sorted_gaps = sorted(gap_set, key=lambda p: -grid.count_generated_neighbors(p))

    for p in sorted_gaps:
      if p not in gap_set:
        continue

      # Try 1x2 horizontal (use proper validation with short-side constraint)
      right = Point(p.x + 1, p.y)
      if right in gap_set and can_generate_1x2_horizontal(grid, p):
        result.append(([p, right], f"1x2 at {p}"))
        grid.mark_multiple_generated([p, right])
        gap_set -= {p, right}
        found = True
        break

      # Try 2x1 vertical (use proper validation with short-side constraint)
      bottom = Point(p.x, p.y + 1)
      if bottom in gap_set and can_generate_2x1_vertical(grid, p):
        result.append(([p, bottom], f"2x1 at {p}"))
        grid.mark_multiple_generated([p, bottom])
        gap_set -= {p, bottom}
        found = True
        break

      # Single quadrant as last resort
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


def create_and_save_plan(
  conn: sqlite3.Connection,
  bounds: BoundingBox,
  generation_dir: Path,
  visualize: bool = False,
) -> GenerationPlan | None:
  """
  Create a generation plan and save it to automatic_plan.json.

  Returns the plan, or None if no generation is needed.
  """
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
    return None

  # Show initial state
  if visualize:
    print("\n" + grid.visualize())

  # Create plan
  print("\n🔧 Creating generation plan...")

  # Create a copy of the grid for planning
  plan_grid = QuadrantGrid(bounds)
  for p in generated:
    if bounds.contains(p):
      plan_grid.mark_generated(p)

  steps = create_generation_plan(plan_grid)

  print(f"\n📋 Generation plan: {len(steps)} steps")

  # Show plan summary
  total_quadrants = sum(len(step.quadrants) for step in steps)
  by_size: dict[int, int] = {}
  for step in steps:
    size = len(step.quadrants)
    by_size[size] = by_size.get(size, 0) + 1

  print(f"   Total quadrants to generate: {total_quadrants}")
  print("   Steps by tile size:")
  for size in sorted(by_size.keys(), reverse=True):
    label = {4: "2x2 tiles", 2: "1x2/2x1 tiles", 1: "single quadrants"}
    print(f"     {label.get(size, f'{size}-quadrant')}: {by_size[size]}")

  # Create and save plan
  plan = GenerationPlan(
    created_at=datetime.now().isoformat(),
    bounds=bounds,
    steps=steps,
    generation_dir=str(generation_dir),
  )

  plan_path = generation_dir / "automatic_plan.json"
  plan.save(plan_path)

  # Show plan details if visualizing
  if visualize:
    print("\n🔍 Generation plan details:\n")
    viz_grid = QuadrantGrid(bounds)
    for p in generated:
      if bounds.contains(p):
        viz_grid.mark_generated(p)

    for step in steps:
      coords = ", ".join(str(q) for q in step.quadrants)
      print(f"Step {step.step_number}: {step.description}")
      print(f"  Quadrants: [{coords}]")
      print(viz_grid.visualize(step.quadrants, step.step_number))
      viz_grid.mark_multiple_generated(step.quadrants)
      print()

  return plan


def execute_plan(
  conn: sqlite3.Connection,
  plan: GenerationPlan,
  plan_path: Path,
  port: int,
  bucket: str,
  no_start_server: bool,
  max_steps: int | None = None,
) -> int:
  """
  Execute a generation plan, updating status as we go.

  Stops on first error.

  Returns exit code (0 for success, 1 for error).
  """
  # Import here to avoid circular imports
  from isometric_nyc.e2e_generation.shared import (
    WEB_DIR,
    get_generation_config,
    start_web_server,
  )

  # Get pending steps
  pending_steps = plan.get_pending_steps()
  if not pending_steps:
    print("\n✅ All steps in the plan are already complete!")
    summary = plan.get_summary()
    print(f"   Summary: {summary['generated']} generated, {summary['error']} errors")
    return 0

  if max_steps is not None:
    pending_steps = pending_steps[:max_steps]

  print("\n🚀 Executing generation plan...")
  print(f"   {len(pending_steps)} steps to execute")

  web_server = None
  generation_dir = Path(plan.generation_dir)

  try:
    if not no_start_server:
      web_server = start_web_server(WEB_DIR, port)

    config = get_generation_config(conn)

    for step in pending_steps:
      print(f"\n{'=' * 60}")
      print(f"Step {step.step_number}/{len(plan.steps)}: {step.description}")
      print("=" * 60)

      # Convert Points to tuples for the generation API
      quadrant_tuples = [(q.x, q.y) for q in step.quadrants]

      try:
        result = run_generation_step(
          conn,
          config,
          quadrant_tuples,
          generation_dir,
          port,
          bucket,
        )
        if result.get("success"):
          print(f"✅ Step {step.step_number} complete: {result.get('message')}")
          plan.update_step_status(step.step_number, StepStatus.GENERATED)
          plan.save(plan_path)
        else:
          error_msg = result.get("error", "Unknown error")
          print(f"❌ Step {step.step_number} failed: {error_msg}")
          plan.update_step_status(step.step_number, StepStatus.ERROR, error_msg)
          plan.save(plan_path)
          print("\n⛔ Stopping execution due to error.")
          return 1
      except Exception as e:
        error_msg = str(e)
        print(f"❌ Step {step.step_number} failed: {error_msg}")
        plan.update_step_status(step.step_number, StepStatus.ERROR, error_msg)
        plan.save(plan_path)
        print("\n⛔ Stopping execution due to error.")
        return 1

  finally:
    if web_server:
      print("\n🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()

  # Print summary
  summary = plan.get_summary()
  print(f"\n{'=' * 60}")
  print("✅ Plan execution complete!")
  print(
    f"   Generated: {summary['generated']}, Pending: {summary['pending']}, Errors: {summary['error']}"
  )
  print("=" * 60)

  return 0


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

  # Plan creation arguments (mutually exclusive with --plan-json)
  plan_group = parser.add_argument_group("plan creation")
  plan_group.add_argument(
    "--top-left",
    type=str,
    help="Top-left corner of bounding box (x,y)",
  )
  plan_group.add_argument(
    "--bottom-right",
    type=str,
    help="Bottom-right corner of bounding box (x,y)",
  )
  plan_group.add_argument(
    "--visualize",
    action="store_true",
    help="Show ASCII visualization of the plan",
  )

  # Plan execution arguments
  exec_group = parser.add_argument_group("plan execution")
  exec_group.add_argument(
    "--plan-json",
    type=Path,
    help="Path to an existing plan JSON file to execute",
  )
  exec_group.add_argument(
    "--port",
    type=int,
    default=5173,
    help="Web server port (default: 5173)",
  )
  exec_group.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  exec_group.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum number of steps to execute",
  )
  exec_group.add_argument(
    "--bucket",
    default="isometric-nyc-infills",
    help="GCS bucket name for uploading images",
  )

  args = parser.parse_args()

  # Load database
  generation_dir = args.generation_dir.resolve()
  db_path = generation_dir / "quadrants.db"

  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  conn = sqlite3.connect(db_path)

  try:
    # Mode 1: Execute existing plan
    if args.plan_json:
      plan_path = args.plan_json.resolve()
      if not plan_path.exists():
        print(f"❌ Error: Plan file not found: {plan_path}")
        return 1

      print(f"📄 Loading plan from {plan_path}")
      plan = GenerationPlan.load(plan_path)

      print(f"📦 Bounding box: {plan.bounds.top_left} to {plan.bounds.bottom_right}")
      print(f"📋 Total steps: {len(plan.steps)}")
      summary = plan.get_summary()
      print(
        f"   Status: {summary['pending']} pending, {summary['generated']} generated, {summary['error']} errors"
      )

      return execute_plan(
        conn,
        plan,
        plan_path,
        args.port,
        args.bucket,
        args.no_start_server,
        args.max_steps,
      )

    # Mode 2: Create new plan
    if not args.top_left or not args.bottom_right:
      print(
        "❌ Error: Either --plan-json or both --top-left and --bottom-right are required"
      )
      parser.print_help()
      return 1

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

    plan = create_and_save_plan(conn, bounds, generation_dir, args.visualize)

    if plan is None:
      return 0

    print("\n" + "=" * 60)
    print("📄 Plan created and saved to:")
    print(f"   {generation_dir / 'automatic_plan.json'}")
    print("\nTo execute the plan, run:")
    print(
      "   uv run python src/isometric_nyc/e2e_generation/automatic_generation.py \\"
    )
    print(f"     {generation_dir} \\")
    print(f"     --plan-json {generation_dir / 'automatic_plan.json'}")
    print("=" * 60)

    return 0

  finally:
    conn.close()


if __name__ == "__main__":
  exit(main())
