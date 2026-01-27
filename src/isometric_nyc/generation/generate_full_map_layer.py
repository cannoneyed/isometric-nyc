"""
Generate a full map layer using parallel model endpoints.

This script executes the generation plan created by init_layer.py, using
multiple model inference endpoints in round-robin fashion.

The generation proceeds in strict step order:
1. Complete ALL step 1 items (2x2 tiles) before starting step 2
2. Complete ALL step 2 items (1x2/2x1 strips) before starting step 3
3. Complete ALL step 3 items (1x1 corners)

Usage:
  uv run python src/isometric_nyc/generation/generate_full_map_layer.py \
    --layer-dir layers/snow

  # Resume from where it left off:
  uv run python src/isometric_nyc/generation/generate_full_map_layer.py \
    --layer-dir layers/snow \
    --resume

  # Retry failed items:
  uv run python src/isometric_nyc/generation/generate_full_map_layer.py \
    --layer-dir layers/snow \
    --retry-errors
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from isometric_nyc.generation.shared import (
  get_quadrant_generation,
  get_quadrant_render,
  png_bytes_to_image,
  stitch_quadrants_to_tile,
)

# Load environment variables
load_dotenv()


@dataclass
class ModelEndpoint:
  """Configuration for a model inference endpoint."""

  name: str
  url: str
  api_key_env: str
  prompt: str | None = None

  @property
  def api_key(self) -> str | None:
    return os.getenv(self.api_key_env)


@dataclass
class GenerationItem:
  """A single item from the generation plan."""

  id: int
  step: int
  block_type: str
  top_left_x: int
  top_left_y: int
  width: int
  height: int
  status: str


def load_layer_config(layer_dir: Path) -> dict:
  """Load the layer configuration."""
  config_path = layer_dir / "layer_config.json"
  if not config_path.exists():
    raise FileNotFoundError(f"Layer config not found: {config_path}")
  with open(config_path) as f:
    return json.load(f)


def get_pending_items(conn: sqlite3.Connection, step: int) -> list[GenerationItem]:
  """Get all pending items for a specific step."""
  cursor = conn.cursor()
  cursor.execute(
    """
        SELECT id, step, block_type, top_left_x, top_left_y, width, height, status
        FROM generation_plan
        WHERE step = ? AND status = 'pending'
        ORDER BY id
    """,
    (step,),
  )

  items = []
  for row in cursor.fetchall():
    items.append(
      GenerationItem(
        id=row[0],
        step=row[1],
        block_type=row[2],
        top_left_x=row[3],
        top_left_y=row[4],
        width=row[5],
        height=row[6],
        status=row[7],
      )
    )
  return items


def get_error_items(conn: sqlite3.Connection) -> list[GenerationItem]:
  """Get all items with error status."""
  cursor = conn.cursor()
  cursor.execute("""
        SELECT id, step, block_type, top_left_x, top_left_y, width, height, status
        FROM generation_plan
        WHERE status = 'error'
        ORDER BY step, id
    """)

  items = []
  for row in cursor.fetchall():
    items.append(
      GenerationItem(
        id=row[0],
        step=row[1],
        block_type=row[2],
        top_left_x=row[3],
        top_left_y=row[4],
        width=row[5],
        height=row[6],
        status=row[7],
      )
    )
  return items


def get_step_progress(conn: sqlite3.Connection) -> dict[int, dict[str, int]]:
  """Get progress counts for each step."""
  cursor = conn.cursor()
  cursor.execute("""
        SELECT step, status, COUNT(*)
        FROM generation_plan
        GROUP BY step, status
    """)

  progress: dict[int, dict[str, int]] = {
    1: {"pending": 0, "in_progress": 0, "complete": 0, "error": 0},
    2: {"pending": 0, "in_progress": 0, "complete": 0, "error": 0},
    3: {"pending": 0, "in_progress": 0, "complete": 0, "error": 0},
  }

  for step, status, count in cursor.fetchall():
    if step in progress and status in progress[step]:
      progress[step][status] = count

  return progress


def mark_item_in_progress(
  conn: sqlite3.Connection, item_id: int, model_name: str
) -> None:
  """Mark an item as in progress."""
  cursor = conn.cursor()
  cursor.execute(
    """
        UPDATE generation_plan
        SET status = 'in_progress', model_name = ?, started_at = ?
        WHERE id = ?
    """,
    (model_name, time.time(), item_id),
  )
  conn.commit()


def mark_item_complete(conn: sqlite3.Connection, item_id: int) -> None:
  """Mark an item as complete."""
  cursor = conn.cursor()
  cursor.execute(
    """
        UPDATE generation_plan
        SET status = 'complete', completed_at = ?
        WHERE id = ?
    """,
    (time.time(), item_id),
  )
  conn.commit()


def mark_item_error(conn: sqlite3.Connection, item_id: int, error_message: str) -> None:
  """Mark an item as having an error."""
  cursor = conn.cursor()
  cursor.execute(
    """
        UPDATE generation_plan
        SET status = 'error', error_message = ?, completed_at = ?
        WHERE id = ?
    """,
    (error_message, time.time(), item_id),
  )
  conn.commit()


def reset_error_items(conn: sqlite3.Connection) -> int:
  """Reset all error items to pending. Returns count of reset items."""
  cursor = conn.cursor()
  cursor.execute("""
        UPDATE generation_plan
        SET status = 'pending', error_message = NULL, started_at = NULL, completed_at = NULL
        WHERE status = 'error'
    """)
  count = cursor.rowcount
  conn.commit()
  return count


def get_source_quadrants(
  source_conn: sqlite3.Connection,
  source_layer: str,
  item: GenerationItem,
) -> dict[tuple[int, int], Image.Image] | None:
  """
  Get source quadrant images for the generation item.

  Returns dict mapping (dx, dy) offset to PIL Image, or None if any quadrant is missing.
  """
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dy in range(item.height):
    for dx in range(item.width):
      qx = item.top_left_x + dx
      qy = item.top_left_y + dy

      if source_layer == "generations":
        data = get_quadrant_generation(source_conn, qx, qy)
      else:
        data = get_quadrant_render(source_conn, qx, qy)

      if data is None:
        return None

      quadrants[(dx, dy)] = png_bytes_to_image(data)

  return quadrants


def stitch_block(quadrants: dict[tuple[int, int], Image.Image]) -> Image.Image:
  """Stitch quadrant images into a single block image."""
  if len(quadrants) == 1:
    # Single quadrant
    return quadrants[(0, 0)]
  elif len(quadrants) == 2:
    # 1x2 or 2x1
    if (1, 0) in quadrants:
      # 2x1 horizontal
      left = quadrants[(0, 0)]
      right = quadrants[(1, 0)]
      width = left.width + right.width
      height = left.height
      result = Image.new("RGBA", (width, height))
      result.paste(left, (0, 0))
      result.paste(right, (left.width, 0))
      return result
    else:
      # 1x2 vertical
      top = quadrants[(0, 0)]
      bottom = quadrants[(0, 1)]
      width = top.width
      height = top.height + bottom.height
      result = Image.new("RGBA", (width, height))
      result.paste(top, (0, 0))
      result.paste(bottom, (0, top.height))
      return result
  else:
    # 2x2 - use the shared utility
    return stitch_quadrants_to_tile(quadrants)


def split_block(
  image: Image.Image,
  item: GenerationItem,
  quadrant_size: int = 512,
) -> dict[tuple[int, int], Image.Image]:
  """Split a generated block image back into quadrants."""
  quadrants: dict[tuple[int, int], Image.Image] = {}

  for dy in range(item.height):
    for dx in range(item.width):
      left = dx * quadrant_size
      top = dy * quadrant_size
      right = left + quadrant_size
      bottom = top + quadrant_size

      quadrants[(dx, dy)] = image.crop((left, top, right, bottom))

  return quadrants


def call_gemini_api(
  input_image: Image.Image,
  prompt: str,
) -> Image.Image:
  """
  Call the Gemini API to generate a transformed version of the image.

  Args:
      input_image: The input image to transform
      prompt: The generation prompt

  Returns:
      Generated PIL Image
  """
  from google import genai
  from google.genai import types

  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment")

  client = genai.Client(api_key=api_key)

  # Upload input image
  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    input_path = tmp.name
    input_image.save(input_path)

  try:
    input_ref = client.files.upload(file=input_path)

    response = client.models.generate_content(
      model="gemini-2.0-flash-exp-image-generation",
      contents=[input_ref, prompt],
      config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
      ),
    )

    # Extract the generated image
    for part in response.parts:
      if part.text is not None:
        print(f"      Model: {part.text[:100]}...")
      elif image := part.as_image():
        return image._pil_image

    raise ValueError("No image in Gemini response")

  finally:
    Path(input_path).unlink(missing_ok=True)


def process_item(
  item: GenerationItem,
  model: ModelEndpoint,
  source_conn: sqlite3.Connection,
  source_layer: str,
  layer_dir: Path,
  prompt: str,
  dry_run: bool = False,
) -> tuple[bool, str]:
  """
  Process a single generation item.

  Returns (success, message) tuple.
  """
  # Get source quadrants
  quadrants = get_source_quadrants(source_conn, source_layer, item)
  if quadrants is None:
    return False, f"Missing source quadrants for ({item.top_left_x}, {item.top_left_y})"

  # Stitch into a single image
  input_image = stitch_block(quadrants)

  if dry_run:
    return (
      True,
      f"Dry run: would process {item.block_type} at ({item.top_left_x}, {item.top_left_y})",
    )

  # Call the model API
  try:
    generated_image = call_gemini_api(input_image, prompt)
  except Exception as e:
    return False, f"API error: {e}"

  # Resize if needed (Gemini may return different sizes)
  expected_width = item.width * 512
  expected_height = item.height * 512
  if generated_image.size != (expected_width, expected_height):
    generated_image = generated_image.resize(
      (expected_width, expected_height),
      Image.Resampling.LANCZOS,
    )

  # Split back into quadrants and save
  generated_quadrants = split_block(generated_image, item)
  generations_dir = layer_dir / "generations"
  generations_dir.mkdir(exist_ok=True)

  saved_count = 0
  for (dx, dy), quad_img in generated_quadrants.items():
    qx = item.top_left_x + dx
    qy = item.top_left_y + dy
    output_path = generations_dir / f"{qx}_{qy}.png"
    quad_img.save(output_path)
    saved_count += 1

  return True, f"Generated {saved_count} quadrant(s)"


def print_progress(
  progress: dict[int, dict[str, int]], current_step: int | None = None
) -> None:
  """Print a progress summary."""
  total_complete = sum(p["complete"] for p in progress.values())
  total_pending = sum(p["pending"] for p in progress.values())
  total_error = sum(p["error"] for p in progress.values())
  total = total_complete + total_pending + total_error

  print(
    f"\n📊 Progress: {total_complete}/{total} complete ({100 * total_complete // max(1, total)}%)"
  )
  for step in [1, 2, 3]:
    p = progress[step]
    total_step = p["complete"] + p["pending"] + p["error"]
    marker = "→ " if step == current_step else "  "
    step_names = {1: "2x2 tiles", 2: "1x2/2x1", 3: "1x1"}
    print(
      f"   {marker}Step {step} ({step_names[step]}): {p['complete']}/{total_step} complete, {p['error']} errors"
    )


def generate_layer(
  layer_dir: Path,
  dry_run: bool = False,
  retry_errors: bool = False,
  max_items: int | None = None,
) -> int:
  """
  Execute the generation plan for a layer.

  Args:
      layer_dir: Path to the layer directory
      dry_run: If True, don't actually call APIs
      retry_errors: If True, reset error items and retry them
      max_items: Maximum number of items to process (for testing)

  Returns:
      Exit code (0 for success, 1 for errors)
  """
  print(f"\n{'=' * 60}")
  print("🚀 Full Map Layer Generation")
  print(f"{'=' * 60}")

  # Load config
  layer_config = load_layer_config(layer_dir)
  print(f"   Layer: {layer_config.get('name', 'unnamed')}")
  print(f"   Source: {layer_config['generation_dir']}")
  print(f"   Source layer: {layer_config['source_layer']}")

  # Connect to databases
  progress_db_path = layer_dir / "progress.db"
  if not progress_db_path.exists():
    print(f"❌ Progress database not found: {progress_db_path}")
    return 1

  source_dir = Path(layer_config["generation_dir"])
  if not source_dir.is_absolute():
    if not source_dir.exists():
      source_dir = layer_dir.parent.parent / layer_config["generation_dir"]

  source_db_path = source_dir / "quadrants.db"
  if not source_db_path.exists():
    print(f"❌ Source database not found: {source_db_path}")
    return 1

  progress_conn = sqlite3.connect(progress_db_path)
  source_conn = sqlite3.connect(source_db_path)

  try:
    # Get generation params
    gen_params = layer_config.get("generation_params", {})
    prompt = gen_params.get(
      "prompt", "Transform this image while preserving its style and structure."
    )

    # Get model endpoints
    model_configs = layer_config.get("model_endpoints", [])
    if not model_configs:
      # Default to Gemini
      model_configs = [
        {
          "name": "gemini",
          "url": "gemini-api",
          "api_key_env": "GEMINI_API_KEY",
        }
      ]

    models = [ModelEndpoint(**cfg) for cfg in model_configs]
    print(f"   Models: {[m.name for m in models]}")

    # Reset errors if requested
    if retry_errors:
      reset_count = reset_error_items(progress_conn)
      if reset_count > 0:
        print(f"\n🔄 Reset {reset_count} error items to pending")

    # Show initial progress
    progress = get_step_progress(progress_conn)
    print_progress(progress)

    # Process each step in order
    items_processed = 0
    model_index = 0

    for step in [1, 2, 3]:
      step_names = {1: "2x2 tiles", 2: "1x2/2x1 strips", 3: "1x1 corners"}
      print(f"\n{'=' * 60}")
      print(f"📦 Step {step}: {step_names[step]}")
      print(f"{'=' * 60}")

      pending = get_pending_items(progress_conn, step)
      if not pending:
        print("   ✅ All items complete")
        continue

      print(f"   {len(pending)} items pending")

      for item in pending:
        if max_items is not None and items_processed >= max_items:
          print(f"\n⚠️  Reached max items limit ({max_items})")
          return 0

        model = models[model_index % len(models)]
        model_index += 1

        print(
          f"\n   [{items_processed + 1}] {item.block_type} at ({item.top_left_x}, {item.top_left_y})"
        )
        print(f"       Model: {model.name}")

        # Mark as in progress
        if not dry_run:
          mark_item_in_progress(progress_conn, item.id, model.name)

        # Process the item
        success, message = process_item(
          item=item,
          model=model,
          source_conn=source_conn,
          source_layer=layer_config["source_layer"],
          layer_dir=layer_dir,
          prompt=prompt,
          dry_run=dry_run,
        )

        if success:
          print(f"       ✅ {message}")
          if not dry_run:
            mark_item_complete(progress_conn, item.id)
        else:
          print(f"       ❌ {message}")
          if not dry_run:
            mark_item_error(progress_conn, item.id, message)

        items_processed += 1

        # Print progress every 10 items
        if items_processed % 10 == 0:
          progress = get_step_progress(progress_conn)
          print_progress(progress, step)

    # Final progress
    progress = get_step_progress(progress_conn)
    print_progress(progress)

    total_errors = sum(p["error"] for p in progress.values())
    if total_errors > 0:
      print(f"\n⚠️  Completed with {total_errors} errors")
      print("   Run with --retry-errors to retry failed items")
      return 1

    print("\n✅ Generation complete!")
    return 0

  finally:
    progress_conn.close()
    source_conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate a full map layer using parallel model endpoints.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "--layer-dir",
    type=Path,
    required=True,
    help="Path to the layer directory",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Don't actually call APIs, just show what would be done",
  )
  parser.add_argument(
    "--retry-errors",
    action="store_true",
    help="Reset error items to pending and retry them",
  )
  parser.add_argument(
    "--max-items",
    type=int,
    default=None,
    help="Maximum number of items to process (for testing)",
  )

  args = parser.parse_args()

  layer_dir = args.layer_dir.resolve()
  if not layer_dir.exists():
    print(f"❌ Error: Layer directory not found: {layer_dir}")
    return 1

  try:
    return generate_layer(
      layer_dir=layer_dir,
      dry_run=args.dry_run,
      retry_errors=args.retry_errors,
      max_items=args.max_items,
    )
  except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())
