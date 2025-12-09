"""
Command-line script to generate pixel art for quadrants using the Oxen.ai model.

This script generates pixel art for specified quadrants, supporting both
direct quadrant specification and batch processing via JSON files.

Usage:
  # Generate specific quadrants:
  uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \\
    <generation_dir> \\
    --quadrants "(0,1),(0,2)"

  # Process a batch JSON file:
  uv run python src/isometric_nyc/e2e_generation/generate_tiles_omni.py \\
    <generation_dir> \\
    --quadrants-json path/to/quadrants.json

JSON File Format:
  [
    {"quadrants": "(0,0),(0,1)", "status": "pending"},
    {"quadrants": "(1,0),(1,1)", "status": "pending"},
    ...
  ]

The script will:
- Process entries with status "pending"
- Update status to "done" on success or "error" on failure
- Save progress after each entry so it can resume if interrupted
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.generate_omni import (
  parse_quadrant_list,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)


def load_quadrants_json(json_path: Path) -> list[dict]:
  """
  Load quadrant entries from a JSON file.

  Args:
      json_path: Path to the JSON file

  Returns:
      List of quadrant entry dicts with 'quadrants' and 'status' keys
  """
  with open(json_path) as f:
    return json.load(f)


def save_quadrants_json(json_path: Path, entries: list[dict]) -> None:
  """
  Save quadrant entries to a JSON file.

  Args:
      json_path: Path to the JSON file
      entries: List of quadrant entry dicts
  """
  with open(json_path, "w") as f:
    json.dump(entries, f, indent=2)


def generate_quadrants(
  generation_dir: Path,
  quadrant_tuples: list[tuple[int, int]],
  port: int,
  no_start_server: bool,
) -> dict:
  """
  Generate pixel art for the specified quadrants.

  Args:
      generation_dir: Path to the generation directory
      quadrant_tuples: List of (x, y) quadrant coordinates
      port: Web server port
      no_start_server: If True, don't start the web server

  Returns:
      Dict with success status and message/error
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    return {"success": False, "error": f"Database not found: {db_path}"}

  conn = sqlite3.connect(db_path)
  web_server = None

  try:
    config = get_generation_config(conn)

    if not no_start_server:
      web_server = start_web_server(WEB_DIR, port)

    result = run_generation_for_quadrants(
      conn=conn,
      config=config,
      selected_quadrants=quadrant_tuples,
      port=port,
    )
    return result

  finally:
    conn.close()
    if web_server:
      print("🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()


def process_quadrants_list(
  generation_dir: Path,
  quadrants_str: str,
  port: int,
  no_start_server: bool,
) -> int:
  """
  Process a comma-separated list of quadrant tuples.

  Args:
      generation_dir: Path to the generation directory
      quadrants_str: String like "(0,1),(0,2)"
      port: Web server port
      no_start_server: If True, don't start the web server

  Returns:
      Exit code (0 for success, 1 for error)
  """
  try:
    quadrant_tuples = parse_quadrant_list(quadrants_str)
  except ValueError as e:
    print(f"❌ Error parsing quadrants: {e}")
    return 1

  print(f"🎯 Generating quadrants: {quadrant_tuples}")
  result = generate_quadrants(generation_dir, quadrant_tuples, port, no_start_server)

  if result.get("success"):
    print(f"✅ {result.get('message')}")
    return 0
  else:
    print(f"❌ Error: {result.get('error')}")
    return 1


def process_quadrants_json(
  generation_dir: Path,
  json_path: Path,
  port: int,
  no_start_server: bool,
) -> int:
  """
  Process a JSON file of quadrant entries.

  The script will:
  - Process entries with status "pending"
  - Update status to "done" on success or "error" on failure
  - Save progress after each entry

  Args:
      generation_dir: Path to the generation directory
      json_path: Path to the JSON file
      port: Web server port
      no_start_server: If True, don't start the web server

  Returns:
      Exit code (0 for all success, 1 if any error)
  """
  # Load entries
  try:
    entries = load_quadrants_json(json_path)
  except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"❌ Error loading JSON file: {e}")
    return 1

  # Count pending entries
  pending_entries = [e for e in entries if e.get("status") == "pending"]
  if not pending_entries:
    print("✅ No pending entries to process")
    return 0

  print(f"📋 Found {len(pending_entries)} pending entries out of {len(entries)} total")

  # Connect to database
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Database not found: {db_path}")
    return 1

  conn = sqlite3.connect(db_path)
  web_server = None
  had_error = False

  try:
    config = get_generation_config(conn)

    if not no_start_server:
      web_server = start_web_server(WEB_DIR, port)

    # Process each pending entry
    for i, entry in enumerate(entries):
      if entry.get("status") != "pending":
        continue

      quadrants_str = entry.get("quadrants", "")
      entry_idx = entries.index(entry)

      print(f"\n{'=' * 60}")
      print(f"📦 Processing entry {i + 1}/{len(pending_entries)}: {quadrants_str}")
      print(f"{'=' * 60}")

      try:
        quadrant_tuples = parse_quadrant_list(quadrants_str)
      except ValueError as e:
        print(f"❌ Error parsing quadrants: {e}")
        entries[entry_idx]["status"] = "error"
        entries[entry_idx]["error"] = str(e)
        save_quadrants_json(json_path, entries)
        had_error = True
        break

      try:
        result = run_generation_for_quadrants(
          conn=conn,
          config=config,
          selected_quadrants=quadrant_tuples,
          port=port,
        )

        if result.get("success"):
          print(f"✅ {result.get('message')}")
          entries[entry_idx]["status"] = "done"
        else:
          error_msg = result.get("error", "Unknown error")
          print(f"❌ Error: {error_msg}")
          entries[entry_idx]["status"] = "error"
          entries[entry_idx]["error"] = error_msg
          had_error = True

      except Exception as e:
        print(f"❌ Exception: {e}")
        entries[entry_idx]["status"] = "error"
        entries[entry_idx]["error"] = str(e)
        had_error = True

      # Save progress after each entry
      save_quadrants_json(json_path, entries)

      # Stop on error
      if had_error:
        print("\n⛔ Stopping due to error. Progress has been saved.")
        break

  finally:
    conn.close()
    if web_server:
      print("\n🛑 Stopping web server...")
      web_server.terminate()
      web_server.wait()

  # Print summary
  done_count = sum(1 for e in entries if e.get("status") == "done")
  error_count = sum(1 for e in entries if e.get("status") == "error")
  pending_count = sum(1 for e in entries if e.get("status") == "pending")

  print(f"\n{'=' * 60}")
  print("📊 Summary:")
  print(f"   Done: {done_count}")
  print(f"   Error: {error_count}")
  print(f"   Pending: {pending_count}")
  print(f"{'=' * 60}")

  return 1 if had_error else 0


def main():
  parser = argparse.ArgumentParser(
    description="Generate pixel art for quadrants using the Oxen.ai model.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )

  # Quadrant specification (mutually exclusive)
  input_group = parser.add_mutually_exclusive_group(required=True)
  input_group.add_argument(
    "--quadrants",
    type=str,
    help='Comma-separated quadrant tuples to generate, e.g. "(0,1),(0,2)"',
  )
  input_group.add_argument(
    "--quadrants-json",
    type=Path,
    help="Path to JSON file with quadrant entries to process",
  )

  # Server options
  parser.add_argument(
    "--port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Web server port for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )

  args = parser.parse_args()

  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  if not generation_dir.is_dir():
    print(f"❌ Error: Not a directory: {generation_dir}")
    return 1

  print(f"📂 Generation directory: {generation_dir}")

  try:
    if args.quadrants:
      return process_quadrants_list(
        generation_dir,
        args.quadrants,
        args.port,
        args.no_start_server,
      )
    else:
      json_path = args.quadrants_json.resolve()
      if not json_path.exists():
        print(f"❌ Error: JSON file not found: {json_path}")
        return 1
      print(f"📄 JSON file: {json_path}")
      return process_quadrants_json(
        generation_dir,
        json_path,
        args.port,
        args.no_start_server,
      )

  except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())
