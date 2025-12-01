"""
Export views from both whitebox.py and web viewer.

This script captures screenshots from both rendering pipelines using the same
view parameters defined in a JSON configuration file. The outputs can be compared for
alignment verification.

Usage:
  uv run python src/isometric_nyc/export_views.py [VIEW_JSON_PATH] [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from isometric_nyc.whitebox import render_tile

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "exports"
DEFAULT_VIEW_JSON = Path(__file__).parent.parent / "view.json"

# Web server configuration
WEB_DIR = Path(__file__).parent.parent / "web"
WEB_PORT = 5173  # Vite default port


def load_view_config(path: Path) -> Dict[str, Any]:
  """Load view configuration from JSON file."""
  with open(path, "r") as f:
    return json.load(f)


def export_whitebox(output_dir: Path, view_config: Dict[str, Any]) -> Path:
  """
  Export screenshot from whitebox.py renderer.

  Args:
    output_dir: Directory to save the output
    view_config: View configuration dictionary

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "whitebox.png"

  print("🎨 Rendering whitebox view...")
  try:
    render_tile(
      lat=view_config["lat"],
      lon=view_config["lon"],
      size_meters=view_config.get("size_meters", 300),
      orientation_deg=view_config["camera_azimuth_degrees"],
      use_satellite=True,
      viewport_width=view_config["width_px"],
      viewport_height=view_config["height_px"],
      output_path=str(output_path),
      camera_elevation_deg=view_config["camera_elevation_degrees"],
      view_height_meters=view_config.get("view_height_meters", 200),
    )
  except Exception as e:
    print(f"❌ Error rendering whitebox: {e}")
    raise e

  return output_path


def start_web_server(web_dir: Path, port: int) -> subprocess.Popen:
  """
  Start the Vite dev server.

  Args:
    web_dir: Directory containing the web app
    port: Port to run on

  Returns:
    Popen process handle
  """
  print(f"🌐 Starting web server on port {port}...")
  process = subprocess.Popen(
    ["bun", "run", "dev", "--port", str(port)],
    cwd=web_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  # Wait for server to start
  print("   ⏳ Waiting for server to start...")
  time.sleep(5)
  print(f"   ✅ Server started on http://localhost:{port}")
  return process


def process_web_views(tasks: List[Tuple[Path, Dict[str, Any]]], port: int) -> None:
  """
  Process multiple web view exports in a single browser session.

  Args:
    tasks: List of (output_dir, view_config) tuples
    port: Port where web server is running
  """
  if not tasks:
    return

  print(f"🌐 Capturing {len(tasks)} web views...")

  with sync_playwright() as p:
    # Launch browser in headless mode (ensure WebGL works)
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )

    # We can reuse the context if viewport size is constant, but it might vary per view.
    # To be safe and handle varying sizes, we'll create a new page/context per view
    # or check if we can resize.
    # For now, creating a new page per view is safer and reasonably fast.

    for output_dir, view_config in tasks:
      output_dir.mkdir(parents=True, exist_ok=True)
      output_path = output_dir / "render.png"

      print(f"   Processing {output_dir.name}...")

      # Construct URL parameters from view config
      params = {
        "export": "true",
        "lat": view_config["lat"],
        "lon": view_config["lon"],
        "width": view_config["width_px"],
        "height": view_config["height_px"],
        "azimuth": view_config["camera_azimuth_degrees"],
        "elevation": view_config["camera_elevation_degrees"],
        "view_height": view_config.get("view_height_meters", 200),
      }

      query_string = urlencode(params)
      url = f"http://localhost:{port}/?{query_string}"

      context = browser.new_context(
        viewport={"width": view_config["width_px"], "height": view_config["height_px"]},
        device_scale_factor=1,
      )
      page = context.new_page()

      # Enable console logging from the page
      # page.on("console", lambda msg: print(f"   [browser] {msg.text}"))

      # Navigate to the page
      # print(f"   ⏳ Loading page {url}...")
      page.goto(url, wait_until="networkidle")

      # Wait for tiles to load
      # print("   ⏳ Waiting for tiles to stabilize...")
      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
        # print("   ✅ Tiles loaded")
      except Exception as e:
        print(f"   ⚠️  Timeout waiting for tiles to load: {e}")
        print("   📸 Taking screenshot anyway...")

      # Take screenshot
      page.screenshot(path=str(output_path))
      print(f"   ✅ Saved web render to {output_path}")

      page.close()
      context.close()

    browser.close()


def main():
  parser = argparse.ArgumentParser(
    description="Export views from whitebox and web viewer"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    default=None,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--view_json",
    type=Path,
    help="Path to view.json configuration file (for single view)",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory to save exported images (for single view)",
  )
  parser.add_argument(
    "--whitebox-only",
    action="store_true",
    help="Only export whitebox view",
  )
  parser.add_argument(
    "--web-only",
    action="store_true",
    help="Only export web view",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=WEB_PORT,
    help="Port for web server",
  )
  parser.add_argument(
    "--no-start-server",
    action="store_true",
    help="Don't start web server (assume it's already running)",
  )
  parser.add_argument(
    "--limit",
    type=int,
    help="Limit number of tiles to process (for debugging)",
  )

  args = parser.parse_args()

  # 1. Determine tasks: List of (output_dir, view_config)
  tasks = []

  if args.tile_dir and args.tile_dir.exists():
    # Check if it's a single tile directory (has view.json)
    if (args.tile_dir / "view.json").exists():
      print(f"📂 Found single tile directory: {args.tile_dir}")
      view_config = load_view_config(args.tile_dir / "view.json")
      tasks.append((args.tile_dir, view_config))
    else:
      # Assume it's a parent directory containing tile subdirectories
      print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
      subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
      for d in subdirs:
        if (d / "view.json").exists():
          view_config = load_view_config(d / "view.json")
          tasks.append((d, view_config))
      print(f"   Found {len(tasks)} tile directories.")

  elif args.view_json and args.view_json.exists():
    print(f"📄 Using single view file: {args.view_json}")
    view_config = load_view_config(args.view_json)
    tasks.append((args.output_dir, view_config))

  else:
    # Default fallback
    if DEFAULT_VIEW_JSON.exists():
      print(f"📄 Using default view file: {DEFAULT_VIEW_JSON}")
      view_config = load_view_config(DEFAULT_VIEW_JSON)
      tasks.append((args.output_dir, view_config))
    else:
      print("❌ No configuration found. Please provide --tile_dir or --view_json.")
      sys.exit(1)

  if not tasks:
    print("❌ No tasks found.")
    sys.exit(0)

  # Apply limit if requested
  if args.limit:
    print(f"⚠️  Limiting to first {args.limit} tasks.")
    tasks = tasks[: args.limit]

  print("=" * 60)
  print(f"🏙️  ISOMETRIC NYC VIEW EXPORTER - Processing {len(tasks)} tasks")
  print("=" * 60)

  web_server = None

  try:
    # 2. Export Whitebox Views
    if not args.web_only:
      print(f"\n🎨 Starting Whitebox Exports ({len(tasks)} tasks)...")
      for i, (output_dir, view_config) in enumerate(tasks):
        print(f"   [{i + 1}/{len(tasks)}] {output_dir.name}...")
        try:
          export_whitebox(output_dir, view_config)
        except Exception as e:
          print(f"   ❌ Failed to export whitebox for {output_dir.name}: {e}")

    # 3. Export Web Views
    if not args.whitebox_only:
      print(f"\n🌐 Starting Web Exports ({len(tasks)} tasks)...")

      # Start web server if needed (only once)
      if not args.no_start_server:
        web_server = start_web_server(WEB_DIR, args.port)

      try:
        process_web_views(tasks, args.port)
      finally:
        # Stop web server
        if web_server:
          print("🛑 Stopping web server...")
          web_server.terminate()
          web_server.wait()

  except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    if web_server:
      web_server.terminate()
    sys.exit(1)

  print("\n" + "=" * 60)
  print("📦 EXPORT COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()
