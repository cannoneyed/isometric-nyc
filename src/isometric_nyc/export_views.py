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
from typing import Any, Dict
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


def export_web_view(output_dir: Path, port: int, view_config: Dict[str, Any]) -> Path:
  """
  Export screenshot from web viewer using Playwright.

  Args:
    output_dir: Directory to save the output
    port: Port where web server is running
    view_config: View configuration dictionary

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "render.png"

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

  print(f"🌐 Capturing web view from {url}...")

  with sync_playwright() as p:
    # Launch browser in non-headless mode for proper WebGL rendering
    browser = p.chromium.launch(
      headless=False,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )
    context = browser.new_context(
      viewport={"width": view_config["width_px"], "height": view_config["height_px"]},
      device_scale_factor=1,
    )
    page = context.new_page()

    # Enable console logging from the page
    page.on("console", lambda msg: print(f"   [browser] {msg.text}"))

    # Navigate to the page
    print("   ⏳ Loading page...")
    page.goto(url, wait_until="networkidle")

    # Wait for tiles to load
    print("   ⏳ Waiting for tiles to stabilize (window.TILES_LOADED)...")
    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      print("   ✅ Tiles loaded signal received")
    except Exception as e:
      print(f"   ⚠️  Timeout waiting for tiles to load: {e}")
      print("   📸 Taking screenshot anyway...")

    # Take screenshot
    print("   📸 Taking screenshot...")
    page.screenshot(path=str(output_path))

    browser.close()

  print(f"   ✅ Saved to {output_path}")
  return output_path


def main():
  parser = argparse.ArgumentParser(
    description="Export views from whitebox and web viewer"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    default=None,
    help="Path to tile generation directory",
  )
  parser.add_argument(
    "--view_json",
    type=Path,
    help="Path to view.json configuration file",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory to save exported images",
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

  args = parser.parse_args()

  if args.tile_dir.exists():
    view_json_path = args.tile_dir / "view.json"
    output_dir = args.tile_dir
    if not view_json_path.exists():
      print(f"❌ Error: View config file not found: {view_json_path}")
      sys.exit(1)
  else:
    output_dir = args.output_dir
    if not args.view_json.exists():
      view_json_path = DEFAULT_VIEW_JSON
    else:
      view_json_path = args.view_json

  view_config = load_view_config(view_json_path)

  print("=" * 60)
  print("🏙️  ISOMETRIC NYC VIEW EXPORTER")
  print("=" * 60)
  print(f"📄 Config: {view_json_path}")
  print(f"📁 Output directory: {output_dir}")
  print(f"📍 View: {view_config['lat']}, {view_config['lon']}")
  print(
    f"📐 Size: {view_config.get('size_meters', 300)}m, Orientation: {view_config['camera_azimuth_degrees']}°"
  )
  print(f"🖥️  Resolution: {view_config['width_px']}x{view_config['height_px']}")
  print("=" * 60)

  results = {}
  web_server = None

  try:
    # Export whitebox view
    if not args.web_only:
      whitebox_path = export_whitebox(output_dir, view_config)
      results["whitebox"] = whitebox_path

    # Export web view
    if not args.whitebox_only:
      # Start web server if needed
      if not args.no_start_server:
        web_server = start_web_server(WEB_DIR, args.port)

      try:
        web_path = export_web_view(output_dir, args.port, view_config)
        results["web"] = web_path
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

  # Summary
  print("\n" + "=" * 60)
  print("📦 EXPORT COMPLETE")
  print("=" * 60)
  for name, path in results.items():
    print(f"   {name}: {path}")

  return results


if __name__ == "__main__":
  main()
