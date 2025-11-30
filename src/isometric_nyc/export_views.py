"""
Export views from both whitebox.py and web viewer.

This script captures screenshots from both rendering pipelines using the same
view parameters defined in view.json. The outputs can be compared for
alignment verification.

Usage:
  uv run python src/isometric_nyc/export_views.py [--output-dir OUTPUT_DIR]
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from isometric_nyc.whitebox import (
  LAT,
  LON,
  ORIENTATION_DEG,
  SIZE_METERS,
  VIEWPORT_HEIGHT,
  VIEWPORT_WIDTH,
  render_tile,
)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "exports"

# Web server configuration
WEB_DIR = Path(__file__).parent.parent / "web"
WEB_PORT = 5173  # Vite default port


def export_whitebox(output_dir: Path) -> Path:
  """
  Export screenshot from whitebox.py renderer.

  Args:
    output_dir: Directory to save the output

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"whitebox_{timestamp}.png"

  print("🎨 Rendering whitebox view...")
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
    output_path=str(output_path),
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


def export_web_view(output_dir: Path, port: int) -> Path:
  """
  Export screenshot from web viewer using Playwright.

  Args:
    output_dir: Directory to save the output
    port: Port where web server is running

  Returns:
    Path to the saved image
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_path = output_dir / f"web_{timestamp}.png"

  url = f"http://localhost:{port}/?export=true"

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
      viewport={"width": 1280, "height": 720},
      device_scale_factor=1,
    )
    page = context.new_page()

    # Enable console logging from the page
    page.on("console", lambda msg: print(f"   [browser] {msg.text}"))

    # Navigate to the page
    print("   ⏳ Loading page...")
    page.goto(url, wait_until="networkidle")

    # Reload page - Playwright has initialization issues on first load
    print("   🔄 Reloading page...")
    page.wait_for_timeout(2000)
    page.reload(wait_until="networkidle")

    # Wait for tiles to load
    print("   ⏳ Waiting 10 seconds for tiles to load...")
    page.wait_for_timeout(10000)

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

  print("=" * 60)
  print("🏙️  ISOMETRIC NYC VIEW EXPORTER")
  print("=" * 60)
  print(f"📁 Output directory: {args.output_dir}")
  print(f"📍 View: {LAT}, {LON}")
  print(f"📐 Size: {SIZE_METERS}m, Orientation: {ORIENTATION_DEG}°")
  print(f"🖥️  Resolution: {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT}")
  print("=" * 60)

  results = {}
  web_server = None

  try:
    # Export whitebox view
    if not args.web_only:
      whitebox_path = export_whitebox(args.output_dir)
      results["whitebox"] = whitebox_path

    # Export web view
    if not args.whitebox_only:
      # Start web server if needed
      if not args.no_start_server:
        web_server = start_web_server(WEB_DIR, args.port)

      try:
        web_path = export_web_view(args.output_dir, args.port)
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
