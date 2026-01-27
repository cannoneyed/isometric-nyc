"""
Persisted web renderer for tile generation.

This module manages a single web server process that stays running
for the lifetime of the application, with an internal queue for
handling render requests.
"""

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from PIL import Image
from playwright.sync_api import sync_playwright

from isometric_nyc.generation.shared import (
  CHROMIUM_ARGS,
  DEFAULT_WEB_PORT,
  WEB_RENDER_DIR,
  image_to_png_bytes,
  split_tile_into_quadrants,
  wait_for_server,
)


@dataclass
class RenderRequest:
  """A request to render a quadrant."""

  quadrant_x: int
  quadrant_y: int
  lat: float
  lng: float
  width_px: int
  height_px: int
  camera_azimuth_degrees: float
  camera_elevation_degrees: float
  view_height_meters: float
  callback: Callable[[dict[tuple[int, int], bytes] | None, str | None], None]


class WebRenderer:
  """
  Manages a persistent web server and render queue.

  This class:
  - Starts a web server (bun/vite) when initialized
  - Maintains an internal queue of render requests
  - Processes renders one at a time using Playwright
  - Provides thread-safe access to queue operations
  """

  def __init__(
    self, web_render_dir: Path = WEB_RENDER_DIR, port: int = DEFAULT_WEB_PORT
  ):
    self.web_render_dir = web_render_dir
    self.port = port
    self.web_process: subprocess.Popen | None = None
    self.render_queue: queue.Queue[RenderRequest | None] = queue.Queue()
    self.worker_thread: threading.Thread | None = None
    self.running = False
    self._lock = threading.Lock()

  def start(self) -> None:
    """Start the web server and render worker thread."""
    with self._lock:
      if self.running:
        return

      print(f"🌐 Starting web renderer on port {self.port}...")

      # Start the web server
      self._start_web_server()

      # Start the worker thread
      self.running = True
      self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
      self.worker_thread.start()

      print(f"✅ Web renderer ready on http://localhost:{self.port}")

  def stop(self) -> None:
    """Stop the web server and worker thread."""
    with self._lock:
      if not self.running:
        return

      print("🛑 Stopping web renderer...")
      self.running = False

      # Signal worker to stop
      self.render_queue.put(None)

      # Wait for worker to finish
      if self.worker_thread and self.worker_thread.is_alive():
        self.worker_thread.join(timeout=5.0)

      # Stop the web server
      if self.web_process:
        self.web_process.terminate()
        try:
          self.web_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
          self.web_process.kill()
        self.web_process = None

      print("✅ Web renderer stopped")

  def _start_web_server(self) -> None:
    """Start the web server subprocess."""
    if not self.web_render_dir.exists():
      raise RuntimeError(f"Web directory not found: {self.web_render_dir}")

    self.web_process = subprocess.Popen(
      ["bun", "run", "dev", "--port", str(self.port)],
      cwd=self.web_render_dir,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )

    # Give the process a moment to start
    time.sleep(2)

    # Check if process died immediately
    if self.web_process.poll() is not None:
      stdout = (
        self.web_process.stdout.read().decode() if self.web_process.stdout else ""
      )
      stderr = (
        self.web_process.stderr.read().decode() if self.web_process.stderr else ""
      )
      raise RuntimeError(
        f"Web server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
      )

    # Wait for server to be ready
    print("   ⏳ Waiting for web server to be ready...")
    if wait_for_server(self.port, timeout=30.0):
      print(f"   ✅ Web server ready on http://localhost:{self.port}")
    else:
      if self.web_process.poll() is not None:
        stdout = (
          self.web_process.stdout.read().decode() if self.web_process.stdout else ""
        )
        stderr = (
          self.web_process.stderr.read().decode() if self.web_process.stderr else ""
        )
        raise RuntimeError(
          f"Web server died during startup.\nstdout: {stdout}\nstderr: {stderr}"
        )
      print("   ⚠️  Web server may not be fully ready, continuing anyway...")

  def _ensure_web_server_running(self) -> bool:
    """Ensure the web server is still running, restart if needed."""
    if self.web_process is None or self.web_process.poll() is not None:
      print("   ⚠️  Web server died, restarting...")
      try:
        self._start_web_server()
        return True
      except Exception as e:
        print(f"   ❌ Failed to restart web server: {e}")
        return False
    return True

  def _worker_loop(self) -> None:
    """Main worker loop that processes render requests."""
    print("🔄 Render worker started")

    while self.running:
      try:
        # Wait for a request with timeout to allow checking running flag
        try:
          request = self.render_queue.get(timeout=1.0)
        except queue.Empty:
          continue

        # Check for stop signal
        if request is None:
          break

        # Ensure web server is running
        if not self._ensure_web_server_running():
          request.callback(None, "Web server not available")
          continue

        # Process the render request
        try:
          result = self._render_tile(request)
          request.callback(result, None)
        except Exception as e:
          print(f"   ❌ Render error: {e}")
          request.callback(None, str(e))

      except Exception as e:
        print(f"   ❌ Worker loop error: {e}")

    print("🛑 Render worker stopped")

  def _render_tile(self, request: RenderRequest) -> dict[tuple[int, int], bytes]:
    """
    Render a tile and return all 4 quadrants as PNG bytes.

    Returns a dict mapping (dx, dy) offset to PNG bytes.
    """
    # Build URL for rendering
    params = {
      "export": "true",
      "lat": request.lat,
      "lon": request.lng,
      "width": request.width_px,
      "height": request.height_px,
      "azimuth": request.camera_azimuth_degrees,
      "elevation": request.camera_elevation_degrees,
      "view_height": request.view_height_meters,
    }
    query_string = urlencode(params)
    url = f"http://localhost:{self.port}/?{query_string}"

    print(f"   🎨 Rendering tile at ({request.lat:.6f}, {request.lng:.6f})...")

    # Render using Playwright
    with sync_playwright() as p:
      browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)

      context = browser.new_context(
        viewport={"width": request.width_px, "height": request.height_px},
        device_scale_factor=1,
      )
      page = context.new_page()

      page.goto(url, wait_until="networkidle")

      try:
        page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      except Exception:
        print("      ⚠️  Timeout waiting for tiles, continuing anyway...")

      # Get screenshot as bytes
      screenshot_bytes = page.screenshot(type="png")

      page.close()
      context.close()
      browser.close()

    # Open as PIL image and split into quadrants
    full_tile = Image.open(BytesIO(screenshot_bytes))
    quadrant_images = split_tile_into_quadrants(full_tile)

    # Convert to bytes
    result = {}
    for offset, quad_img in quadrant_images.items():
      result[offset] = image_to_png_bytes(quad_img)

    return result

  def render_quadrant(
    self,
    quadrant_x: int,
    quadrant_y: int,
    lat: float,
    lng: float,
    width_px: int,
    height_px: int,
    camera_azimuth_degrees: float,
    camera_elevation_degrees: float,
    view_height_meters: float,
    timeout: float = 120.0,
  ) -> dict[tuple[int, int], bytes]:
    """
    Synchronously render a quadrant and wait for the result.

    This is a blocking call that waits for the render to complete.

    Args:
      quadrant_x, quadrant_y: Quadrant coordinates
      lat, lng: Geographic coordinates for the render
      width_px, height_px: Tile dimensions
      camera_azimuth_degrees, camera_elevation_degrees: Camera angles
      view_height_meters: View height
      timeout: Maximum time to wait for render

    Returns:
      Dict mapping (dx, dy) offset to PNG bytes for all 4 quadrants

    Raises:
      RuntimeError: If render fails or times out
    """
    result_event = threading.Event()
    result_data: dict[str, Any] = {"result": None, "error": None}

    def callback(
      result: dict[tuple[int, int], bytes] | None, error: str | None
    ) -> None:
      result_data["result"] = result
      result_data["error"] = error
      result_event.set()

    request = RenderRequest(
      quadrant_x=quadrant_x,
      quadrant_y=quadrant_y,
      lat=lat,
      lng=lng,
      width_px=width_px,
      height_px=height_px,
      camera_azimuth_degrees=camera_azimuth_degrees,
      camera_elevation_degrees=camera_elevation_degrees,
      view_height_meters=view_height_meters,
      callback=callback,
    )

    self.render_queue.put(request)

    if not result_event.wait(timeout=timeout):
      raise RuntimeError(f"Render timed out after {timeout}s")

    if result_data["error"]:
      raise RuntimeError(result_data["error"])

    return result_data["result"]

  @property
  def queue_size(self) -> int:
    """Get the current size of the render queue."""
    return self.render_queue.qsize()

  @property
  def is_running(self) -> bool:
    """Check if the renderer is running."""
    return self.running


# Global singleton instance
_renderer: WebRenderer | None = None
_renderer_lock = threading.Lock()


def get_web_renderer(
  web_render_dir: Path = WEB_RENDER_DIR, port: int = DEFAULT_WEB_PORT
) -> WebRenderer:
  """
  Get the global web renderer instance, creating it if necessary.

  This provides a singleton instance of the WebRenderer that can be
  shared across the application.
  """
  global _renderer

  with _renderer_lock:
    if _renderer is None:
      _renderer = WebRenderer(web_render_dir, port)
    return _renderer


def start_global_renderer(
  web_render_dir: Path = WEB_RENDER_DIR, port: int = DEFAULT_WEB_PORT
) -> WebRenderer:
  """Start the global web renderer."""
  renderer = get_web_renderer(web_render_dir, port)
  renderer.start()
  return renderer


def stop_global_renderer() -> None:
  """Stop the global web renderer."""
  global _renderer

  with _renderer_lock:
    if _renderer is not None:
      _renderer.stop()
      _renderer = None
