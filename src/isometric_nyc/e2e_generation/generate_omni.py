"""
Reusable library for generating pixel art using the Oxen.ai model.

This module provides the core generation logic that can be used by:
- view_generations.py (Flask web server)
- generate_tiles_omni.py (command-line script)
- automatic_generation.py (automated generation)

The main entry point is `run_generation_for_quadrants()` which handles:
1. Validating the quadrant selection
2. Rendering any missing quadrants
3. Building the template image
4. Uploading to GCS and calling the Oxen API
5. Saving the generated quadrants to the database
"""

import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable

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
  DEFAULT_WEB_PORT,
  build_tile_render_url,
  ensure_quadrant_exists,
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
  render_url_to_image,
  save_quadrant_generation,
  save_quadrant_render,
  split_tile_into_quadrants,
  upload_to_gcs,
)

# Load environment variables
load_dotenv()

# Oxen API configuration
OMNI_MODEL_ID = "cannoneyed-gentle-gold-antlion"
OMNI_WATER_MODEL_ID = "cannoneyed-quiet-green-lamprey"
OMNI_WATER_V2_MODEL_ID = "cannoneyed-rural-rose-dingo"

GCS_BUCKET_NAME = "isometric-nyc-infills"


# =============================================================================
# Quadrant Parsing Utilities
# =============================================================================


def parse_quadrant_tuple(s: str) -> tuple[int, int]:
  """
  Parse a quadrant tuple string like "(0,1)" or "0,1" into a tuple.

  Args:
      s: String in format "(x,y)" or "x,y"

  Returns:
      Tuple of (x, y) coordinates

  Raises:
      ValueError: If the format is invalid
  """
  s = s.strip()
  # Remove optional parentheses
  if s.startswith("(") and s.endswith(")"):
    s = s[1:-1]
  parts = s.split(",")
  if len(parts) != 2:
    raise ValueError(f"Invalid quadrant tuple format: {s}")
  return (int(parts[0].strip()), int(parts[1].strip()))


def parse_quadrant_list(s: str) -> list[tuple[int, int]]:
  """
  Parse a comma-separated list of quadrant tuples.

  Args:
      s: String like "(0,1),(0,2)" or "(0,1), (0,2)"

  Returns:
      List of (x, y) coordinate tuples

  Raises:
      ValueError: If the format is invalid
  """
  # Use regex to find all (x,y) patterns
  pattern = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
  matches = re.findall(pattern, s)
  if not matches:
    raise ValueError(f"No valid quadrant tuples found in: {s}")
  return [(int(x), int(y)) for x, y in matches]


# =============================================================================
# Local Inference API Functions
# =============================================================================


def call_local_api_b64(
  image: "Image.Image",
  model_config: "ModelConfig | None" = None,  # noqa: F821
  additional_prompt: str | None = None,
  use_jpeg: bool = True,
  jpeg_quality: int = 90,
) -> "Image.Image":
  """
  Call the local inference API using base64 encoding (faster than multipart).

  The local API expects JSON with:
    - image_b64: Base64-encoded input image
    - prompt: The generation prompt
    - steps: Number of inference steps

  Args:
      image: PIL Image of the input template
      model_config: Optional model configuration (ModelConfig from model_config.py).
        If not provided, uses defaults.
      additional_prompt: Optional custom prompt text to override the base prompt
      use_jpeg: If True, compress image as JPEG (much smaller). Default True.
      jpeg_quality: JPEG quality 1-100 (default 90, good balance of size/quality)

  Returns:
      PIL Image of the generated result

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  import base64
  import time
  from io import BytesIO

  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "http://localhost:8888/edit-b64"
    num_inference_steps = 15

  # Build prompt - custom prompt overrides default
  if additional_prompt:
    prompt = additional_prompt
    print(f"   📝 Using custom prompt: {additional_prompt}")
  else:
    prompt = (
      "Fill in the outlined section with the missing pixels corresponding to "
      "the <isometric nyc pixel art> style, removing the border and exactly "
      "following the shape/style/structure of the surrounding image (if present)."
    )

  # Convert PIL image to base64 (use JPEG for compression if enabled)
  img_buffer = BytesIO()
  if use_jpeg:
    # Convert to RGB if needed (JPEG doesn't support alpha)
    if image.mode in ("RGBA", "LA", "P"):
      rgb_image = Image.new("RGB", image.size, (255, 255, 255))
      if image.mode == "P":
        image = image.convert("RGBA")
      rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
      image = rgb_image
    image.save(img_buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    img_format = "JPEG"
    tmp_ext = ".jpg"
  else:
    image.save(img_buffer, format="PNG")
    img_format = "PNG"
    tmp_ext = ".png"

  # Save temp image for debugging before encoding to base64
  tmp_file = tempfile.NamedTemporaryFile(
    delete=False, suffix=f"_local_api_input{tmp_ext}", prefix="isometric_"
  )
  tmp_file.write(img_buffer.getvalue())
  tmp_file.close()
  print(f"   💾 Saved temp image for debugging: {tmp_file.name}")

  image_b64 = base64.b64encode(img_buffer.getvalue()).decode()

  # Prepare JSON payload
  payload = {
    "image_b64": image_b64,
    "prompt": prompt,
    "steps": num_inference_steps,
  }

  print(f"   🏠 Calling local API (base64) at {endpoint}...")
  print(f"   📊 Steps: {num_inference_steps}")
  print(f"   📦 Image: {len(image_b64):,} bytes ({img_format}, base64)")

  start_time = time.time()
  response = requests.post(
    endpoint,
    json=payload,
    headers={"Content-Type": "application/json"},
    timeout=600,  # 10 minute timeout for local inference
  )
  response.raise_for_status()
  elapsed_time = time.time() - start_time

  # Parse JSON response
  result = response.json()

  if "image_b64" not in result:
    raise ValueError(
      f"Expected 'image_b64' in response, got keys: {list(result.keys())}"
    )

  result_b64 = result["image_b64"]
  print(f"   ✓ Received {len(result_b64)} bytes (base64) from local API")
  print(f"   ⏱️  Generation took {elapsed_time:.1f}s")

  # Decode base64 to PIL Image
  result_image = Image.open(BytesIO(base64.b64decode(result_b64)))

  # Save generated image to temp file for debugging
  tmp_output = tempfile.NamedTemporaryFile(
    delete=False, suffix="_local_api_output.png", prefix="isometric_"
  )
  result_image.save(tmp_output.name, format="PNG")
  print(f"   💾 Saved generated image for debugging: {tmp_output.name}")

  return result_image


def call_local_api(
  image: "Image.Image",
  model_config: "ModelConfig | None" = None,  # noqa: F821
  additional_prompt: str | None = None,
  use_jpeg: bool = True,
  jpeg_quality: int = 90,
) -> "Image.Image":
  """
  Call the local inference API to generate pixel art.

  The local API expects multipart/form-data with:
    - file: The input image as PNG or JPEG
    - prompt: The generation prompt
    - steps: Number of inference steps

  Args:
      image: PIL Image of the input template
      model_config: Optional model configuration (ModelConfig from model_config.py).
        If not provided, uses defaults.
      additional_prompt: Optional custom prompt text to override the base prompt
      use_jpeg: If True, compress image as JPEG (much smaller). Default True.
      jpeg_quality: JPEG quality 1-100 (default 90, good balance of size/quality)

  Returns:
      PIL Image of the generated result

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  import time
  from io import BytesIO

  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "http://localhost:8888/edit"
    num_inference_steps = 15

  # Build prompt - custom prompt overrides default
  if additional_prompt:
    prompt = additional_prompt
    print(f"   📝 Using custom prompt: {additional_prompt}")
  else:
    prompt = (
      "Fill in the outlined section with the missing pixels corresponding to "
      "the <isometric nyc pixel art> style, removing the border and exactly "
      "following the shape/style/structure of the surrounding image (if present)."
    )

  # Convert PIL image to bytes (use JPEG for compression if enabled)
  img_buffer = BytesIO()
  if use_jpeg:
    # Convert to RGB if needed (JPEG doesn't support alpha)
    if image.mode in ("RGBA", "LA", "P"):
      rgb_image = Image.new("RGB", image.size, (255, 255, 255))
      if image.mode == "P":
        image = image.convert("RGBA")
      rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
      image = rgb_image
    image.save(img_buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    filename = "input.jpg"
    mime_type = "image/jpeg"
    img_format = "JPEG"
  else:
    image.save(img_buffer, format="PNG")
    filename = "input.png"
    mime_type = "image/png"
    img_format = "PNG"
  img_buffer.seek(0)

  # Prepare multipart form data
  files = {
    "file": (filename, img_buffer, mime_type),
  }
  data = {
    "prompt": prompt,
    "steps": str(num_inference_steps),
  }

  print(f"   🏠 Calling local API at {endpoint}...")
  print(f"   📊 Steps: {num_inference_steps}")
  print(f"   📦 Image: {img_buffer.getbuffer().nbytes:,} bytes ({img_format})")

  start_time = time.time()
  response = requests.post(
    endpoint,
    files=files,
    data=data,
    headers={"accept": "image/png"},
    timeout=600,  # 10 minute timeout for local inference
  )
  response.raise_for_status()
  elapsed_time = time.time() - start_time

  # Check content type
  content_type = response.headers.get("content-type", "")
  if "image" not in content_type:
    raise ValueError(f"Expected image response, got content-type: {content_type}")

  print(f"   ✓ Received {len(response.content)} bytes from local API")
  print(f"   ⏱️  Generation took {elapsed_time:.1f}s")

  # Parse the response as an image
  return Image.open(BytesIO(response.content))


# =============================================================================
# Oxen API Functions
# =============================================================================


def call_oxen_api(
  image_url: str,
  model_config: "ModelConfig | None" = None,  # noqa: F821
  additional_prompt: str | None = None,
) -> str:
  """
  Call the Oxen API to generate pixel art.

  Args:
      image_url: Public URL of the input template image
      model_config: Optional model configuration (ModelConfig from model_config.py).
        If not provided, uses defaults.
      additional_prompt: Optional custom prompt text to override the base prompt

  Returns:
      URL of the generated image

  Raises:
      requests.HTTPError: If the API call fails
      ValueError: If the response format is unexpected
  """
  import time

  # Use provided config or defaults
  if model_config is not None:
    endpoint = model_config.endpoint
    model_id = model_config.model_id
    api_key = model_config.api_key
    num_inference_steps = model_config.num_inference_steps
  else:
    endpoint = "https://hub.oxen.ai/api/images/edit"
    model_id = OMNI_WATER_MODEL_ID
    api_key = os.getenv("OXEN_OMNI_v04_WATER_API_KEY")
    num_inference_steps = 28

  if not api_key:
    raise ValueError(f"API key not found for model {model_id}")

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }

  # Build prompt - custom prompt overrides default
  if additional_prompt:
    prompt = additional_prompt
    print(f"   📝 Using custom prompt: {additional_prompt}")
  else:
    prompt = (
      "Fill in the outlined section with the missing pixels corresponding to "
      "the <isometric nyc pixel art> style, removing the border and exactly "
      "following the shape/style/structure of the surrounding image (if present)."
    )

  payload = {
    "model": model_id,
    "input_image": image_url,
    "prompt": prompt,
    "num_inference_steps": num_inference_steps,
  }

  print(f"   🤖 Calling Oxen API with model {model_id}...")
  start_time = time.time()
  response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
  response.raise_for_status()
  elapsed_time = time.time() - start_time

  result = response.json()
  print(f"   ⏱️  Oxen API call took {elapsed_time:.1f}s")

  # Log the response structure for debugging
  print(f"   📥 API response keys: {list(result.keys())}")

  # Try various response formats
  if "images" in result and len(result["images"]) > 0:
    image_data = result["images"][0]
    print(
      f"   📥 Image data keys: {list(image_data.keys()) if isinstance(image_data, dict) else type(image_data)}"
    )

    # Try different possible keys for the image URL
    if isinstance(image_data, dict):
      if "url" in image_data:
        return image_data["url"]
      elif "image_url" in image_data:
        return image_data["image_url"]
      elif "data" in image_data:
        # Some APIs return base64 data - we'd need to handle this differently
        raise ValueError(
          f"API returned base64 data instead of URL: {list(image_data.keys())}"
        )
      else:
        raise ValueError(
          f"Image data missing 'url' key. Available keys: {list(image_data.keys())}"
        )
    elif isinstance(image_data, str):
      # Direct URL string
      return image_data
    else:
      raise ValueError(f"Unexpected image data type: {type(image_data)}")
  elif "url" in result:
    return result["url"]
  elif "image_url" in result:
    return result["image_url"]
  elif "output" in result:
    return result["output"]
  elif "error" in result:
    raise ValueError(f"API returned error: {result['error']}")
  elif "message" in result:
    raise ValueError(f"API returned message: {result['message']}")
  else:
    raise ValueError(
      f"Unexpected API response format. Keys: {list(result.keys())}, Full response: {result}"
    )


def download_image_to_pil(
  url: str,
  max_retries: int = 3,
  retry_delay: float = 10.0,
) -> Image.Image:
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
# Rendering Functions
# =============================================================================


def render_quadrant(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
  port: int,
) -> bytes | None:
  """
  Render a quadrant and save to database.

  This renders the tile containing the quadrant and saves all 4 quadrants.

  Args:
      conn: Database connection
      config: Generation config dict
      x: Quadrant x coordinate
      y: Quadrant y coordinate
      port: Web server port for rendering

  Returns:
      PNG bytes of the rendered quadrant, or None if failed
  """
  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  # Build URL and render using shared utilities
  url = build_tile_render_url(
    port=port,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    azimuth=config["camera_azimuth_degrees"],
    elevation=config["camera_elevation_degrees"],
    view_height=config.get("view_height_meters", 200),
  )

  full_tile = render_url_to_image(url, config["width_px"], config["height_px"])
  quadrant_images = split_tile_into_quadrants(full_tile)

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), quad_img in quadrant_images.items():
    qx, qy = x + dx, y + dy
    png_bytes = image_to_png_bytes(quad_img)
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


# =============================================================================
# Core Generation Logic
# =============================================================================


def run_generation_for_quadrants(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  port: int = DEFAULT_WEB_PORT,
  bucket_name: str = GCS_BUCKET_NAME,
  status_callback: Callable[[str, str], None] | None = None,
  model_config: "ModelConfig | None" = None,  # noqa: F821
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is the main entry point for generation. It:
  1. Validates the quadrant selection
  2. Renders any missing quadrants
  3. Builds the template image with appropriate borders
  4. Uploads to GCS and calls the Oxen API
  5. Saves the generated quadrants to the database

  Args:
      conn: Database connection
      config: Generation config dict
      selected_quadrants: List of (x, y) quadrant coordinates to generate
      port: Web server port for rendering (default: 5173)
      bucket_name: GCS bucket name for uploads
      status_callback: Optional callback(status, message) for progress updates
      model_config: Optional model configuration for the Oxen API (ModelConfig from model_config.py)
      context_quadrants: Optional list of (x, y) quadrant coordinates to use as
        context. These quadrants provide surrounding pixel art context for the
        generation. If a context quadrant has a generation, that will be used;
        otherwise the render will be used.
      prompt: Optional custom prompt text for generation (overrides the default prompt)

  Returns:
      Dict with:
          - success: bool
          - message: str (on success)
          - error: str (on failure)
          - quadrants: list of generated quadrant coords (on success)
  """
  # Convert context quadrants to a set for fast lookup
  context_set: set[tuple[int, int]] = (
    set(context_quadrants) if context_quadrants else set()
  )
  if context_set:
    print(f"   📋 Using {len(context_set)} context quadrant(s): {list(context_set)}")
  if prompt:
    print(f"   📝 Additional prompt: {prompt}")

  def update_status(status: str, message: str = "") -> None:
    if status_callback:
      status_callback(status, message)

  update_status("validating", "Checking API key...")

  # Create helper functions for validation
  # These are modified to treat context quadrants as "generated" if they have
  # either a generation or a render
  def has_generation_in_db(qx: int, qy: int) -> bool:
    # Check if this quadrant has an actual generation
    gen = shared_get_quadrant_generation(conn, qx, qy)
    if gen is not None:
      return True

    # For context quadrants, treat them as "generated" if they have a render
    # This allows context quadrants to provide surrounding content even if
    # they don't have generations yet
    if (qx, qy) in context_set:
      render = shared_get_quadrant_render(conn, qx, qy)
      return render is not None

    return False

  def get_render_from_db_with_render(qx: int, qy: int) -> Image.Image | None:
    """Get render, rendering if it doesn't exist yet."""
    render_bytes = shared_get_quadrant_render(conn, qx, qy)
    if render_bytes:
      return png_bytes_to_image(render_bytes)

    # Need to render
    update_status("rendering", f"Rendering quadrant ({qx}, {qy})...")
    print(f"   📦 Rendering quadrant ({qx}, {qy})...")
    render_bytes = render_quadrant(conn, config, qx, qy, port)
    if render_bytes:
      return png_bytes_to_image(render_bytes)
    return None

  def get_generation_from_db(qx: int, qy: int) -> Image.Image | None:
    """Get generation, falling back to render for context quadrants."""
    gen_bytes = shared_get_quadrant_generation(conn, qx, qy)
    if gen_bytes:
      return png_bytes_to_image(gen_bytes)

    # For context quadrants, fall back to render if no generation exists
    # This allows context quadrants to provide surrounding pixel art context
    # even if they only have renders
    if (qx, qy) in context_set:
      render_bytes = shared_get_quadrant_render(conn, qx, qy)
      if render_bytes:
        print(f"   📋 Using render as context for ({qx}, {qy})")
        return png_bytes_to_image(render_bytes)

    return None

  update_status("validating", "Validating quadrant selection...")

  # Validate selection with auto-expansion
  is_valid, msg, placement = validate_quadrant_selection(
    selected_quadrants, has_generation_in_db, allow_expansion=True
  )

  if not is_valid:
    update_status("error", msg)
    return {"success": False, "error": msg}

  print(f"✅ Validation: {msg}")

  # Get primary quadrants (the ones user selected, not padding)
  primary_quadrants = (
    placement.primary_quadrants if placement.primary_quadrants else selected_quadrants
  )
  padding_quadrants = placement.padding_quadrants if placement else []

  if padding_quadrants:
    print(f"   📦 Padding quadrants: {padding_quadrants}")

  # Create the infill region (may be expanded)
  if placement._expanded_region is not None:
    region = placement._expanded_region
  else:
    region = InfillRegion.from_quadrants(selected_quadrants)

  # Build the template
  update_status("rendering", "Building template image...")
  builder = TemplateBuilder(
    region,
    has_generation_in_db,
    get_render_from_db_with_render,
    get_generation_from_db,
    model_config=model_config,
  )

  print("📋 Building template...")
  result = builder.build(border_width=2, allow_expansion=True)

  if result is None:
    error_msg = builder._last_validation_error or "Failed to build template"
    update_status("error", error_msg)
    return {
      "success": False,
      "error": error_msg,
    }

  template_image, placement = result

  # Check if we're using local or Oxen API
  is_local = model_config is not None and model_config.is_local
  use_base64 = model_config is not None and model_config.use_base64

  template_path = None
  try:
    if is_local:
      # Local inference - no GCS upload needed
      print(f"📋 Template size: {template_image.size[0]}x{template_image.size[1]}")

      update_status("generating", "Calling local model (this may take a minute)...")
      if use_base64:
        print("🏠 Using local inference (base64 mode)...")
        generated_image = call_local_api_b64(template_image, model_config, prompt)
      else:
        print("🏠 Using local inference (multipart mode)...")
        generated_image = call_local_api(template_image, model_config, prompt)
      print("   ✓ Local inference complete")
    else:
      # Oxen API - upload to GCS first
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        template_path = Path(tmp.name)
        template_image.save(template_path)

      update_status("uploading", "Uploading template to cloud...")
      print("📤 Uploading template to GCS...")
      print(f"   Template path: {template_path}")
      print(f"   Template size: {template_image.size[0]}x{template_image.size[1]}")
      image_url = upload_to_gcs(template_path, bucket_name)
      print(f"   Uploaded URL: {image_url}")

      update_status("generating", "Calling AI model (this may take a minute)...")
      print("🤖 Calling Oxen API...")
      generated_url = call_oxen_api(image_url, model_config, prompt)

      update_status("saving", "Downloading and saving results...")
      print("📥 Downloading generated image...")
      print(f"   Generated URL: {generated_url}")
      generated_image = download_image_to_pil(generated_url)

    # For local inference, update status to saving now
    if is_local:
      update_status("saving", "Saving results...")

    # Extract quadrants from generated image and save to database
    print("💾 Saving generated quadrants to database...")

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
          print(f"   ✓ Saved generation for ({qx}, {qy})")
          saved_count += 1
        else:
          print(f"   ⚠️ Failed to save generation for ({qx}, {qy})")
      else:
        print(f"   ⏭️ Skipped padding quadrant ({qx}, {qy})")

    update_status("complete", f"Generated {saved_count} quadrant(s)")
    return {
      "success": True,
      "message": f"Generated {saved_count} quadrant{'s' if saved_count != 1 else ''}",
      "quadrants": list(primary_quadrants),
    }

  finally:
    # Clean up temp file if it was created (only for Oxen API path)
    if template_path is not None:
      template_path.unlink(missing_ok=True)
