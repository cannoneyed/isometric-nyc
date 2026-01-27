"""
Water Shader Demo Backend Server.

Serves tile images from SQLite database and mask images from disk.

Usage:
  uv run python src/water_shader_demo/server.py \
    --mask_dir synthetic_data/datasets/water_masks/generations \
    --generations_dir generations/nyc

  # Then open: http://localhost:5001/?x=0&y=0
"""

import argparse
import io
import sqlite3
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory
from PIL import Image
from werkzeug.routing import BaseConverter

# =============================================================================
# Database Utilities (from shared.py)
# =============================================================================


def get_quadrant_generation(conn: sqlite3.Connection, x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant at position (x, y)."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
    (x, y),
  )
  row = cursor.fetchone()
  return row[0] if row else None


def png_bytes_to_image(png_bytes: bytes) -> Image.Image:
  """Convert PNG bytes to a PIL Image."""
  return Image.open(io.BytesIO(png_bytes))


def image_to_png_bytes(img: Image.Image) -> bytes:
  """Convert a PIL Image to PNG bytes."""
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  return buffer.getvalue()


def stitch_quadrants_to_tile(
  quadrants: dict[tuple[int, int], Image.Image],
) -> Image.Image:
  """
  Stitch 4 quadrant images into a single tile image.

  Args:
    quadrants: Dict mapping (dx, dy) offset to the quadrant image:
      (0, 0) = top-left
      (1, 0) = top-right
      (0, 1) = bottom-left
      (1, 1) = bottom-right

  Returns:
    Combined tile image
  """
  # Assume all quadrants are the same size
  q_size = quadrants[(0, 0)].size[0]
  tile_size = q_size * 2

  tile = Image.new("RGBA", (tile_size, tile_size))

  for (dx, dy), quad_img in quadrants.items():
    tile.paste(quad_img, (dx * q_size, dy * q_size))

  return tile


# =============================================================================
# Flask App
# =============================================================================


class SignedIntConverter(BaseConverter):
  """URL converter that handles signed (positive and negative) integers."""

  regex = r"-?\d+"

  def to_python(self, value: str) -> int:
    return int(value)

  def to_url(self, value: int) -> str:
    return str(value)


app = Flask(__name__, static_folder=None)
app.url_map.converters["signed"] = SignedIntConverter

# Global config - set via CLI args
MASK_DIR: Path | None = None
GENERATIONS_DIR: Path | None = None
FRONTEND_DIR: Path | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the generations database."""
  if GENERATIONS_DIR is None:
    raise RuntimeError("GENERATIONS_DIR not configured")
  db_path = GENERATIONS_DIR / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")
  return sqlite3.connect(db_path)


@app.route("/")
def index():
  """Serve the main index.html."""
  if FRONTEND_DIR is None:
    return "Frontend not configured", 500
  return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/assets/<path:filename>")
def serve_assets(filename: str):
  """Serve static assets."""
  if FRONTEND_DIR is None:
    return "Frontend not configured", 500
  return send_from_directory(FRONTEND_DIR / "assets", filename)


@app.route("/api/tile/<signed:x>/<signed:y>")
def get_tile(x: int, y: int):
  """
  Get a 2x2 tile image starting at (x, y) from the generations database.

  Returns a 1024x1024 PNG image composed of 4 quadrants:
    (x, y)     (x+1, y)
    (x, y+1)   (x+1, y+1)
  """
  try:
    conn = get_db_connection()

    # Load all 4 quadrants
    quadrant_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR
    quadrants: dict[tuple[int, int], Image.Image] = {}

    for dx, dy in quadrant_offsets:
      qx, qy = x + dx, y + dy
      gen_bytes = get_quadrant_generation(conn, qx, qy)

      if gen_bytes is None:
        conn.close()
        return jsonify({"error": f"Quadrant ({qx}, {qy}) not found in database"}), 404

      quadrants[(dx, dy)] = png_bytes_to_image(gen_bytes)

    conn.close()

    # Stitch quadrants into tile
    tile_image = stitch_quadrants_to_tile(quadrants)

    # Convert to PNG bytes
    png_bytes = image_to_png_bytes(tile_image)

    return Response(png_bytes, mimetype="image/png")

  except FileNotFoundError as e:
    return jsonify({"error": str(e)}), 404
  except Exception as e:
    return jsonify({"error": str(e)}), 500


@app.route("/api/mask/<signed:x>/<signed:y>")
def get_mask(x: int, y: int):
  """
  Get the water mask image for tile at (x, y).

  Looks for a file named <x>_<y>.png in the mask directory.
  """
  if MASK_DIR is None:
    return jsonify({"error": "MASK_DIR not configured"}), 500

  # Try both naming conventions
  mask_filename = f"{x}_{y}.png"
  mask_path = MASK_DIR / mask_filename

  if not mask_path.exists():
    return jsonify(
      {"error": f"Mask not found: {mask_filename}", "path": str(mask_path)}
    ), 404

  return send_from_directory(MASK_DIR, mask_filename, mimetype="image/png")


@app.route("/api/available-tiles")
def get_available_tiles():
  """
  Get a list of available tile coordinates.

  Returns tiles where all 4 quadrants have generations.
  """
  try:
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all quadrants that have generations
    cursor.execute(
      "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
    )
    generated_quadrants = set(cursor.fetchall())
    conn.close()

    # Find all possible 2x2 tiles (where all 4 quadrants exist)
    available_tiles: list[tuple[int, int]] = []

    for qx, qy in generated_quadrants:
      # Check if this quadrant could be the top-left of a complete tile
      required = [(qx, qy), (qx + 1, qy), (qx, qy + 1), (qx + 1, qy + 1)]
      if all(q in generated_quadrants for q in required):
        available_tiles.append((qx, qy))

    # Sort by x, then y
    available_tiles.sort()

    return jsonify({"tiles": available_tiles})

  except Exception as e:
    return jsonify({"error": str(e)}), 500


@app.route("/api/available-masks")
def get_available_masks():
  """
  Get a list of available mask coordinates.

  Returns coordinates parsed from <x>_<y>.png filenames in the mask directory.
  """
  if MASK_DIR is None:
    return jsonify({"error": "MASK_DIR not configured"}), 500

  if not MASK_DIR.exists():
    return jsonify({"error": f"Mask directory not found: {MASK_DIR}"}), 404

  masks: list[tuple[int, int]] = []

  for mask_file in MASK_DIR.glob("*.png"):
    try:
      # Parse <x>_<y>.png format
      parts = mask_file.stem.split("_")
      if len(parts) >= 2:
        x = int(parts[0])
        y = int(parts[1])
        masks.append((x, y))
    except (ValueError, IndexError):
      # Skip files that don't match the expected format
      continue

  # Sort by x, then y
  masks.sort()

  return jsonify({"masks": masks})


@app.route("/api/status")
def get_status():
  """Get server status and configuration."""
  status = {
    "mask_dir": str(MASK_DIR) if MASK_DIR else None,
    "generations_dir": str(GENERATIONS_DIR) if GENERATIONS_DIR else None,
    "mask_dir_exists": MASK_DIR.exists() if MASK_DIR else False,
    "generations_dir_exists": GENERATIONS_DIR.exists() if GENERATIONS_DIR else False,
  }

  # Check database
  if GENERATIONS_DIR:
    db_path = GENERATIONS_DIR / "quadrants.db"
    status["db_exists"] = db_path.exists()

  return jsonify(status)


# =============================================================================
# Main
# =============================================================================


def main():
  global MASK_DIR, GENERATIONS_DIR, FRONTEND_DIR

  parser = argparse.ArgumentParser(
    description="Water Shader Demo Backend Server",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Start server with mask and generation directories:
  %(prog)s --mask_dir synthetic_data/datasets/water_masks/generations \\
           --generations_dir generations/nyc

  # Then open: http://localhost:5001/?x=0&y=0
""",
  )
  parser.add_argument(
    "--mask_dir",
    type=Path,
    required=True,
    help="Directory containing <x>_<y>.png mask images",
  )
  parser.add_argument(
    "--generations_dir",
    type=Path,
    required=True,
    help="Directory containing quadrants.db SQLite database",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=5001,
    help="Port to run server on (default: 5001)",
  )
  parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--frontend",
    type=Path,
    default=None,
    help="Path to built frontend directory (default: auto-detect from dist/)",
  )

  args = parser.parse_args()

  # Resolve paths
  MASK_DIR = args.mask_dir.resolve()
  GENERATIONS_DIR = args.generations_dir.resolve()

  # Find frontend directory
  script_dir = Path(__file__).parent.resolve()
  if args.frontend:
    FRONTEND_DIR = args.frontend.resolve()
  else:
    # Try dist/ directory (built frontend)
    dist_dir = script_dir / "dist"
    if dist_dir.exists():
      FRONTEND_DIR = dist_dir
    else:
      print("⚠️  No frontend dist/ directory found. Run 'bun run build' first.")
      print("   The API endpoints will still work.")
      FRONTEND_DIR = None

  # Validate paths
  if not MASK_DIR.exists():
    print(f"⚠️  Warning: Mask directory not found: {MASK_DIR}")

  if not GENERATIONS_DIR.exists():
    print(f"❌ Error: Generations directory not found: {GENERATIONS_DIR}")
    return 1

  db_path = GENERATIONS_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print(f"\n{'=' * 60}")
  print("🌊 Water Shader Demo Server")
  print(f"{'=' * 60}")
  print(f"   Mask directory: {MASK_DIR}")
  print(f"   Generations directory: {GENERATIONS_DIR}")
  print(f"   Frontend directory: {FRONTEND_DIR}")
  print(f"   Server: http://{args.host}:{args.port}")
  print(f"{'=' * 60}\n")

  print("API Endpoints:")
  print("   GET /api/tile/<x>/<y>        - Get 2x2 tile image")
  print("   GET /api/mask/<x>/<y>        - Get water mask image")
  print("   GET /api/available-tiles     - List available tiles")
  print("   GET /api/available-masks     - List available masks")
  print("   GET /api/status              - Server status")
  print()

  app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
  exit(main() or 0)
