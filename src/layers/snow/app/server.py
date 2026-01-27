"""
Snow Demo Backend Server.

Serves daytime (input) tile images and nighttime (generation) images from disk.
Provides a simple API for the day/night comparison viewer.

Usage:
  uv run python src/layers/snow/app/server.py \
    --dataset_dir synthetic_data/datasets/snow

  # Then open: http://localhost:5002/?x=0&y=0

The dataset directory should contain:
  - inputs/       - <x>_<y>.png daytime images
  - generations/  - <x>_<y>.png nighttime images
"""

import argparse
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory
from werkzeug.routing import BaseConverter

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
DATASET_DIR: Path | None = None
FRONTEND_DIR: Path | None = None


def get_inputs_dir() -> Path:
  """Get the inputs subdirectory."""
  assert DATASET_DIR is not None
  return DATASET_DIR / "inputs"


def get_generations_dir() -> Path:
  """Get the generations subdirectory."""
  assert DATASET_DIR is not None
  return DATASET_DIR / "generations"


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


@app.route("/api/input/<signed:x>/<signed:y>")
def get_input(x: int, y: int):
  """
  Get the daytime (input) image for tile at (x, y).

  Looks for a file named <x>_<y>.png in the inputs directory.
  """
  if DATASET_DIR is None:
    return jsonify({"error": "DATASET_DIR not configured"}), 500

  inputs_dir = get_inputs_dir()
  filename = f"{x}_{y}.png"
  file_path = inputs_dir / filename

  if not file_path.exists():
    return jsonify(
      {"error": f"Input not found: {filename}", "path": str(file_path)}
    ), 404

  # Read and serve the file directly to handle filenames with leading hyphens
  with open(file_path, "rb") as f:
    return Response(f.read(), mimetype="image/png")


@app.route("/api/generation/<signed:x>/<signed:y>")
def get_generation(x: int, y: int):
  """
  Get the nighttime (generation) image for tile at (x, y).

  Looks for a file named <x>_<y>.png in the generations directory.
  """
  if DATASET_DIR is None:
    return jsonify({"error": "DATASET_DIR not configured"}), 500

  generations_dir = get_generations_dir()
  filename = f"{x}_{y}.png"
  file_path = generations_dir / filename

  if not file_path.exists():
    return jsonify(
      {"error": f"Generation not found: {filename}", "path": str(file_path)}
    ), 404

  # Read and serve the file directly to handle filenames with leading hyphens
  with open(file_path, "rb") as f:
    return Response(f.read(), mimetype="image/png")


@app.route("/api/available-tiles")
def get_available_tiles():
  """
  Get a list of available tile coordinates.

  Returns tiles that exist in both input and generation directories.
  """
  if DATASET_DIR is None:
    return jsonify({"error": "DATASET_DIR not configured"}), 500

  inputs_dir = get_inputs_dir()
  generations_dir = get_generations_dir()

  if not inputs_dir.exists():
    return jsonify({"error": f"Inputs directory not found: {inputs_dir}"}), 404

  # Get input tiles
  input_tiles: set[tuple[int, int]] = set()
  for input_file in inputs_dir.glob("*.png"):
    try:
      parts = input_file.stem.split("_")
      if len(parts) >= 2:
        x = int(parts[0])
        y = int(parts[1])
        input_tiles.add((x, y))
    except (ValueError, IndexError):
      continue

  # Get generation tiles (if directory exists)
  generation_tiles: set[tuple[int, int]] = set()
  if generations_dir.exists():
    for gen_file in generations_dir.glob("*.png"):
      try:
        parts = gen_file.stem.split("_")
        if len(parts) >= 2:
          x = int(parts[0])
          y = int(parts[1])
          generation_tiles.add((x, y))
      except (ValueError, IndexError):
        continue

  # Find tiles that have both input and generation
  complete_tiles = list(input_tiles & generation_tiles)
  complete_tiles.sort()

  # Also return tiles that only have input (no generation yet)
  input_only = list(input_tiles - generation_tiles)
  input_only.sort()

  return jsonify(
    {
      "tiles": complete_tiles,
      "input_only": input_only,
      "generation_only": list(generation_tiles - input_tiles),
    }
  )


@app.route("/api/status")
def get_status():
  """Get server status and configuration."""
  inputs_dir = get_inputs_dir() if DATASET_DIR else None
  generations_dir = get_generations_dir() if DATASET_DIR else None

  status = {
    "dataset_dir": str(DATASET_DIR) if DATASET_DIR else None,
    "inputs_dir": str(inputs_dir) if inputs_dir else None,
    "generations_dir": str(generations_dir) if generations_dir else None,
    "inputs_dir_exists": inputs_dir.exists() if inputs_dir else False,
    "generations_dir_exists": generations_dir.exists() if generations_dir else False,
  }

  return jsonify(status)


# =============================================================================
# Main
# =============================================================================


def main():
  global DATASET_DIR, FRONTEND_DIR

  parser = argparse.ArgumentParser(
    description="Snow Demo Backend Server",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Start server with dataset directory:
  %(prog)s --dataset_dir synthetic_data/datasets/snow

  # Then open: http://localhost:5002/?x=0&y=0

The dataset directory should contain:
  - inputs/       - <x>_<y>.png daytime images
  - generations/  - <x>_<y>.png nighttime images
""",
  )
  parser.add_argument(
    "--dataset_dir",
    type=Path,
    required=True,
    help="Directory containing inputs/ and generations/ subdirectories",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=5002,
    help="Port to run server on (default: 5002)",
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
  DATASET_DIR = args.dataset_dir.resolve()
  inputs_dir = DATASET_DIR / "inputs"
  generations_dir = DATASET_DIR / "generations"

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
  if not DATASET_DIR.exists():
    print(f"❌ Error: Dataset directory not found: {DATASET_DIR}")
    return 1

  if not inputs_dir.exists():
    print(f"⚠️  Warning: Inputs directory not found: {inputs_dir}")

  if not generations_dir.exists():
    print(f"⚠️  Warning: Generations directory not found: {generations_dir}")

  print(f"\n{'=' * 60}")
  print("⛄ Snow Mode Demo Server")
  print(f"{'=' * 60}")
  print(f"   Dataset directory: {DATASET_DIR}")
  print(f"   Inputs (day): {inputs_dir}")
  print(f"   Generations (night): {generations_dir}")
  print(f"   Frontend directory: {FRONTEND_DIR}")
  print(f"   Server: http://{args.host}:{args.port}")
  print(f"{'=' * 60}\n")

  print("API Endpoints:")
  print("   GET /api/input/<x>/<y>       - Get daytime input image")
  print("   GET /api/generation/<x>/<y>  - Get nighttime generation image")
  print("   GET /api/available-tiles     - List available tiles")
  print("   GET /api/status              - Server status")
  print()

  app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
  exit(main() or 0)
