"""
Bounds utilities for loading and managing geographic boundary files.
"""

import json
from pathlib import Path
from typing import Any

# Default bounds directory
BOUNDS_DIR = Path(__file__).parent

# Default NYC boundary file
DEFAULT_BOUNDS_FILE = BOUNDS_DIR / "nyc.json"


def load_bounds(bounds_path: Path | str | None = None) -> dict[str, Any]:
  """
  Load a boundary GeoJSON file.

  Args:
      bounds_path: Path to the bounds JSON file. If None, loads the default
                  NYC boundary.

  Returns:
      GeoJSON dictionary with the boundary features.

  Raises:
      FileNotFoundError: If the bounds file doesn't exist.
      json.JSONDecodeError: If the file is not valid JSON.
  """
  if bounds_path is None:
    bounds_path = DEFAULT_BOUNDS_FILE
  elif isinstance(bounds_path, str):
    bounds_path = Path(bounds_path)

  if not bounds_path.exists():
    raise FileNotFoundError(f"Bounds file not found: {bounds_path}")

  with open(bounds_path) as f:
    return json.load(f)


def get_bounds_dir() -> Path:
  """Get the path to the bounds directory."""
  return BOUNDS_DIR


def list_bounds_files() -> list[Path]:
  """List all available bounds JSON files."""
  return list(BOUNDS_DIR.glob("*.json"))


def save_bounds(
  bounds: dict[str, Any],
  name: str,
  bounds_dir: Path | None = None,
) -> Path:
  """
  Save a boundary GeoJSON to a file.

  Args:
      bounds: GeoJSON dictionary to save.
      name: Name for the bounds file (without .json extension).
      bounds_dir: Directory to save to. Defaults to the standard bounds dir.

  Returns:
      Path to the saved file.
  """
  if bounds_dir is None:
    bounds_dir = BOUNDS_DIR

  # Ensure .json extension
  if not name.endswith(".json"):
    name = f"{name}.json"

  output_path = bounds_dir / name

  with open(output_path, "w") as f:
    json.dump(bounds, f, indent=2)

  return output_path
