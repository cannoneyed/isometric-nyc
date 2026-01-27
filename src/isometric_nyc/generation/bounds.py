"""
Utilities for loading and saving GeoJSON bounds files.

Bounds files define geographic polygons used for clipping tile exports.
"""

import json
from pathlib import Path
from typing import Any


def get_bounds_dir() -> Path:
    """Get the default bounds directory."""
    return Path("bounds")


def load_bounds(path: Path | str | None = None) -> dict[str, Any]:
    """
    Load a GeoJSON bounds file.

    Args:
        path: Path to the bounds file. If None, loads the default NYC boundary.

    Returns:
        GeoJSON dict with FeatureCollection containing the bounds polygon.
    """
    if path is None:
        # Load default NYC boundary from bundled data
        default_path = Path(__file__).parent / "data" / "nyc_boundary.json"
        if not default_path.exists():
            # Return empty feature collection if no default exists
            return {"type": "FeatureCollection", "features": []}
        path = default_path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bounds file not found: {path}")

    with open(path) as f:
        return json.load(f)


def save_bounds(geojson: dict[str, Any], name: str, bounds_dir: Path | None = None) -> Path:
    """
    Save a GeoJSON bounds file.

    Args:
        geojson: GeoJSON dict to save
        name: Name for the bounds file (without extension)
        bounds_dir: Directory to save to (default: bounds/)

    Returns:
        Path to the saved file
    """
    if bounds_dir is None:
        bounds_dir = get_bounds_dir()

    bounds_dir = Path(bounds_dir)
    bounds_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    if not safe_name:
        safe_name = "bounds"

    output_path = bounds_dir / f"{safe_name}.json"

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    return output_path
