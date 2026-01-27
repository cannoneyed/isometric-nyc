"""
Shared utilities for oxen sync operations.
"""

import csv
import hashlib
from pathlib import Path


def compute_hash(blob: bytes) -> str:
  """
  Compute a short hash for image content.

  Uses MD5 and returns the first 8 characters for a compact identifier
  that's still sufficiently unique for our use case.

  Args:
    blob: Raw bytes of the image

  Returns:
    8-character hex hash string
  """
  return hashlib.md5(blob).hexdigest()[:8]


def format_filename(x: int, y: int, hash_str: str) -> str:
  """
  Format a filename for a quadrant or render image.

  Args:
    x: Quadrant x coordinate
    y: Quadrant y coordinate
    hash_str: Content hash

  Returns:
    Formatted filename like "0012_0045_a1b2c3d4.png"
  """
  return f"{x:04d}_{y:04d}_{hash_str}.png"


def parse_filename(filename: str) -> tuple[int, int, str] | None:
  """
  Parse a filename to extract coordinates and hash.

  Args:
    filename: Filename like "0012_0045_a1b2c3d4.png"

  Returns:
    Tuple of (x, y, hash) or None if invalid format
  """
  try:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 3:
      return None
    x = int(parts[0])
    y = int(parts[1])
    hash_str = parts[2]
    return x, y, hash_str
  except (ValueError, IndexError):
    return None


def write_csv(filepath: Path, rows: list[dict]) -> None:
  """
  Write rows to a CSV file.

  Args:
    filepath: Path to output CSV
    rows: List of dicts with consistent keys
  """
  if not rows:
    # Write empty CSV with headers
    fieldnames = ["x", "y", "render", "quadrant", "hash_render", "hash_quadrant"]
    with open(filepath, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
    return

  fieldnames = list(rows[0].keys())
  with open(filepath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def read_csv(filepath: Path) -> list[dict]:
  """
  Read rows from a CSV file.

  Args:
    filepath: Path to CSV file

  Returns:
    List of dicts, one per row
  """
  with open(filepath, newline="") as f:
    reader = csv.DictReader(f)
    return list(reader)


def write_readme(filepath: Path, quadrant_count: int, generations_dir: str) -> None:
  """
  Generate a README.md for the dataset.

  Args:
    filepath: Path to output README
    quadrant_count: Number of quadrants in the dataset
    generations_dir: Name of the source generations directory
  """
  from datetime import datetime

  content = f"""# Isometric NYC Generations Dataset

This dataset contains generated isometric pixel art tiles of New York City.

## Statistics

- **Quadrants**: {quadrant_count}
- **Source**: `{generations_dir}`
- **Last Updated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Structure

```
renders/           # 3D reference renders (PNG)
quadrants/         # Generated pixel art tiles (PNG)
generations.csv    # Manifest with all tile mappings
README.md          # This file
```

## File Naming

Files are named: `<x>_<y>_<hash>.png`

- `x`, `y`: Quadrant coordinates (4-digit zero-padded)
- `hash`: 8-character content hash for change detection

## CSV Format

The `generations.csv` file contains:

| Column | Description |
|--------|-------------|
| x | Quadrant X coordinate |
| y | Quadrant Y coordinate |
| render | Path to render PNG (if exists) |
| quadrant | Path to generation PNG (if exists) |
| hash_render | Content hash of render |
| hash_quadrant | Content hash of generation |

## Usage

Import this dataset into your local generations database:

```bash
uv run import_from_oxen --generations_dir {generations_dir} --oxen_dataset <this-dataset-id>
```
"""
  filepath.write_text(content)
