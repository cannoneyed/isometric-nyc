# Image quadrant splitting script
_Exported on 12/14/2025 at 07:50:52 EST from Cursor (2.1.39)_

---

**User**

Write a quick script in src/isometric_nyc called "split_into_quadrants.py" that takes a squre image and splits it into 4 quadrants (tl, tr, bl, br) and saves them in the same dir as the original image as <name>_tl.png, <name>_tr.png, etc

---

**Cursor**

```python
"""Split a square image into 4 quadrants (tl, tr, bl, br)."""

import argparse
from pathlib import Path

from PIL import Image


def split_into_quadrants(image_path: str) -> list[Path]:
    """
    Split a square image into 4 quadrants and save them.

    Args:
        image_path: Path to the input image

    Returns:
        List of paths to the saved quadrant images
    """
    path = Path(image_path)
    img = Image.open(path)

    width, height = img.size
    if width != height:
        raise ValueError(f"Image must be square, got {width}x{height}")

    half = width // 2

    # Define quadrant crops: (left, upper, right, lower)
    quadrants = {
        "tl": (0, 0, half, half),
        "tr": (half, 0, width, half),
        "bl": (0, half, half, height),
        "br": (half, half, width, height),
    }

    output_paths: list[Path] = []
    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    for name, box in quadrants.items():
        quadrant_img = img.crop(box)
        output_path = parent / f"{stem}_{name}{suffix}"
        quadrant_img.save(output_path)
        output_paths.append(output_path)
        print(f"Saved: {output_path}")

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a square image into 4 quadrants"
    )
    parser.add_argument("image", help="Path to the input image")
    args = parser.parse_args()

    split_into_quadrants(args.image)


if __name__ == "__main__":
    main()

```

Done! Created `src/isometric_nyc/split_into_quadrants.py`. You can run it with:

```bash
uv run python src/isometric_nyc/split_into_quadrants.py path/to/image.png
```

The script will:
1. Validate the image is square
2. Split it into 4 equal quadrants
3. Save each as `<name>_tl.png`, `<name>_tr.png`, `<name>_bl.png`, `<name>_br.png` in the same directory

```shellscript
uv run python src/isometric_nyc/split_into_quadrants.py path/to/image.png
```

