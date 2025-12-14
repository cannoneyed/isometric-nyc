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

