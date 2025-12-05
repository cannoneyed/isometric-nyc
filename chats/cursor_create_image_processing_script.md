# Create image processing script
_Exported on 12/4/2025 at 19:29:08 EST from Cursor (2.1.39)_

---

**User**

Create a copy of @src/isometric_nyc/synthetic_data/create_inline_examples.py that processes images in a `tile_dir` (argument) and does the following:



creates a 1024x1024 image where the left half is from `generation.png` and the right half is from `render.png`, and saves it to the same dir as `infill.png`

---

**Cursor**

```python
"""
Create infill examples by compositing generation and render images.

This script takes a tile directory (or a directory of tile directories), loads
'generation.png' and 'render.png' from each, and composites them into a
single 1024x1024 image named 'infill.png'.

The output image consists of:
- Left half: The left half of 'generation.png'
- Right half: The right half of 'render.png'

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_infill_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def create_infill_image(tile_dir: Path) -> None:
    """
    Create an infill image for a single tile directory.
    
    Output: 1024x1024 'infill.png'
    Left half (0-512): Left half of generation.png
    Right half (512-1024): Right half of render.png
    """
    images = {
        "generation": tile_dir / "generation.png",
        "render": tile_dir / "render.png",
    }

    # Check if required images exist
    missing = [name for name, path in images.items() if not path.exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return

    try:
        # Open images
        img_gen = Image.open(images["generation"])
        img_render = Image.open(images["render"])

        target_size = (1024, 1024)
        
        # Resize images to target size if they aren't already
        if img_gen.size != target_size:
            img_gen = img_gen.resize(target_size, Image.Resampling.LANCZOS)
        
        if img_render.size != target_size:
            img_render = img_render.resize(target_size, Image.Resampling.LANCZOS)

        # Create final image
        final_image = Image.new("RGB", target_size)

        # Crop and paste
        # Left half of generation (0, 0, 512, 1024)
        left_half_gen = img_gen.crop((0, 0, 512, 1024))
        final_image.paste(left_half_gen, (0, 0))

        # Right half of render (512, 0, 1024, 1024)
        right_half_render = img_render.crop((512, 0, 1024, 1024))
        final_image.paste(right_half_render, (512, 0))

        # Save
        output_path = tile_dir / "infill.png"
        final_image.save(output_path)
        print(f"✅ Saved infill composition to {output_path}")

    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Composite generation and render images into split-screen infill examples"
    )
    parser.add_argument(
        "--tile_dir",
        type=Path,
        required=True,
        help="Path to tile generation directory (can be single tile or parent of multiple)",
    )

    args = parser.parse_args()

    if not args.tile_dir.exists():
        print(f"❌ Directory not found: {args.tile_dir}")
        sys.exit(1)

    # Determine directories to process
    tile_dirs: List[Path] = []

    if (args.tile_dir / "view.json").exists() or (args.tile_dir / "generation.png").exists():
        # Single tile directory
        # We check for generation.png as well since that's critical for this script
        print(f"📂 Found single tile directory: {args.tile_dir}")
        tile_dirs.append(args.tile_dir)
    else:
        # Parent directory
        print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
        subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
        for d in subdirs:
            # Check if it looks like a tile dir
            if (d / "view.json").exists() or (d / "generation.png").exists():
                tile_dirs.append(d)

        print(f"   Found {len(tile_dirs)} potential tile directories.")

    if not tile_dirs:
        print("❌ No tile directories found to process.")
        sys.exit(0)

    print("=" * 60)
    print(f"🖼️  CREATING INFILL EXAMPLES - Processing {len(tile_dirs)} directories")
    print("=" * 60)

    for d in tile_dirs:
        create_infill_image(d)

    print("\n" + "=" * 60)
    print("✨ COMPOSITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

The script has been created at `src/isometric_nyc/synthetic_data/create_infill_examples.py`. It iterates through the specified directory (or parent directory of tiles), checks for `generation.png` and `render.png`, and composites them into a 1024x1024 `infill.png` as requested.

You can run it using:
```bash
uv run python src/isometric_nyc/synthetic_data/create_infill_examples.py --tile_dir <path_to_tiles>
```

```shellscript
uv run python src/isometric_nyc/synthetic_data/create_infill_examples.py --tile_dir <path_to_tiles>
```

