# Create image composition script
_Exported on 12/4/2025 at 14:25:05 EST from Cursor (2.1.39)_

---

**User**

Create a new script in src/isometric_nyc/synthetic_data called `create_inline_examples.png` that takes 3 images from each tile_dir (whitebox.png, render.png, and generation.png) and composites them next to each other in a single 21:9 image (padding with whitespace if necessary) - it should save the composite image in the tile dir as `composition.png`) and should take a single tile_dir parameter that will iterate over all subdirs if there's no view.json found in the tile dir (sort of like how @src/isometric_nyc/export_views.py works

---

**Cursor**

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them side-by-side into a single 21:9 image named 'composition.png',
padding with whitespace if necessary to maintain the aspect ratio.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image

# Target aspect ratio (21:9)
ASPECT_RATIO = 21 / 9


def composite_images(tile_dir: Path) -> None:
    """
    Composite images for a single tile directory.
    """
    images = {
        "whitebox": tile_dir / "whitebox.png",
        "render": tile_dir / "render.png",
        "generation": tile_dir / "generation.png",
    }

    # Check if all images exist
    missing = [name for name, path in images.items() if not path.exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return

    try:
        # Open images
        imgs = [Image.open(images[name]) for name in ["whitebox", "render", "generation"]]
        
        # Calculate dimensions for the stitched image
        total_width = sum(img.width for img in imgs)
        max_height = max(img.height for img in imgs)
        
        # Create stitched image
        stitched = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        x_offset = 0
        for img in imgs:
            # Center vertically if heights differ
            y_offset = (max_height - img.height) // 2
            stitched.paste(img, (x_offset, y_offset))
            x_offset += img.width
        
        # Calculate final dimensions for 21:9 aspect ratio
        current_ratio = total_width / max_height
        
        if current_ratio > ASPECT_RATIO:
            # Too wide, need to increase height (pad top/bottom)
            final_width = total_width
            final_height = int(total_width / ASPECT_RATIO)
        else:
            # Too tall (or narrow), need to increase width (pad left/right)
            final_height = max_height
            final_width = int(max_height * ASPECT_RATIO)
            
        # Create final composite
        final_image = Image.new("RGB", (final_width, final_height), (255, 255, 255))
        
        # Center stitched image on final canvas
        final_x = (final_width - total_width) // 2
        final_y = (final_height - max_height) // 2
        
        final_image.paste(stitched, (final_x, final_y))
        
        # Save
        output_path = tile_dir / "composition.png"
        final_image.save(output_path)
        print(f"✅ Saved composition to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Composite tile images into 21:9 examples"
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
    
    if (args.tile_dir / "view.json").exists():
        # Single tile directory
        print(f"📂 Found single tile directory: {args.tile_dir}")
        tile_dirs.append(args.tile_dir)
    else:
        # Parent directory
        print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
        # Look for subdirectories that might be tiles (checking for view.json is safest)
        # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
        # export_views checks for view.json in subdirs.
        subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
        for d in subdirs:
            # We'll check if it looks like a tile dir (has view.json OR has the images we need)
            # Sticking to view.json as per export_views logic reference is safest, 
            # but checking for the images is also valid if view.json is missing but images exist.
            # Let's check for view.json OR existence of at least one of the images to be permissive.
            if (d / "view.json").exists() or (d / "whitebox.png").exists():
                tile_dirs.append(d)
        
        print(f"   Found {len(tile_dirs)} potential tile directories.")

    if not tile_dirs:
        print("❌ No tile directories found to process.")
        sys.exit(0)

    print("=" * 60)
    print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
    print("=" * 60)

    for i, d in enumerate(tile_dirs):
        composite_images(d)

    print("\n" + "=" * 60)
    print("✨ COMPOSITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

---

**User**

Can you modify the script to use a 1:1 ratio and put the square images in each quadrant?

---

**Cursor**

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path) -> None:
    """
    Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
    Layout:
    [Whitebox] [Render]
    [Generation] [Empty]
    """
    images = {
        "whitebox": tile_dir / "whitebox.png",
        "render": tile_dir / "render.png",
        "generation": tile_dir / "generation.png",
    }

    # Check if all images exist
    missing = [name for name, path in images.items() if not path.exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return

    try:
        # Open images
        img_whitebox = Image.open(images["whitebox"])
        img_render = Image.open(images["render"])
        img_generation = Image.open(images["generation"])
        
        imgs = [img_whitebox, img_render, img_generation]
        
        # Determine quadrant size
        # We use the maximum dimension found across all images to ensure square quadrants that fit everything
        max_dim = max(max(img.width, img.height) for img in imgs)
        
        # Final image size (2x2 grid)
        final_size = max_dim * 2
        
        # Create final square composite
        final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))
        
        # Paste locations
        # Quadrant 1 (Top-Left): Whitebox
        # Quadrant 2 (Top-Right): Render
        # Quadrant 3 (Bottom-Left): Generation
        # Quadrant 4 (Bottom-Right): Empty
        
        positions = [
            (0, 0),             # Top-Left
            (max_dim, 0),       # Top-Right
            (0, max_dim),       # Bottom-Left
        ]
        
        for img, (x_offset, y_offset) in zip(imgs, positions):
            # Center image within its quadrant
            x = x_offset + (max_dim - img.width) // 2
            y = y_offset + (max_dim - img.height) // 2
            final_image.paste(img, (x, y))
        
        # Save
        output_path = tile_dir / "composition.png"
        final_image.save(output_path)
        print(f"✅ Saved composition to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Composite tile images into 21:9 examples"
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
    
    if (args.tile_dir / "view.json").exists():
        # Single tile directory
        print(f"📂 Found single tile directory: {args.tile_dir}")
        tile_dirs.append(args.tile_dir)
    else:
        # Parent directory
        print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
        # Look for subdirectories that might be tiles (checking for view.json is safest)
        # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
        # export_views checks for view.json in subdirs.
        subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
        for d in subdirs:
            # We'll check if it looks like a tile dir (has view.json OR has the images we need)
            # Sticking to view.json as per export_views logic reference is safest, 
            # but checking for the images is also valid if view.json is missing but images exist.
            # Let's check for view.json OR existence of at least one of the images to be permissive.
            if (d / "view.json").exists() or (d / "whitebox.png").exists():
                tile_dirs.append(d)
        
        print(f"   Found {len(tile_dirs)} potential tile directories.")

    if not tile_dirs:
        print("❌ No tile directories found to process.")
        sys.exit(0)

    print("=" * 60)
    print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
    print("=" * 60)

    for i, d in enumerate(tile_dirs):
        composite_images(d)

    print("\n" + "=" * 60)
    print("✨ COMPOSITION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


```

The script has been successfully modified and executed. It now composites the `whitebox.png`, `render.png`, and `generation.png` images into a 1:1 square image with the images arranged in the first three quadrants (Top-Left, Top-Right, Bottom-Left), leaving the fourth quadrant (Bottom-Right) empty. The execution confirmed it processed 22 directories (skipping one with missing files) and saved `composition.png` in each valid directories.

I'm done with the task.



---

**User**

ensure that the `generation.png` is stretched to the full quadrant

---

**Cursor**

```python
        # Determine quadrant size
        # We use the maximum dimension found across all images to ensure square quadrants that fit everything
        max_dim = max(max(img.width, img.height) for img in imgs)
        
        # Resize generation image to fill the quadrant exactly
        if img_generation.width != max_dim or img_generation.height != max_dim:
            img_generation = img_generation.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
            imgs[2] = img_generation

        # Final image size (2x2 grid)
        final_size = max_dim * 2
        
        # Create final square composite
        final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))
        
        # Paste locations
        # Quadrant 1 (Top-Left): Whitebox
        # Quadrant 2 (Top-Right): Render
        # Quadrant 3 (Bottom-Left): Generation
        # Quadrant 4 (Bottom-Right): Empty
        
        positions = [
            (0, 0),             # Top-Left
            (max_dim, 0),       # Top-Right
            (0, max_dim),       # Bottom-Left
        ]
        
        for img, (x_offset, y_offset) in zip(imgs, positions):
            # Center image within its quadrant (except generation which is now full size)
            x = x_offset + (max_dim - img.width) // 2
            y = y_offset + (max_dim - img.height) // 2
            final_image.paste(img, (x, y))
```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout:
  [Whitebox] [Render]
  [Generation] [Empty]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if all images exist
  missing = [name for name, path in images.items() if not path.exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])
    img_generation = Image.open(images["generation"])

    imgs = [img_whitebox, img_render, img_generation]

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(max(img.width, img.height) for img in imgs)

    # Resize generation image to fill the quadrant exactly
    if img_generation.width != max_dim or img_generation.height != max_dim:
      img_generation = img_generation.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
      imgs[2] = img_generation

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Paste locations
    # Quadrant 1 (Top-Left): Whitebox
    # Quadrant 2 (Top-Right): Render
    # Quadrant 3 (Bottom-Left): Generation
    # Quadrant 4 (Bottom-Right): Empty

    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
      (0, max_dim),  # Bottom-Left
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      # Center image within its quadrant
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
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

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

The script has been updated to resize `generation.png` to the full quadrant size. The composites have been regenerated.

I'm done with the task.

task.



---

**User**

Can you also add a flag to leave the bottom left quadrant blank?

---

**Cursor**

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path, blank_bottom_left: bool = False) -> None:
    """
    Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
    Layout:
    [Whitebox] [Render]
    [Generation] [Empty]
    """
    images = {
        "whitebox": tile_dir / "whitebox.png",
        "render": tile_dir / "render.png",
        "generation": tile_dir / "generation.png",
    }

    # Check if required images exist
    required = ["whitebox", "render"]
    if not blank_bottom_left:
        required.append("generation")
    
    missing = [name for name in required if not images[name].exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return

    try:
        # Open images
        img_whitebox = Image.open(images["whitebox"])
        img_render = Image.open(images["render"])
        
        if not blank_bottom_left:
            img_generation = Image.open(images["generation"])
            imgs = [img_whitebox, img_render, img_generation]
        else:
            imgs = [img_whitebox, img_render]
        
        # Determine quadrant size
        # We use the maximum dimension found across all images to ensure square quadrants that fit everything
        max_dim = max(max(img.width, img.height) for img in imgs)
        
        # Resize generation image to fill the quadrant exactly (if present)
        if not blank_bottom_left and (img_generation.width != max_dim or img_generation.height != max_dim):
            img_generation = img_generation.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
            imgs[2] = img_generation

        # Final image size (2x2 grid)
        final_size = max_dim * 2
        
        # Create final square composite
        final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))
        
        # Paste locations
        # Quadrant 1 (Top-Left): Whitebox
        # Quadrant 2 (Top-Right): Render
        # Quadrant 3 (Bottom-Left): Generation (or blank)
        # Quadrant 4 (Bottom-Right): Empty
        
        positions = [
            (0, 0),             # Top-Left
            (max_dim, 0),       # Top-Right
        ]
        
        if not blank_bottom_left:
            positions.append((0, max_dim)) # Bottom-Left
        
        for img, (x_offset, y_offset) in zip(imgs, positions):
            # Center image within its quadrant (except generation which is now full size)
            x = x_offset + (max_dim - img.width) // 2
            y = y_offset + (max_dim - img.height) // 2
            final_image.paste(img, (x, y))
        
        # Save
        output_path = tile_dir / "composition.png"
        final_image.save(output_path)
        print(f"✅ Saved composition to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
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

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path, blank_bottom_left: bool = False) -> None:
    """
    Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
    Layout:
    [Whitebox] [Render]
    [Generation] [Empty]
    """
    images = {
        "whitebox": tile_dir / "whitebox.png",
        "render": tile_dir / "render.png",
        "generation": tile_dir / "generation.png",
    }

    # Check if required images exist
    required = ["whitebox", "render"]
    if not blank_bottom_left:
        required.append("generation")
    
    missing = [name for name in required if not images[name].exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return

    try:
        # Open images
        img_whitebox = Image.open(images["whitebox"])
        img_render = Image.open(images["render"])
        
        if not blank_bottom_left:
            img_generation = Image.open(images["generation"])
            imgs = [img_whitebox, img_render, img_generation]
        else:
            imgs = [img_whitebox, img_render]
        
        # Determine quadrant size
        # We use the maximum dimension found across all images to ensure square quadrants that fit everything
        max_dim = max(max(img.width, img.height) for img in imgs)
        
        # Resize generation image to fill the quadrant exactly (if present)
        if not blank_bottom_left and (img_generation.width != max_dim or img_generation.height != max_dim):
            img_generation = img_generation.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
            imgs[2] = img_generation

        # Final image size (2x2 grid)
        final_size = max_dim * 2
        
        # Create final square composite
        final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))
        
        # Paste locations
        # Quadrant 1 (Top-Left): Whitebox
        # Quadrant 2 (Top-Right): Render
        # Quadrant 3 (Bottom-Left): Generation (or blank)
        # Quadrant 4 (Bottom-Right): Empty
        
        positions = [
            (0, 0),             # Top-Left
            (max_dim, 0),       # Top-Right
        ]
        
        if not blank_bottom_left:
            positions.append((0, max_dim)) # Bottom-Left
        
        for img, (x_offset, y_offset) in zip(imgs, positions):
            # Center image within its quadrant (except generation which is now full size)
            x = x_offset + (max_dim - img.width) // 2
            y = y_offset + (max_dim - img.height) // 2
            final_image.paste(img, (x, y))
        
        # Save
        output_path = tile_dir / "composition.png"
        final_image.save(output_path)
        print(f"✅ Saved composition to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--blank-bottom-left",
    action="store_true",
    help="Leave the bottom-left quadrant blank (skip generation.png)",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path, blank_bottom_left: bool = False) -> None:
    """
    Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
    Layout:
    [Whitebox] [Render]
    [Generation] [Empty]
    """
    images = {
        "whitebox": tile_dir / "whitebox.png",
        "render": tile_dir / "render.png",
        "generation": tile_dir / "generation.png",
    }

    # Check if required images exist
    required = ["whitebox", "render"]
    if not blank_bottom_left:
        required.append("generation")
    
    missing = [name for name in required if not images[name].exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return

    try:
        # Open images
        img_whitebox = Image.open(images["whitebox"])
        img_render = Image.open(images["render"])
        
        if not blank_bottom_left:
            img_generation = Image.open(images["generation"])
            imgs = [img_whitebox, img_render, img_generation]
        else:
            imgs = [img_whitebox, img_render]
        
        # Determine quadrant size
        # We use the maximum dimension found across all images to ensure square quadrants that fit everything
        max_dim = max(max(img.width, img.height) for img in imgs)
        
        # Resize generation image to fill the quadrant exactly (if present)
        if not blank_bottom_left and (img_generation.width != max_dim or img_generation.height != max_dim):
            img_generation = img_generation.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
            imgs[2] = img_generation

        # Final image size (2x2 grid)
        final_size = max_dim * 2
        
        # Create final square composite
        final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))
        
        # Paste locations
        # Quadrant 1 (Top-Left): Whitebox
        # Quadrant 2 (Top-Right): Render
        # Quadrant 3 (Bottom-Left): Generation (or blank)
        # Quadrant 4 (Bottom-Right): Empty
        
        positions = [
            (0, 0),             # Top-Left
            (max_dim, 0),       # Top-Right
        ]
        
        if not blank_bottom_left:
            positions.append((0, max_dim)) # Bottom-Left
        
        for img, (x_offset, y_offset) in zip(imgs, positions):
            # Center image within its quadrant (except generation which is now full size)
            x = x_offset + (max_dim - img.width) // 2
            y = y_offset + (max_dim - img.height) // 2
            final_image.paste(img, (x, y))
        
        # Save
        output_path = tile_dir / "composition.png"
        final_image.save(output_path)
        print(f"✅ Saved composition to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--blank-bottom-left",
    action="store_true",
    help="Leave the bottom-left quadrant blank (skip generation.png)",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d, blank_bottom_left=args.blank_bottom_left)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

The script has been updated with the `--blank-bottom-left` flag, and the test run successfully generated composites with the bottom-left quadrant blank.

I'm done with the task.



---

**User**

OK - now add another flag to the image composition inline examples script called "infill" that changes the behavior in the following way - the bottom left quadrant should now only show the left half of the generation.png image (with white pixels on the right half) and the bottom right quadrant would have the full generation.png

---

**Cursor**

```python

def composite_images(tile_dir: Path, blank_bottom_left: bool = False, infill: bool = False) -> None:
    """
    Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
    Layout options:
    Default:
      [Whitebox] [Render]
      [Generation] [Empty]
    
    blank_bottom_left:
      [Whitebox] [Render]
      [Empty] [Empty]

    infill:
      [Whitebox] [Render]
      [Left-Half-Generation] [Generation]
    """
    images = {
        "whitebox": tile_dir / "whitebox.png",
        "render": tile_dir / "render.png",
        "generation": tile_dir / "generation.png",
    }

    # Check if required images exist
    required = ["whitebox", "render"]
    if not blank_bottom_left:
        required.append("generation")
    
    missing = [name for name in required if not images[name].exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return

    try:
        # Open images
        img_whitebox = Image.open(images["whitebox"])
        img_render = Image.open(images["render"])
        
        imgs = [img_whitebox, img_render]
        
        if not blank_bottom_left:
            img_generation = Image.open(images["generation"])
        else:
            img_generation = None
        
        # Determine quadrant size
        # We use the maximum dimension found across all images to ensure square quadrants that fit everything
        max_dim = max(img.width for img in imgs)
        if img_generation:
             max_dim = max(max_dim, max(img_generation.width, img_generation.height))
        
        # Final image size (2x2 grid)
        final_size = max_dim * 2
        
        # Create final square composite
        final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))
        
        # Prepare generation images (resize if needed)
        img_generation_full = None
        if img_generation:
             if img_generation.width != max_dim or img_generation.height != max_dim:
                  img_generation_full = img_generation.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
             else:
                  img_generation_full = img_generation

        # Paste locations and logic
        # 1. Whitebox (TL)
        # 2. Render (TR)
        positions = [
            (0, 0),             # Top-Left
            (max_dim, 0),       # Top-Right
        ]
        
        for img, (x_offset, y_offset) in zip(imgs, positions):
             x = x_offset + (max_dim - img.width) // 2
             y = y_offset + (max_dim - img.height) // 2
             final_image.paste(img, (x, y))

        # 3. Bottom-Left Logic
        if infill and img_generation_full:
            # Left half of generation, right half white
            # Create a white image
            half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            # Paste the left half of generation
            crop_box = (0, 0, max_dim // 2, max_dim)
            left_half = img_generation_full.crop(crop_box)
            half_gen.paste(left_half, (0, 0))
            
            final_image.paste(half_gen, (0, max_dim))
            
        elif not blank_bottom_left and img_generation_full:
            # Full generation
            final_image.paste(img_generation_full, (0, max_dim))

        # 4. Bottom-Right Logic
        if infill and img_generation_full:
            # Full generation in bottom right
            final_image.paste(img_generation_full, (max_dim, max_dim))
        
        # Save
        output_path = tile_dir / "composition.png"
        final_image.save(output_path)
        print(f"✅ Saved composition to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")

```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path, blank_bottom_left: bool = False, infill: bool = False) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  blank_bottom_left:
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if not blank_bottom_left or infill:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if not blank_bottom_left or infill:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not blank_bottom_left and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--blank-bottom-left",
    action="store_true",
    help="Leave the bottom-left quadrant blank (skip generation.png)",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d, blank_bottom_left=args.blank_bottom_left)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path, blank_bottom_left: bool = False, infill: bool = False) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  blank_bottom_left:
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if not blank_bottom_left or infill:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if not blank_bottom_left or infill:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not blank_bottom_left and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--blank-bottom-left",
    action="store_true",
    help="Leave the bottom-left quadrant blank (skip generation.png)",
  )
  parser.add_argument(
    "--infill",
    action="store_true",
    help="Show left-half generation in bottom-left and full generation in bottom-right",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d, blank_bottom_left=args.blank_bottom_left)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(tile_dir: Path, blank_bottom_left: bool = False, infill: bool = False) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  blank_bottom_left:
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if not blank_bottom_left or infill:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if not blank_bottom_left or infill:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not blank_bottom_left and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--blank-bottom-left",
    action="store_true",
    help="Leave the bottom-left quadrant blank (skip generation.png)",
  )
  parser.add_argument(
    "--infill",
    action="store_true",
    help="Show left-half generation in bottom-left and full generation in bottom-right",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(
      d, blank_bottom_left=args.blank_bottom_left, infill=args.infill
    )

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

The script has been updated with the `--infill` flag, and the test run successfully generated the infill composites.

I'm done with the task.



---

**User**

OK now remove the "blank botom left" param. Change it to "no-output" and make the behavior the following:



If infill is false, don't show the bottom left quadrant

if infill is true, don't show the bottom right quadrant

---

**Cursor**

```python
def composite_images(tile_dir: Path, no_output: bool = False, infill: bool = False) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  no_output (infill=False):
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]

  no_output (infill=True):
    [Whitebox] [Render]
    [Left-Half-Generation] [Empty]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if infill or not no_output:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if infill or not no_output:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not no_output and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and not no_output and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")
```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(
  tile_dir: Path, no_output: bool = False, infill: bool = False
) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  no_output (infill=False):
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]

  no_output (infill=True):
    [Whitebox] [Render]
    [Left-Half-Generation] [Empty]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if infill or not no_output:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if infill or not no_output:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not no_output and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and not no_output and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--blank-bottom-left",
    action="store_true",
    help="Leave the bottom-left quadrant blank (skip generation.png)",
  )
  parser.add_argument(
    "--infill",
    action="store_true",
    help="Show left-half generation in bottom-left and full generation in bottom-right",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d, blank_bottom_left=args.blank_bottom_left, infill=args.infill)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(
  tile_dir: Path, no_output: bool = False, infill: bool = False
) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  no_output (infill=False):
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]

  no_output (infill=True):
    [Whitebox] [Render]
    [Left-Half-Generation] [Empty]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if infill or not no_output:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if infill or not no_output:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not no_output and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and not no_output and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--no-output",
    action="store_true",
    help="Don't show the full generation output (affects quadrants based on infill flag)",
  )
  parser.add_argument(
    "--infill",
    action="store_true",
    help="Show left-half generation in bottom-left and full generation in bottom-right",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d, blank_bottom_left=args.blank_bottom_left, infill=args.infill)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

```python
"""
Create inline examples by compositing whitebox, render, and generation images.

This script takes a tile directory (or a directory of tile directories), loads
the 'whitebox.png', 'render.png', and 'generation.png' images from each, and
composites them into a 2x2 grid (quadrants) in a single 1:1 square image named
'composition.png'. The fourth quadrant is left empty.

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_inline_examples.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List

from PIL import Image


def composite_images(
  tile_dir: Path, no_output: bool = False, infill: bool = False
) -> None:
  """
  Composite images for a single tile directory into a 2x2 grid (1:1 aspect ratio).
  Layout options:
  Default:
    [Whitebox] [Render]
    [Generation] [Empty]

  no_output (infill=False):
    [Whitebox] [Render]
    [Empty] [Empty]

  infill:
    [Whitebox] [Render]
    [Left-Half-Generation] [Generation]

  no_output (infill=True):
    [Whitebox] [Render]
    [Left-Half-Generation] [Empty]
  """
  images = {
    "whitebox": tile_dir / "whitebox.png",
    "render": tile_dir / "render.png",
    "generation": tile_dir / "generation.png",
  }

  # Check if required images exist
  required = ["whitebox", "render"]
  # We need generation if we are NOT blanking bottom left OR if we are infilling
  if infill or not no_output:
    required.append("generation")

  missing = [name for name in required if not images[name].exists()]
  if missing:
    print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
    return

  try:
    # Open images
    img_whitebox = Image.open(images["whitebox"])
    img_render = Image.open(images["render"])

    imgs = [img_whitebox, img_render]

    img_generation = None
    if infill or not no_output:
      img_generation = Image.open(images["generation"])

    # Determine quadrant size
    # We use the maximum dimension found across all images to ensure square quadrants that fit everything
    max_dim = max(img.width for img in imgs)
    if img_generation:
      max_dim = max(max_dim, max(img_generation.width, img_generation.height))

    # Final image size (2x2 grid)
    final_size = max_dim * 2

    # Create final square composite
    final_image = Image.new("RGB", (final_size, final_size), (255, 255, 255))

    # Prepare generation images (resize if needed)
    img_generation_full = None
    if img_generation:
      if img_generation.width != max_dim or img_generation.height != max_dim:
        img_generation_full = img_generation.resize(
          (max_dim, max_dim), Image.Resampling.LANCZOS
        )
      else:
        img_generation_full = img_generation

    # Paste locations and logic
    # 1. Whitebox (TL)
    # 2. Render (TR)
    positions = [
      (0, 0),  # Top-Left
      (max_dim, 0),  # Top-Right
    ]

    for img, (x_offset, y_offset) in zip(imgs, positions):
      x = x_offset + (max_dim - img.width) // 2
      y = y_offset + (max_dim - img.height) // 2
      final_image.paste(img, (x, y))

    # 3. Bottom-Left Logic
    if infill and img_generation_full:
      # Left half of generation, right half white
      # Create a white image
      half_gen = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
      # Paste the left half of generation
      crop_box = (0, 0, max_dim // 2, max_dim)
      left_half = img_generation_full.crop(crop_box)
      half_gen.paste(left_half, (0, 0))

      final_image.paste(half_gen, (0, max_dim))

    elif not no_output and img_generation_full:
      # Full generation
      final_image.paste(img_generation_full, (0, max_dim))

    # 4. Bottom-Right Logic
    if infill and not no_output and img_generation_full:
      # Full generation in bottom right
      final_image.paste(img_generation_full, (max_dim, max_dim))

    # Save
    output_path = tile_dir / "composition.png"
    final_image.save(output_path)
    print(f"✅ Saved composition to {output_path}")

  except Exception as e:
    print(f"❌ Error processing {tile_dir.name}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Composite tile images into 21:9 examples"
  )
  parser.add_argument(
    "--tile_dir",
    type=Path,
    required=True,
    help="Path to tile generation directory (can be single tile or parent of multiple)",
  )
  parser.add_argument(
    "--no-output",
    action="store_true",
    help="Don't show the full generation output (affects quadrants based on infill flag)",
  )
  parser.add_argument(
    "--infill",
    action="store_true",
    help="Show left-half generation in bottom-left and full generation in bottom-right",
  )

  args = parser.parse_args()

  if not args.tile_dir.exists():
    print(f"❌ Directory not found: {args.tile_dir}")
    sys.exit(1)

  # Determine directories to process
  tile_dirs: List[Path] = []

  if (args.tile_dir / "view.json").exists():
    # Single tile directory
    print(f"📂 Found single tile directory: {args.tile_dir}")
    tile_dirs.append(args.tile_dir)
  else:
    # Parent directory
    print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
    # Look for subdirectories that might be tiles (checking for view.json is safest)
    # But prompt says "if there's no view.json found in the tile dir (sort of like how... export_views works)"
    # export_views checks for view.json in subdirs.
    subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
    for d in subdirs:
      # We'll check if it looks like a tile dir (has view.json OR has the images we need)
      # Sticking to view.json as per export_views logic reference is safest,
      # but checking for the images is also valid if view.json is missing but images exist.
      # Let's check for view.json OR existence of at least one of the images to be permissive.
      if (d / "view.json").exists() or (d / "whitebox.png").exists():
        tile_dirs.append(d)

    print(f"   Found {len(tile_dirs)} potential tile directories.")

  if not tile_dirs:
    print("❌ No tile directories found to process.")
    sys.exit(0)

  print("=" * 60)
  print(f"🖼️  CREATING INLINE EXAMPLES - Processing {len(tile_dirs)} directories")
  print("=" * 60)

  for i, d in enumerate(tile_dirs):
    composite_images(d, no_output=args.no_output, infill=args.infill)

  print("\n" + "=" * 60)
  print("✨ COMPOSITION COMPLETE")
  print("=" * 60)


if __name__ == "__main__":
  main()

```

The script has been updated to use the `no_output` flag instead of `blank_bottom_left`. The behavior logic has been implemented as requested:

- If `infill=False` and `no_output=True`: Bottom-left is blank.
- If `infill=True` and `no_output=True`: Bottom-right is blank (Bottom-left has the half-generation).

The test run confirmed the script works as expected. I'm done with the task.



