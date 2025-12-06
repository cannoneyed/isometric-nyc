# Execute dataset infill plan
_Exported on 12/5/2025 at 18:04:30 EST from Cursor (2.1.39)_

---

**User**

Execute the plan in @tasks/006_infill_dataset.md 

---

**Cursor**

I'll execute the plan to generate infill dataset variants. Let me start by exploring the structure of the tile directories to understand what files we're working with.

Let me look at an example tile to understand the image dimensions:

I can see the tiles - `render.png` is the aerial photo and `generation.png` is the pixel art version. Let me check the existing synthetic data scripts to understand the patterns and then create the infill script:

Now I have a good understanding of the existing patterns. Let me create the infill dataset generation script that implements all 8 variants as specified in the task file:

```python
"""
Create infill dataset variants for tile-based image generation training.

This script generates 8 variants of infill training data for each tile directory.
Each variant shows different combinations of "generated" and "rendered" regions,
with rendered regions outlined in red.

The 8 variants are:
1. Left half generated, right half rendered (infill_g_r_g_r.png)
2. Right half generated, left half rendered (infill_r_g_r_g.png)
3. Top half generated, bottom half rendered (infill_g_g_r_r.png)
4. Bottom half generated, top half rendered (infill_r_r_g_g.png)
5. Top left quadrant rendered, others generated (infill_r_g_g_g.png)
6. Top right quadrant rendered, others generated (infill_g_r_g_g.png)
7. Bottom left quadrant rendered, others generated (infill_g_g_r_g.png)
8. Bottom right quadrant rendered, others generated (infill_g_g_g_r.png)

Naming convention: infill_<TL>_<TR>_<BL>_<BR>.png
  - g = generated (from generation.png)
  - r = rendered (from render.png)

Usage:
  uv run python src/isometric_nyc/synthetic_data/create_infill_dataset.py --tile_dir PATH
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


# Red outline color and width
OUTLINE_COLOR = (255, 0, 0)
OUTLINE_WIDTH = 1


def draw_outline(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = OUTLINE_COLOR,
    width: int = OUTLINE_WIDTH,
) -> None:
    """
    Draw an outline around a rectangular region.

    Args:
        img: The image to draw on (modified in place).
        box: (x1, y1, x2, y2) coordinates of the region.
        color: RGB color tuple for the outline.
        width: Width of the outline in pixels.
    """
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    # Draw rectangle outline (PIL rectangle is exclusive of x2, y2)
    # We draw the outline inside the box
    for i in range(width):
        draw.rectangle(
            [x1 + i, y1 + i, x2 - 1 - i, y2 - 1 - i],
            outline=color,
            fill=None,
        )


def create_infill_variant(
    img_gen: Image.Image,
    img_render: Image.Image,
    quadrant_sources: Tuple[str, str, str, str],
    target_size: Tuple[int, int] = (1024, 1024),
) -> Image.Image:
    """
    Create an infill variant by compositing generation and render images.

    Args:
        img_gen: The generation (pixel art) image.
        img_render: The render (photo) image.
        quadrant_sources: Tuple of 4 strings ('g' or 'r') indicating source for
                          (top_left, top_right, bottom_left, bottom_right).
        target_size: Output image size.

    Returns:
        Composited image with red outlines around rendered regions.
    """
    width, height = target_size
    half_w = width // 2
    half_h = height // 2

    # Resize images if needed
    if img_gen.size != target_size:
        img_gen = img_gen.resize(target_size, Image.Resampling.LANCZOS)
    if img_render.size != target_size:
        img_render = img_render.resize(target_size, Image.Resampling.LANCZOS)

    # Create output image
    final_image = Image.new("RGB", target_size)

    # Define quadrant boxes: (x1, y1, x2, y2)
    quadrant_boxes = [
        (0, 0, half_w, half_h),  # Top-left
        (half_w, 0, width, half_h),  # Top-right
        (0, half_h, half_w, height),  # Bottom-left
        (half_w, half_h, width, height),  # Bottom-right
    ]

    # Track which quadrants are rendered (for outline drawing)
    rendered_boxes = []

    # Composite each quadrant
    for idx, (source, box) in enumerate(zip(quadrant_sources, quadrant_boxes)):
        x1, y1, x2, y2 = box
        source_img = img_gen if source == "g" else img_render

        # Crop the quadrant from source
        quadrant_crop = source_img.crop(box)

        # Paste into final image
        final_image.paste(quadrant_crop, (x1, y1))

        # Track rendered regions
        if source == "r":
            rendered_boxes.append(box)

    # Draw red outlines around rendered regions
    for box in rendered_boxes:
        draw_outline(final_image, box)

    return final_image


def get_variant_name(quadrant_sources: Tuple[str, str, str, str]) -> str:
    """Generate filename from quadrant sources."""
    return f"infill_{'_'.join(quadrant_sources)}.png"


def create_all_infill_variants(tile_dir: Path) -> int:
    """
    Create all 8 infill variants for a single tile directory.

    Returns:
        Number of variants successfully created.
    """
    images = {
        "generation": tile_dir / "generation.png",
        "render": tile_dir / "render.png",
    }

    # Check if required images exist
    missing = [name for name, path in images.items() if not path.exists()]
    if missing:
        print(f"⚠️  Skipping {tile_dir.name}: Missing {', '.join(missing)}")
        return 0

    try:
        # Open images
        img_gen = Image.open(images["generation"]).convert("RGB")
        img_render = Image.open(images["render"]).convert("RGB")

        # Define all 8 variants
        # Format: (top_left, top_right, bottom_left, bottom_right)
        variants = [
            # Half variants
            ("g", "r", "g", "r"),  # Left generated, right rendered
            ("r", "g", "r", "g"),  # Right generated, left rendered
            ("g", "g", "r", "r"),  # Top generated, bottom rendered
            ("r", "r", "g", "g"),  # Bottom generated, top rendered
            # Single quadrant rendered variants
            ("r", "g", "g", "g"),  # Only TL rendered
            ("g", "r", "g", "g"),  # Only TR rendered
            ("g", "g", "r", "g"),  # Only BL rendered
            ("g", "g", "g", "r"),  # Only BR rendered
        ]

        created = 0
        for quadrant_sources in variants:
            try:
                variant_img = create_infill_variant(img_gen, img_render, quadrant_sources)
                output_path = tile_dir / get_variant_name(quadrant_sources)
                variant_img.save(output_path)
                created += 1
            except Exception as e:
                print(f"  ❌ Error creating {get_variant_name(quadrant_sources)}: {e}")

        print(f"✅ {tile_dir.name}: Created {created}/8 infill variants")
        return created

    except Exception as e:
        print(f"❌ Error processing {tile_dir.name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate infill dataset variants for tile-based image generation training"
    )
    parser.add_argument(
        "--tile_dir",
        type=Path,
        required=True,
        help="Path to tile directory (single tile or parent of multiple tiles)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without creating files",
    )

    args = parser.parse_args()

    if not args.tile_dir.exists():
        print(f"❌ Directory not found: {args.tile_dir}")
        sys.exit(1)

    # Determine directories to process
    tile_dirs: List[Path] = []

    if (args.tile_dir / "view.json").exists() or (
        args.tile_dir / "generation.png"
    ).exists():
        # Single tile directory
        print(f"📂 Found single tile directory: {args.tile_dir}")
        tile_dirs.append(args.tile_dir)
    else:
        # Parent directory - scan for tile subdirectories
        print(f"📂 Scanning parent directory for tiles: {args.tile_dir}")
        subdirs = sorted([d for d in args.tile_dir.iterdir() if d.is_dir()])
        for d in subdirs:
            if (d / "view.json").exists() or (d / "generation.png").exists():
                tile_dirs.append(d)

        print(f"   Found {len(tile_dirs)} potential tile directories.")

    if not tile_dirs:
        print("❌ No tile directories found to process.")
        sys.exit(0)

    print("=" * 60)
    print(f"🖼️  CREATING INFILL DATASET - Processing {len(tile_dirs)} directories")
    print("=" * 60)

    if args.dry_run:
        print("\n🔍 DRY RUN - No files will be created\n")
        for d in tile_dirs:
            gen_exists = (d / "generation.png").exists()
            render_exists = (d / "render.png").exists()
            if gen_exists and render_exists:
                print(f"  Would process: {d.name}")
                for v in [
                    ("g", "r", "g", "r"),
                    ("r", "g", "r", "g"),
                    ("g", "g", "r", "r"),
                    ("r", "r", "g", "g"),
                    ("r", "g", "g", "g"),
                    ("g", "r", "g", "g"),
                    ("g", "g", "r", "g"),
                    ("g", "g", "g", "r"),
                ]:
                    print(f"    - {get_variant_name(v)}")
            else:
                missing = []
                if not gen_exists:
                    missing.append("generation.png")
                if not render_exists:
                    missing.append("render.png")
                print(f"  Would skip: {d.name} (missing: {', '.join(missing)})")
        sys.exit(0)

    total_created = 0
    total_possible = len(tile_dirs) * 8

    for d in tile_dirs:
        created = create_all_infill_variants(d)
        total_created += created

    print("\n" + "=" * 60)
    print(f"✨ INFILL DATASET COMPLETE")
    print(f"   Created {total_created}/{total_possible} variant images")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

Now let me test the script on a single tile first:

The script ran successfully. Let me verify the outputs by viewing one of the generated images:

All 8 variants were created. Let me visually inspect a couple to verify the compositing and red outline work correctly:

The infill variants are working correctly:
1. `infill_g_r_g_r.png` - Left half is pixel art (generated), right half is photo (rendered) with red outlines around both rendered quadrants
2. `infill_g_g_g_r.png` - Only the bottom-right quadrant is rendered (photo) with a red outline, everything else is generated (pixel art)

Now let me run the script on all v04 tiles:

