# Implement water fix plan
_Exported on 12/11/2025 at 09:41:00 EST from Cursor (2.1.39)_

---

**User**

Please implement the plan in @tasks/018_water_fix.md 

---

**Cursor**

I'll implement the `replace_color.py` script as specified in the task. Let me first explore the existing code structure to understand how the generation database works and how other scripts are structured.

Now I have a good understanding of the codebase structure. Let me check the directory structure for exports:

I have all the information I need. Let me create the `replace_color.py` script:

```python
"""
Replace a color in a quadrant's generation using soft blending.

This script performs color replacement while preserving anti-aliased edges
using a soft mask based on color distance (Euclidean norm in RGB space).

Usage:
  # Dry run - exports to exports subdir without saving to db
  uv run python src/isometric_nyc/e2e_generation/replace_color.py \\
    <generation_dir> -x 0 -y 0 \\
    --target-color "#4B697D" --replacement-color "#FF32FF" \\
    --dry-run

  # Apply to database
  uv run python src/isometric_nyc/e2e_generation/replace_color.py \\
    <generation_dir> -x 0 -y 0 \\
    --target-color "#4B697D" --replacement-color "#2A4A5F"

  # Custom softness (default is 60.0)
  uv run python src/isometric_nyc/e2e_generation/replace_color.py \\
    <generation_dir> -x 0 -y 0 \\
    --target-color "#4B697D" --replacement-color "#2A4A5F" \\
    --softness 80.0
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from isometric_nyc.e2e_generation.shared import (
    get_generation_config,
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a hex color string to an RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#4B697D" or "4B697D")

    Returns:
        Tuple of (R, G, B) values (0-255)
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}. Expected 6 characters.")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def soft_color_replace(
    image: np.ndarray,
    target_color: tuple[int, int, int],
    new_color: tuple[int, int, int],
    blend_softness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace a color while preserving anti-aliased edges using a soft mask.

    Uses Euclidean distance in RGB space to create a soft alpha mask,
    then performs alpha blending to smoothly replace the target color.

    Args:
        image: Input image (RGB, uint8).
        target_color: (R, G, B) color to replace (uint8).
        new_color: (R, G, B) replacement color (uint8).
        blend_softness: Controls the width of the blend transition.
                        Higher values = wider, softer edges. Try ranges 20.0 - 100.0.

    Returns:
        Tuple of (result_image_uint8, soft_mask_uint8)
    """
    # --- PREPARATION ---
    # Convert everything to floats between 0.0 and 1.0 for accurate math.
    img_float = image.astype(np.float32) / 255.0
    target_float = np.array(target_color, dtype=np.float32) / 255.0
    new_color_float = np.array(new_color, dtype=np.float32) / 255.0

    # Normalize softness parameter to 0-1 scale roughly matching color distance space
    # (Max Euclidean distance in unit RGB cube is sqrt(3) ~= 1.732)
    softness_scale = (blend_softness / 255.0) * np.sqrt(3)
    # Ensure it's not zero to avoid division errors
    softness_scale = max(softness_scale, 1e-5)

    # --- STEP 1: Calculate Color Distance Map ---
    # Find Euclidean distance between every pixel's color and the target color.
    # axis=2 calculates the norm across the R,G,B channels.
    # Result shape is (H, W). 0.0 means exact match, higher means very different.
    distances = np.linalg.norm(img_float - target_float, axis=2)

    # --- STEP 2: Create the Soft Mask (Alpha Matte) ---
    # We invert the distances: 0 distance should be 1.0 opacity.
    # We divide by softness_scale to control how quickly opacity drops off.
    alpha_mask = 1.0 - (distances / softness_scale)

    # Clip results to ensure the mask stays strictly between 0.0 and 1.0
    alpha_mask = np.clip(alpha_mask, 0.0, 1.0)

    # Reshape mask from (H, W) to (H, W, 1) so it can be multiplied with RGB images
    alpha_expanded = alpha_mask[:, :, np.newaxis]

    # --- STEP 3: Perform Alpha Blending ---
    # Create a solid image filled entirely with the new color
    solid_new_color_img = np.full_like(img_float, new_color_float)

    # Standard Alpha Compositing Formula:
    # Final = (Foreground * Alpha) + (Background * (1 - Alpha))
    # Foreground is the new color, Background is the original image.
    blended_float = (solid_new_color_img * alpha_expanded) + (
        img_float * (1.0 - alpha_expanded)
    )

    # --- FINALIZE ---
    # Convert back to uint8 format
    result_img = (blended_float * 255).astype(np.uint8)
    soft_mask_uint8 = (alpha_mask * 255).astype(np.uint8)

    return result_img, soft_mask_uint8


def process_quadrant(
    conn: sqlite3.Connection,
    config: dict,
    x: int,
    y: int,
    target_color: tuple[int, int, int],
    replacement_color: tuple[int, int, int],
    softness: float,
    dry_run: bool,
    exports_dir: Path,
) -> bool:
    """
    Process a single quadrant, replacing the target color.

    Args:
        conn: Database connection
        config: Generation config
        x: Quadrant x coordinate
        y: Quadrant y coordinate
        target_color: RGB color to replace
        replacement_color: RGB color to use as replacement
        softness: Blend softness value
        dry_run: If True, export to file instead of saving to db
        exports_dir: Directory for dry-run exports

    Returns:
        True if successful, False otherwise
    """
    # Get the generation image
    generation_bytes = get_quadrant_generation(conn, x, y)
    if generation_bytes is None:
        print(f"❌ Error: No generation found for quadrant ({x}, {y})")
        return False

    # Convert to PIL Image, then to numpy array
    pil_image = png_bytes_to_image(generation_bytes)

    # Handle RGBA images - extract RGB channels for processing
    has_alpha = pil_image.mode == "RGBA"
    if has_alpha:
        # Store original alpha channel
        original_alpha = np.array(pil_image)[:, :, 3]
        # Convert to RGB for processing
        pil_image_rgb = pil_image.convert("RGB")
    else:
        pil_image_rgb = pil_image.convert("RGB")
        original_alpha = None

    img_array = np.array(pil_image_rgb)

    print(f"   🎨 Processing quadrant ({x}, {y})...")
    print(f"      Target color: RGB{target_color}")
    print(f"      Replacement color: RGB{replacement_color}")
    print(f"      Softness: {softness}")

    # Apply soft color replacement
    result_array, mask_array = soft_color_replace(
        img_array, target_color, replacement_color, softness
    )

    # Convert back to PIL Image
    result_image = Image.fromarray(result_array, mode="RGB")

    # Restore alpha channel if original had one
    if has_alpha and original_alpha is not None:
        result_rgba = np.zeros(
            (result_array.shape[0], result_array.shape[1], 4), dtype=np.uint8
        )
        result_rgba[:, :, :3] = result_array
        result_rgba[:, :, 3] = original_alpha
        result_image = Image.fromarray(result_rgba, mode="RGBA")

    # Calculate how many pixels were affected (mask > 0)
    affected_pixels = np.sum(mask_array > 0)
    total_pixels = mask_array.size
    affected_percent = (affected_pixels / total_pixels) * 100
    print(f"      Affected pixels: {affected_pixels:,} ({affected_percent:.1f}%)")

    if dry_run:
        # Export to file
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = exports_dir / f"color_replace_{x}_{y}.png"
        mask_path = exports_dir / f"color_replace_{x}_{y}_mask.png"

        result_image.save(output_path, "PNG")
        Image.fromarray(mask_array, mode="L").save(mask_path, "PNG")

        print(f"   ✅ Dry run - exported to:")
        print(f"      Result: {output_path}")
        print(f"      Mask: {mask_path}")
    else:
        # Save back to database
        png_bytes = image_to_png_bytes(result_image)
        save_quadrant_generation(conn, config, x, y, png_bytes)
        print(f"   ✅ Saved to database")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace a color in a quadrant's generation using soft blending."
    )
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation directory containing quadrants.db",
    )
    parser.add_argument(
        "-x",
        type=int,
        required=True,
        help="X coordinate of the quadrant to edit",
    )
    parser.add_argument(
        "-y",
        type=int,
        required=True,
        help="Y coordinate of the quadrant to edit",
    )
    parser.add_argument(
        "--target-color",
        type=str,
        required=True,
        help="Hex code of the color to be replaced (e.g., '#4B697D' or '4B697D')",
    )
    parser.add_argument(
        "--replacement-color",
        type=str,
        required=True,
        help="Hex code of the replacement color (e.g., '#2A4A5F' or '2A4A5F')",
    )
    parser.add_argument(
        "--softness",
        type=float,
        default=60.0,
        help="Blend softness (higher = wider, softer edges). Default: 60.0. Try 20-100.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export result to exports subdir without saving to database",
    )

    args = parser.parse_args()

    # Resolve paths
    generation_dir = args.generation_dir.resolve()

    if not generation_dir.exists():
        print(f"❌ Error: Generation directory not found: {generation_dir}")
        return 1

    db_path = generation_dir / "quadrants.db"
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        return 1

    # Parse colors
    try:
        target_color = hex_to_rgb(args.target_color)
    except ValueError as e:
        print(f"❌ Error parsing target color: {e}")
        return 1

    try:
        replacement_color = hex_to_rgb(args.replacement_color)
    except ValueError as e:
        print(f"❌ Error parsing replacement color: {e}")
        return 1

    exports_dir = generation_dir / "exports"

    print(f"\n🎨 Color Replacement Tool")
    print(f"   Generation dir: {generation_dir}")
    print(f"   Quadrant: ({args.x}, {args.y})")
    if args.dry_run:
        print(f"   Mode: DRY RUN (will export to {exports_dir})")
    else:
        print(f"   Mode: LIVE (will save to database)")

    conn = sqlite3.connect(db_path)

    try:
        config = get_generation_config(conn)

        success = process_quadrant(
            conn=conn,
            config=config,
            x=args.x,
            y=args.y,
            target_color=target_color,
            replacement_color=replacement_color,
            softness=args.softness,
            dry_run=args.dry_run,
            exports_dir=exports_dir,
        )

        return 0 if success else 1

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

```

Let me verify the script works by running it with `--help`:

The script is working. Now let me test it with a dry run on an actual quadrant to verify the color replacement works:

The script is working. Let me verify the exported files exist:

The `replace_color.py` script has been implemented successfully. Here's a summary:

## Implementation Complete

The script is located at `src/isometric_nyc/e2e_generation/replace_color.py` and implements the soft blend color replacement algorithm as specified.

### Features

1. **Soft Blend Algorithm**: Uses Euclidean distance in RGB space to create a soft alpha mask that preserves anti-aliased edges when replacing colors

2. **Parameters**:
   - `generation_dir` - Directory containing the generation database
   - `-x` / `-y` - Quadrant coordinates
   - `--target-color` - Hex code of color to replace (e.g., `#4B697D`)
   - `--replacement-color` - Hex code of replacement color
   - `--softness` - Controls edge blend width (default: 60.0, range: 20-100)
   - `--dry-run` - Exports to `exports/` subdir without saving to database

3. **RGBA Support**: Preserves alpha channels if present in the original image

4. **Outputs**:
   - In dry-run mode: Exports both the result image and a grayscale mask showing the blend areas
   - In live mode: Saves directly to the database

### Example Usage

```bash
# Dry run - preview the result
uv run python src/isometric_nyc/e2e_generation/replace_color.py \
  generations/v01 -x 0 -y 0 \
  --target-color "#4B697D" --replacement-color "#2A4A5F" \
  --dry-run

# Apply to database
uv run python src/isometric_nyc/e2e_generation/replace_color.py \
  generations/v01 -x 0 -y 0 \
  --target-color "#4B697D" --replacement-color "#2A4A5F"
```

```shellscript
# Dry run - preview the result
uv run python src/isometric_nyc/e2e_generation/replace_color.py \
  generations/v01 -x 0 -y 0 \
  --target-color "#4B697D" --replacement-color "#2A4A5F" \
  --dry-run

# Apply to database
uv run python src/isometric_nyc/e2e_generation/replace_color.py \
  generations/v01 -x 0 -y 0 \
  --target-color "#4B697D" --replacement-color "#2A4A5F"
```

---

**User**

ok great - now we want to wire this color fix up to the frontend @src/isometric_nyc/e2e_generation/app.py - add a new button next to "select" called "fix water" - when the tool is selected, the user should be able to select a single quadrant. We need to select the color of the pixel value of wherever the user clicked within the quadrant, and store that as the "target color" - display this with a small swatch in the status bar below the controls

---

**Cursor**

I'll add the "Fix Water" feature to the frontend. Let me first examine the current UI structure:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Generated Tiles Viewer</title>
    <link rel="stylesheet" href="/static/viewer.css" />
    <style>
      /* Dynamic grid sizing based on template variables */
      .grid {
        grid-template-columns: repeat({{ nx }}, {{ size_px }}px);
        grid-auto-rows: {{ size_px }}px;
      }
      .tile.placeholder {
        min-width: {{ size_px }}px;
        min-height: {{ size_px }}px;
      }
    </style>
  </head>
  <body>
    <!-- Config data for JS -->
    <script
      id="app-config"
      type="application/json"
      data-config='{"x": {{ x }}, "y": {{ y }}, "nx": {{ nx }}, "ny": {{ ny }}, "size_px": {{ size_px }}}'
    ></script>

    <!-- Toast notification container -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- Fixed toolbar -->
    <div class="toolbar">
      <div class="toolbar-row">
        <div class="controls">
          <label>X: <input type="number" id="x" value="{{ x }}" /></label>
          <label>Y: <input type="number" id="y" value="{{ y }}" /></label>
          <label
            >NX: <input type="number" id="nx" value="{{ nx }}" min="1" max="20"
          /></label>
          <label
            >NY: <input type="number" id="ny" value="{{ ny }}" min="1" max="20"
          /></label>
          <label
            >Size:
            <input type="number" id="sizePx" value="{{ size_px }}" step="32"
          /></label>
          <button onclick="goTo()">Go</button>
          <div class="toggle-group">
            <label
              ><input
                type="checkbox"
                id="showLines"
                {%
                if
                show_lines
                %}checked{%
                endif
                %}
                onchange="toggleLines()"
              />
              Lines</label
            >
            <label
              ><input
                type="checkbox"
                id="showCoords"
                {%
                if
                show_coords
                %}checked{%
                endif
                %}
                onchange="toggleCoords()"
              />
              Coords</label
            >
            <label
              ><input
                type="checkbox"
                id="showRender"
                {%
                if
                show_render
                %}checked{%
                endif
                %}
                onchange="toggleRender()"
              />
              Renders</label
            >
          </div>
          <div class="toggle-group tools-group">
            <span class="tools-label">Tools:</span>
            <button
              id="selectTool"
              class="tool-btn"
              onclick="toggleSelectTool()"
              title="Select quadrants (S)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
                <path d="M13 13l6 6"></path>
              </svg>
              Select
            </button>
            <button
              id="fixWaterTool"
              class="tool-btn"
              onclick="toggleFixWaterTool()"
              title="Fix water color (W)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
              </svg>
              Fix Water
            </button>
          </div>
        </div>
        <div class="toolbar-info">
          <span>({{ x }}, {{ y }}) → ({{ x + nx - 1 }}, {{ y + ny - 1 }})</span>
          <span>{{ generation_dir }}</span>
        </div>
      </div>
      <div class="selection-status" id="selectionStatus">
        <span id="selectionCount">0 quadrants selected</span>
        <span class="selection-limit">(max 4)</span>
        <button
          id="deselectAllBtn"
          class="deselect-btn"
          onclick="deselectAll()"
          disabled
        >
          Deselect All
        </button>
        <button
          id="deleteBtn"
          class="delete-btn"
          onclick="deleteSelected()"
          disabled
        >
          Delete
        </button>
        <button
          id="renderBtn"
          class="render-btn"
          onclick="renderSelected()"
          disabled
        >
          Render
        </button>
        <button
          id="generateBtn"
          class="generate-btn"
          onclick="generateSelected()"
          disabled
        >
          Generate
        </button>
      </div>
    </div>

    <div
      class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}"
      id="gridContainer"
    >
      <div class="grid">
        {% for dy in range(ny) %} {% for dx in range(nx) %} {% set qx = x + dx
        %} {% set qy = y + dy %} {% set has_gen = tiles.get((dx, dy), False) %}
        <div
          class="tile {% if not has_gen %}placeholder{% endif %}"
          data-coords="{{ qx }},{{ qy }}"
        >
          <span class="coords">({{ qx }}, {{ qy }})</span>
          {% if has_gen %}
          <img
            src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}"
            alt="Tile {{ qx }},{{ qy }}"
          />
          {% endif %}
        </div>
        {% endfor %} {% endfor %}
      </div>
    </div>

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

Now let me add a section for the water fix status bar in the HTML:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Generated Tiles Viewer</title>
    <link rel="stylesheet" href="/static/viewer.css" />
    <style>
      /* Dynamic grid sizing based on template variables */
      .grid {
        grid-template-columns: repeat({{ nx }}, {{ size_px }}px);
        grid-auto-rows: {{ size_px }}px;
      }
      .tile.placeholder {
        min-width: {{ size_px }}px;
        min-height: {{ size_px }}px;
      }
    </style>
  </head>
  <body>
    <!-- Config data for JS -->
    <script
      id="app-config"
      type="application/json"
      data-config='{"x": {{ x }}, "y": {{ y }}, "nx": {{ nx }}, "ny": {{ ny }}, "size_px": {{ size_px }}}'
    ></script>

    <!-- Toast notification container -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- Fixed toolbar -->
    <div class="toolbar">
      <div class="toolbar-row">
        <div class="controls">
          <label>X: <input type="number" id="x" value="{{ x }}" /></label>
          <label>Y: <input type="number" id="y" value="{{ y }}" /></label>
          <label
            >NX: <input type="number" id="nx" value="{{ nx }}" min="1" max="20"
          /></label>
          <label
            >NY: <input type="number" id="ny" value="{{ ny }}" min="1" max="20"
          /></label>
          <label
            >Size:
            <input type="number" id="sizePx" value="{{ size_px }}" step="32"
          /></label>
          <button onclick="goTo()">Go</button>
          <div class="toggle-group">
            <label
              ><input
                type="checkbox"
                id="showLines"
                {%
                if
                show_lines
                %}checked{%
                endif
                %}
                onchange="toggleLines()"
              />
              Lines</label
            >
            <label
              ><input
                type="checkbox"
                id="showCoords"
                {%
                if
                show_coords
                %}checked{%
                endif
                %}
                onchange="toggleCoords()"
              />
              Coords</label
            >
            <label
              ><input
                type="checkbox"
                id="showRender"
                {%
                if
                show_render
                %}checked{%
                endif
                %}
                onchange="toggleRender()"
              />
              Renders</label
            >
          </div>
          <div class="toggle-group tools-group">
            <span class="tools-label">Tools:</span>
            <button
              id="selectTool"
              class="tool-btn"
              onclick="toggleSelectTool()"
              title="Select quadrants (S)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
                <path d="M13 13l6 6"></path>
              </svg>
              Select
            </button>
            <button
              id="fixWaterTool"
              class="tool-btn"
              onclick="toggleFixWaterTool()"
              title="Fix water color (W)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
              </svg>
              Fix Water
            </button>
          </div>
        </div>
        <div class="toolbar-info">
          <span>({{ x }}, {{ y }}) → ({{ x + nx - 1 }}, {{ y + ny - 1 }})</span>
          <span>{{ generation_dir }}</span>
        </div>
      </div>
      <div class="selection-status" id="selectionStatus">
        <span id="selectionCount">0 quadrants selected</span>
        <span class="selection-limit">(max 4)</span>
        <button
          id="deselectAllBtn"
          class="deselect-btn"
          onclick="deselectAll()"
          disabled
        >
          Deselect All
        </button>
        <button
          id="deleteBtn"
          class="delete-btn"
          onclick="deleteSelected()"
          disabled
        >
          Delete
        </button>
        <button
          id="renderBtn"
          class="render-btn"
          onclick="renderSelected()"
          disabled
        >
          Render
        </button>
        <button
          id="generateBtn"
          class="generate-btn"
          onclick="generateSelected()"
          disabled
        >
          Generate
        </button>
      </div>
      <div class="water-fix-status" id="waterFixStatus" style="display: none;">
        <span class="water-fix-label">Target Color:</span>
        <span class="color-swatch" id="targetColorSwatch"></span>
        <span class="color-hex" id="targetColorHex">Click a quadrant to pick color</span>
        <span class="water-fix-quadrant" id="waterFixQuadrant"></span>
        <button
          id="applyWaterFixBtn"
          class="apply-water-fix-btn"
          onclick="applyWaterFix()"
          disabled
        >
          Apply Fix
        </button>
        <button
          id="cancelWaterFixBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFix()"
        >
          Cancel
        </button>
      </div>
    </div>

    <div
      class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}"
      id="gridContainer"
    >
      <div class="grid">
        {% for dy in range(ny) %} {% for dx in range(nx) %} {% set qx = x + dx
        %} {% set qy = y + dy %} {% set has_gen = tiles.get((dx, dy), False) %}
        <div
          class="tile {% if not has_gen %}placeholder{% endif %}"
          data-coords="{{ qx }},{{ qy }}"
        >
          <span class="coords">({{ qx }}, {{ qy }})</span>
          {% if has_gen %}
          <img
            src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}"
            alt="Tile {{ qx }},{{ qy }}"
          />
          {% endif %}
        </div>
        {% endfor %} {% endfor %}
      </div>
    </div>

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
  font-size: 12px;
  background: #1a1a2e;
  color: #eee;
  min-height: 100vh;
  padding: 15px;
  padding-top: 95px; /* Space for fixed toolbar */
}

.toolbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: #1a1a2e;
  padding: 10px 15px;
  z-index: 100;
  border-bottom: 1px solid #333;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.toolbar-row {
  display: flex;
  align-items: center;
  gap: 15px;
  flex-wrap: wrap;
}

.toolbar-info {
  color: #666;
  font-size: 0.75rem;
}

.toolbar-info span + span {
  margin-left: 12px;
}

.controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.controls label {
  color: #888;
  font-size: 0.8rem;
}

.controls input[type="number"] {
  width: 50px;
  padding: 5px 6px;
  border: 1px solid #333;
  border-radius: 3px;
  background: #16213e;
  color: #fff;
  font-family: inherit;
  font-size: 0.8rem;
}

.controls button {
  padding: 5px 12px;
  background: #00d9ff;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.75rem;
  font-weight: bold;
}

.controls button:hover {
  background: #00b8d4;
}

.toggle-group {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: 8px;
  padding-left: 12px;
  border-left: 1px solid #333;
}

.toggle-group label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  user-select: none;
  font-size: 0.75rem;
}

.toggle-group input[type="checkbox"] {
  width: 14px;
  height: 14px;
  accent-color: #00d9ff;
  cursor: pointer;
}

.grid-container {
  display: inline-block;
  border-radius: 8px;
  overflow: hidden;
}

.grid-container.show-lines {
  border: 2px solid #333;
}

.grid {
  display: grid;
  background: #333;
}

.grid-container.show-lines .grid {
  gap: 2px;
}

.grid-container:not(.show-lines) .grid {
  gap: 0;
  background: transparent;
}

.grid-container:not(.show-lines) {
  border: none;
}

.tile {
  position: relative;
  background: #2a2a4a;
  display: flex;
  align-items: center;
  justify-content: center;
}

.tile img {
  display: block;
  max-width: 100%;
  height: auto;
}

.tile.placeholder {
  background: #3a3a5a;
}

.tile .coords {
  position: absolute;
  top: 6px;
  left: 6px;
  background: rgba(0, 0, 0, 0.7);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.65rem;
  color: #00d9ff;
  transition: opacity 0.2s;
}

.tile.placeholder .coords {
  color: #666;
}

.grid-container:not(.show-coords) .tile .coords {
  opacity: 0;
}

/* Tool button styles */
.tools-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.tools-label {
  color: #666;
  font-size: 0.7rem;
}

.tool-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: #2a2a3e;
  color: #666;
  border: 1px solid #444;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  transition: all 0.15s ease;
}

.tool-btn:hover {
  background: #3a3a4e;
  color: #999;
  border-color: #555;
}

.tool-btn.active {
  background: #10b981;
  color: #fff;
  border-color: #10b981;
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.tool-btn.active:hover {
  background: #059669;
  border-color: #059669;
}

.tool-btn svg {
  width: 12px;
  height: 12px;
}

/* Selection styles */
.tile.selected {
  outline: 3px solid #ff3333;
  outline-offset: -3px;
  z-index: 10;
}

.grid-container.show-lines .tile.selected {
  outline-color: #ff3333;
}

.tile.selectable {
  cursor: pointer;
}

.tile.placeholder.selected {
  background: rgba(255, 51, 51, 0.15);
}

/* Locked tiles (during generation) */
.tile.locked {
  outline: 3px solid #a855f7;
  outline-offset: -3px;
  z-index: 10;
}

.grid-container.show-lines .tile.locked {
  outline-color: #a855f7;
}

.tile.placeholder.locked {
  background: rgba(168, 85, 247, 0.2);
}

/* Queued tiles (in queue, waiting to be generated) */
.tile.queued {
  outline: 3px dashed #c084fc !important;
  outline-offset: -3px;
  z-index: 9;
  animation: pulse-queue 1.5s ease-in-out infinite;
  /* Fallback: box-shadow for browsers that don't render dashed outlines well */
  box-shadow: inset 0 0 0 3px rgba(192, 132, 252, 0.5);
}

@keyframes pulse-queue {
  0%,
  100% {
    outline-color: #c084fc;
    box-shadow: inset 0 0 0 3px rgba(192, 132, 252, 0.5);
  }
  50% {
    outline-color: #8b5cf6;
    box-shadow: inset 0 0 0 3px rgba(139, 92, 246, 0.7);
  }
}

.grid-container.show-lines .tile.queued {
  outline-color: #c084fc;
}

.tile.placeholder.queued {
  background: rgba(168, 85, 247, 0.3) !important;
}

/* Allow selection while generating - only disable clicking on locked/queued tiles */
body.generating .tile.locked.selectable,
body.generating .tile.queued.selectable {
  cursor: not-allowed;
}

/* Selection status bar */
.selection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  background: rgba(255, 51, 51, 0.1);
  border: 1px solid rgba(255, 51, 51, 0.3);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #ff6666;
  margin-top: 8px;
}

.selection-status.empty {
  background: transparent;
  border-color: #333;
  color: #666;
}

.selection-status.generating {
  background: rgba(168, 85, 247, 0.1);
  border-color: rgba(168, 85, 247, 0.4);
  color: #a855f7;
}

.selection-limit {
  color: #888;
  font-size: 0.65rem;
}

.deselect-btn {
  padding: 3px 8px;
  background: #ff3333;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  margin-left: auto;
  transition: all 0.2s;
}

.deselect-btn:hover:not(:disabled) {
  background: #ff5555;
}

.deselect-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
}

.delete-btn {
  padding: 3px 8px;
  background: #dc2626;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  transition: all 0.2s;
}

.delete-btn:hover:not(:disabled) {
  background: #b91c1c;
}

.delete-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
}

.render-btn {
  padding: 4px 12px;
  background: #f59e0b;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
}

.render-btn:hover:not(:disabled) {
  background: #d97706;
}

.render-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

.render-btn.loading {
  background: #666;
  cursor: wait;
}

.render-btn .spinner {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-left: 4px;
  vertical-align: middle;
}

.generate-btn {
  padding: 4px 12px;
  background: #00d9ff;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
}

.generate-btn:hover:not(:disabled) {
  background: #00b8d4;
}

.generate-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

/* Toast notification system */
.toast-container {
  position: fixed;
  bottom: 15px;
  right: 15px;
  z-index: 1000;
  display: flex;
  flex-direction: column-reverse;
  gap: 8px;
  max-width: 320px;
}

.toast {
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 0.75rem;
  animation: slideIn 0.3s ease-out;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.toast.success {
  background: linear-gradient(135deg, #065f46 0%, #047857 100%);
  border: 1px solid #10b981;
  color: #d1fae5;
}

.toast.error {
  background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
  border: 1px solid #ef4444;
  color: #fee2e2;
}

.toast.info {
  background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%);
  border: 1px solid #3b82f6;
  color: #dbeafe;
}

.toast.loading {
  background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
  border: 1px solid #6b7280;
  color: #e5e7eb;
}

.toast-icon {
  flex-shrink: 0;
  font-size: 1rem;
}

.toast-content {
  flex: 1;
}

.toast-title {
  font-weight: 600;
  margin-bottom: 2px;
}

.toast-message {
  opacity: 0.9;
  font-size: 0.7rem;
}

.toast-close {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  opacity: 0.7;
  font-size: 1rem;
  padding: 0;
  line-height: 1;
}

.toast-close:hover {
  opacity: 1;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@keyframes slideOut {
  from {
    transform: translateX(0);
    opacity: 1;
  }
  to {
    transform: translateX(100%);
    opacity: 0;
  }
}

.toast.removing {
  animation: slideOut 0.3s ease-in forwards;
}

/* Loading spinner for generate button */
.generate-btn.loading {
  background: #666;
  color: #fff;
  cursor: wait;
}

.generate-btn .spinner {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-left: 4px;
  vertical-align: middle;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* Water fix tool styles */
.water-fix-status {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #60a5fa;
  margin-top: 8px;
}

.water-fix-label {
  color: #93c5fd;
  font-weight: 500;
}

.color-swatch {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.3), inset 0 0 0 1px rgba(0, 0, 0, 0.1);
  background: #333;
}

.color-swatch.has-color {
  cursor: pointer;
}

.color-hex {
  font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
  font-size: 0.75rem;
  color: #93c5fd;
  min-width: 180px;
}

.water-fix-quadrant {
  color: #60a5fa;
  font-size: 0.7rem;
}

.apply-water-fix-btn {
  padding: 4px 12px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
  margin-left: auto;
}

.apply-water-fix-btn:hover:not(:disabled) {
  background: #2563eb;
}

.apply-water-fix-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

.apply-water-fix-btn.loading {
  background: #666;
  cursor: wait;
}

.cancel-water-fix-btn {
  padding: 4px 8px;
  background: #4b5563;
  color: #d1d5db;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  transition: all 0.2s;
}

.cancel-water-fix-btn:hover {
  background: #6b7280;
}

/* Fix water tool active state */
.tile.fix-water-selectable {
  cursor: crosshair;
}

.tile.fix-water-selectable:hover {
  outline: 2px solid #3b82f6;
  outline-offset: -2px;
}

.tile.water-fix-selected {
  outline: 3px solid #3b82f6;
  outline-offset: -3px;
  z-index: 10;
}

```

Now let me add the fix water tool logic to the JavaScript. I'll add it after the keyboard navigation section:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  if (locked.length === 0 && queued.length === 0) {
    console.log("applyLockedStyles - nothing to apply, early return");
    return;
  }

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  }

  // Apply locked/queued style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now let me add the fix water tool state and functions after the select tool section:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  if (locked.length === 0 && queued.length === 0) {
    console.log("applyLockedStyles - nothing to apply, early return");
    return;
  }

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  }

  // Apply locked/queued style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  document.getElementById("applyWaterFixBtn").disabled = true;
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById("targetColorHex").textContent =
      `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById("waterFixQuadrant").textContent =
      `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast(
      "info",
      "Color picked",
      `Target color: ${hex} at (${qx}, ${qy})`
    );
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Reset and reload after a short delay
      setTimeout(() => {
        cancelWaterFix();
        window.location.reload();
      }, 1500);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now I need to update the tile click handler to handle fix water clicks:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  if (locked.length === 0 && queued.length === 0) {
    console.log("applyLockedStyles - nothing to apply, early return");
    return;
  }

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  }

  // Apply locked/queued style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  document.getElementById("applyWaterFixBtn").disabled = true;
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById("targetColorHex").textContent =
      `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById("waterFixQuadrant").textContent =
      `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast(
      "info",
      "Color picked",
      `Target color: ${hex} at (${qx}, ${qy})`
    );
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Reset and reload after a short delay
      setTimeout(() => {
        cancelWaterFix();
        window.location.reload();
      }, 1500);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now I need to add the API endpoint in `app.py` to handle the water fix request:

```python
"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import hashlib
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.replace_color import (
  hex_to_rgb,
  process_quadrant,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
}

# Generation queue for multiple requests
generation_queue: list[
  dict
] = []  # List of {"quadrants": [...], "type": "generate"|"render"}
queue_lock = threading.Lock()
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Ensure web server is running before generation
  ensure_web_server_running()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
  )


def process_queue_item(item: dict) -> dict:
  """Process a single queue item (generate or render)."""
  global generation_state

  quadrants = item["quadrants"]
  item_type = item["type"]

  # Convert to list of tuples
  selected_quadrants = [(q[0], q[1]) for q in quadrants]

  # Initialize generation state
  generation_state["is_generating"] = True
  generation_state["quadrants"] = selected_quadrants
  generation_state["status"] = "starting" if item_type == "generate" else "rendering"
  generation_state["message"] = f"Starting {item_type}..."
  generation_state["error"] = None
  generation_state["started_at"] = time.time()

  print(f"\n{'=' * 60}")
  print(
    f"{'🎯' if item_type == 'generate' else '🎨'} {item_type.title()} request from queue: {selected_quadrants}"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    if item_type == "generate":
      result = run_generation(conn, config, selected_quadrants)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
      return result
    else:  # render
      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def queue_worker():
  """Background worker that processes the generation queue."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    item = None

    # Get next item from queue
    with queue_lock:
      if generation_queue:
        item = generation_queue.pop(0)

    if item is None:
      # No items in queue, wait a bit and check again
      time.sleep(0.5)
      continue

    # Acquire the generation lock and process the item
    with generation_lock:
      try:
        process_queue_item(item)
      finally:
        generation_state["is_generating"] = False

    # Small delay between items
    time.sleep(0.5)

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue(quadrants: list[tuple[int, int]], item_type: str) -> dict:
  """Add a generation/render request to the queue."""
  with queue_lock:
    generation_queue.append(
      {
        "quadrants": quadrants,
        "type": item_type,
      }
    )
    queue_position = len(generation_queue)

  # Ensure the queue worker is running
  start_queue_worker()

  return {
    "success": True,
    "queued": True,
    "position": queue_position,
    "message": f"Added to queue at position {queue_position}",
  }


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  with queue_lock:
    queue_info = [
      {"quadrants": item["quadrants"], "type": item["type"]}
      for item in generation_queue
    ]
  return jsonify(
    {
      **generation_state,
      "queue": queue_info,
      "queue_length": len(queue_info),
    }
  )


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating/rendering - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "render")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state (reuse for rendering)
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "rendering"
    generation_state["message"] = "Starting render..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎨 Render request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)

      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return jsonify(
        {
          "success": True,
          "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
          "quadrants": selected_quadrants,
        }
      ), 200

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """API endpoint to generate tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "generate")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting"
    generation_state["message"] = "Starting generation..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎯 Generation request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)
      result = run_generation(conn, config, selected_quadrants)

      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        return jsonify(result), 200
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        return jsonify(result), 400

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


def ensure_web_server_running() -> None:
  """Ensure the web server for rendering is running."""
  global WEB_SERVER_PROCESS

  if WEB_SERVER_PROCESS is not None:
    # Check if still running
    if WEB_SERVER_PROCESS.poll() is None:
      return  # Still running

  # Start the web server
  print(f"🌐 Starting web server for rendering on port {WEB_SERVER_PORT}...")
  WEB_SERVER_PROCESS = start_web_server(WEB_DIR, WEB_SERVER_PORT)


def main():
  global GENERATION_DIR, WEB_SERVER_PORT

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())

```

---

**User**

OK awesome - now let's wire up the functions in @src/isometric_nyc/e2e_generation/replace_color.py to @src/isometric_nyc/e2e_generation/app.py so that we can replace the color - the water replacement_color should be hard-coded as 4A6372

---

**Cursor**

I'll add the import and create the `/api/fix-water` API endpoint in `app.py`:

```python
"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import hashlib
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.replace_color import (
  hex_to_rgb,
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
}

# Generation queue for multiple requests
generation_queue: list[
  dict
] = []  # List of {"quadrants": [...], "type": "generate"|"render"}
queue_lock = threading.Lock()
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Ensure web server is running before generation
  ensure_web_server_running()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
  )


def process_queue_item(item: dict) -> dict:
  """Process a single queue item (generate or render)."""
  global generation_state

  quadrants = item["quadrants"]
  item_type = item["type"]

  # Convert to list of tuples
  selected_quadrants = [(q[0], q[1]) for q in quadrants]

  # Initialize generation state
  generation_state["is_generating"] = True
  generation_state["quadrants"] = selected_quadrants
  generation_state["status"] = "starting" if item_type == "generate" else "rendering"
  generation_state["message"] = f"Starting {item_type}..."
  generation_state["error"] = None
  generation_state["started_at"] = time.time()

  print(f"\n{'=' * 60}")
  print(
    f"{'🎯' if item_type == 'generate' else '🎨'} {item_type.title()} request from queue: {selected_quadrants}"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    if item_type == "generate":
      result = run_generation(conn, config, selected_quadrants)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
      return result
    else:  # render
      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def queue_worker():
  """Background worker that processes the generation queue."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    item = None

    # Get next item from queue
    with queue_lock:
      if generation_queue:
        item = generation_queue.pop(0)

    if item is None:
      # No items in queue, wait a bit and check again
      time.sleep(0.5)
      continue

    # Acquire the generation lock and process the item
    with generation_lock:
      try:
        process_queue_item(item)
      finally:
        generation_state["is_generating"] = False

    # Small delay between items
    time.sleep(0.5)

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue(quadrants: list[tuple[int, int]], item_type: str) -> dict:
  """Add a generation/render request to the queue."""
  with queue_lock:
    generation_queue.append(
      {
        "quadrants": quadrants,
        "type": item_type,
      }
    )
    queue_position = len(generation_queue)

  # Ensure the queue worker is running
  start_queue_worker()

  return {
    "success": True,
    "queued": True,
    "position": queue_position,
    "message": f"Added to queue at position {queue_position}",
  }


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  with queue_lock:
    queue_info = [
      {"quadrants": item["quadrants"], "type": item["type"]}
      for item in generation_queue
    ]
  return jsonify(
    {
      **generation_state,
      "queue": queue_info,
      "queue_length": len(queue_info),
    }
  )


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating/rendering - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "render")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state (reuse for rendering)
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "rendering"
    generation_state["message"] = "Starting render..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎨 Render request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)

      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return jsonify(
        {
          "success": True,
          "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
          "quadrants": selected_quadrants,
        }
      ), 200

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """API endpoint to generate tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "generate")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting"
    generation_state["message"] = "Starting generation..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎯 Generation request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)
      result = run_generation(conn, config, selected_quadrants)

      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        return jsonify(result), 200
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        return jsonify(result), 400

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


def ensure_web_server_running() -> None:
  """Ensure the web server for rendering is running."""
  global WEB_SERVER_PROCESS

  if WEB_SERVER_PROCESS is not None:
    # Check if still running
    if WEB_SERVER_PROCESS.poll() is None:
      return  # Still running

  # Start the web server
  print(f"🌐 Starting web server for rendering on port {WEB_SERVER_PORT}...")
  WEB_SERVER_PROCESS = start_web_server(WEB_DIR, WEB_SERVER_PORT)


def main():
  global GENERATION_DIR, WEB_SERVER_PORT

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())

```

Now let me find the delete endpoint and add the fix-water endpoint after it:

```python
"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import hashlib
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.replace_color import (
  hex_to_rgb,
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
}

# Generation queue for multiple requests
generation_queue: list[
  dict
] = []  # List of {"quadrants": [...], "type": "generate"|"render"}
queue_lock = threading.Lock()
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Ensure web server is running before generation
  ensure_web_server_running()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
  )


def process_queue_item(item: dict) -> dict:
  """Process a single queue item (generate or render)."""
  global generation_state

  quadrants = item["quadrants"]
  item_type = item["type"]

  # Convert to list of tuples
  selected_quadrants = [(q[0], q[1]) for q in quadrants]

  # Initialize generation state
  generation_state["is_generating"] = True
  generation_state["quadrants"] = selected_quadrants
  generation_state["status"] = "starting" if item_type == "generate" else "rendering"
  generation_state["message"] = f"Starting {item_type}..."
  generation_state["error"] = None
  generation_state["started_at"] = time.time()

  print(f"\n{'=' * 60}")
  print(
    f"{'🎯' if item_type == 'generate' else '🎨'} {item_type.title()} request from queue: {selected_quadrants}"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    if item_type == "generate":
      result = run_generation(conn, config, selected_quadrants)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
      return result
    else:  # render
      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def queue_worker():
  """Background worker that processes the generation queue."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    item = None

    # Get next item from queue
    with queue_lock:
      if generation_queue:
        item = generation_queue.pop(0)

    if item is None:
      # No items in queue, wait a bit and check again
      time.sleep(0.5)
      continue

    # Acquire the generation lock and process the item
    with generation_lock:
      try:
        process_queue_item(item)
      finally:
        generation_state["is_generating"] = False

    # Small delay between items
    time.sleep(0.5)

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue(quadrants: list[tuple[int, int]], item_type: str) -> dict:
  """Add a generation/render request to the queue."""
  with queue_lock:
    generation_queue.append(
      {
        "quadrants": quadrants,
        "type": item_type,
      }
    )
    queue_position = len(generation_queue)

  # Ensure the queue worker is running
  start_queue_worker()

  return {
    "success": True,
    "queued": True,
    "position": queue_position,
    "message": f"Added to queue at position {queue_position}",
  }


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  with queue_lock:
    queue_info = [
      {"quadrants": item["quadrants"], "type": item["type"]}
      for item in generation_queue
    ]
  return jsonify(
    {
      **generation_state,
      "queue": queue_info,
      "queue_length": len(queue_info),
    }
  )


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 60.0


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify(
      {"success": False, "error": f"Invalid replacement color: {e}"}
    ), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating/rendering - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "render")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state (reuse for rendering)
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "rendering"
    generation_state["message"] = "Starting render..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎨 Render request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)

      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return jsonify(
        {
          "success": True,
          "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
          "quadrants": selected_quadrants,
        }
      ), 200

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """API endpoint to generate tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "generate")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting"
    generation_state["message"] = "Starting generation..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎯 Generation request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)
      result = run_generation(conn, config, selected_quadrants)

      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        return jsonify(result), 200
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        return jsonify(result), 400

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


def ensure_web_server_running() -> None:
  """Ensure the web server for rendering is running."""
  global WEB_SERVER_PROCESS

  if WEB_SERVER_PROCESS is not None:
    # Check if still running
    if WEB_SERVER_PROCESS.poll() is None:
      return  # Still running

  # Start the web server
  print(f"🌐 Starting web server for rendering on port {WEB_SERVER_PORT}...")
  WEB_SERVER_PROCESS = start_web_server(WEB_DIR, WEB_SERVER_PORT)


def main():
  global GENERATION_DIR, WEB_SERVER_PORT

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())

```

The lint is just a formatting warning. Let me update the todos and test:

Let me check if the app is already running:

The app is already running. The implementation is complete. Let me provide a summary:

The "Fix Water" feature is now fully wired up. Since the app was already running, you'll need to restart it to pick up the changes:

```bash
# In terminal 3, stop the app with Ctrl+C, then:
uv run python src/isometric_nyc/e2e_generation/app.py generations/v01
```

## Summary of Changes

**1. `viewer.html`** - Added "Fix Water" button with droplet icon next to "Select"

**2. `viewer.css`** - Added styles for:
   - Water fix status bar with color swatch
   - Fix water tool button states
   - Crosshair cursor when hovering tiles

**3. `viewer.js`** - Added fix water tool logic:
   - Toggle between Select and Fix Water tools (keyboard shortcut: `W`)
   - Pixel color picker using off-screen canvas
   - Color swatch display in status bar
   - API call to apply the fix

**4. `app.py`** - Added `/api/fix-water` endpoint:
   - Accepts quadrant coordinates and target color
   - Hardcoded replacement color: `#4A6372`
   - Default softness: 60.0
   - Calls `process_color_replacement` to apply the soft blend

## How to Use

1. Click **Fix Water** tool (or press `W`)
2. Click anywhere on a quadrant tile - the pixel color under your click becomes the target color
3. The color swatch and hex code appear in the blue status bar
4. Click **Apply Fix** to replace that color with `#4A6372` in the selected quadrant
5. Page reloads to show the updated tile

```shellscript
# In terminal 3, stop the app with Ctrl+C, then:
uv run python src/isometric_nyc/e2e_generation/app.py generations/v01
```

---

**User**

OK, it works but I think the softness is way too high since the color is being applied to non-water tiles

---

**Cursor**

You're right - let me reduce the default softness so it only affects pixels that are very close to the target color:

```python
"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import hashlib
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
}

# Generation queue for multiple requests
generation_queue: list[
  dict
] = []  # List of {"quadrants": [...], "type": "generate"|"render"}
queue_lock = threading.Lock()
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Ensure web server is running before generation
  ensure_web_server_running()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
  )


def process_queue_item(item: dict) -> dict:
  """Process a single queue item (generate or render)."""
  global generation_state

  quadrants = item["quadrants"]
  item_type = item["type"]

  # Convert to list of tuples
  selected_quadrants = [(q[0], q[1]) for q in quadrants]

  # Initialize generation state
  generation_state["is_generating"] = True
  generation_state["quadrants"] = selected_quadrants
  generation_state["status"] = "starting" if item_type == "generate" else "rendering"
  generation_state["message"] = f"Starting {item_type}..."
  generation_state["error"] = None
  generation_state["started_at"] = time.time()

  print(f"\n{'=' * 60}")
  print(
    f"{'🎯' if item_type == 'generate' else '🎨'} {item_type.title()} request from queue: {selected_quadrants}"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    if item_type == "generate":
      result = run_generation(conn, config, selected_quadrants)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
      return result
    else:  # render
      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def queue_worker():
  """Background worker that processes the generation queue."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    item = None

    # Get next item from queue
    with queue_lock:
      if generation_queue:
        item = generation_queue.pop(0)

    if item is None:
      # No items in queue, wait a bit and check again
      time.sleep(0.5)
      continue

    # Acquire the generation lock and process the item
    with generation_lock:
      try:
        process_queue_item(item)
      finally:
        generation_state["is_generating"] = False

    # Small delay between items
    time.sleep(0.5)

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue(quadrants: list[tuple[int, int]], item_type: str) -> dict:
  """Add a generation/render request to the queue."""
  with queue_lock:
    generation_queue.append(
      {
        "quadrants": quadrants,
        "type": item_type,
      }
    )
    queue_position = len(generation_queue)

  # Ensure the queue worker is running
  start_queue_worker()

  return {
    "success": True,
    "queued": True,
    "position": queue_position,
    "message": f"Added to queue at position {queue_position}",
  }


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  with queue_lock:
    queue_info = [
      {"quadrants": item["quadrants"], "type": item["type"]}
      for item in generation_queue
    ]
  return jsonify(
    {
      **generation_state,
      "queue": queue_info,
      "queue_length": len(queue_info),
    }
  )


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating/rendering - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "render")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state (reuse for rendering)
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "rendering"
    generation_state["message"] = "Starting render..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎨 Render request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)

      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return jsonify(
        {
          "success": True,
          "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
          "quadrants": selected_quadrants,
        }
      ), 200

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """API endpoint to generate tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "generate")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting"
    generation_state["message"] = "Starting generation..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎯 Generation request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)
      result = run_generation(conn, config, selected_quadrants)

      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        return jsonify(result), 200
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        return jsonify(result), 400

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


def ensure_web_server_running() -> None:
  """Ensure the web server for rendering is running."""
  global WEB_SERVER_PROCESS

  if WEB_SERVER_PROCESS is not None:
    # Check if still running
    if WEB_SERVER_PROCESS.poll() is None:
      return  # Still running

  # Start the web server
  print(f"🌐 Starting web server for rendering on port {WEB_SERVER_PORT}...")
  WEB_SERVER_PROCESS = start_web_server(WEB_DIR, WEB_SERVER_PORT)


def main():
  global GENERATION_DIR, WEB_SERVER_PORT

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())

```

Reduced the softness from 60.0 to 30.0. This will make the color matching much more precise - only pixels very close to the clicked color will be replaced.

Restart the app to pick up the change, then try again. If it's still too broad or now too narrow, we can adjust further (try 20-40 range).

---

**User**

Ensure that the tile is actively reloaded after the fix is applied (instead of having to cmd+r to bust cache)

---

**Cursor**

I'll update the JavaScript to bust the cache by adding a timestamp parameter when reloading after the fix:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  if (locked.length === 0 && queued.length === 0) {
    console.log("applyLockedStyles - nothing to apply, early return");
    return;
  }

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  }

  // Apply locked/queued style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  document.getElementById("applyWaterFixBtn").disabled = true;
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Reset and reload after a short delay with cache-busting
      setTimeout(() => {
        cancelWaterFix();
        // Add timestamp to bust cache
        const url = new URL(window.location.href);
        url.searchParams.set("_t", Date.now());
        window.location.href = url.toString();
      }, 1500);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Actually, the page reload with `_t` parameter won't necessarily bust the image cache. Let me improve this to directly refresh the affected tile image:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  if (locked.length === 0 && queued.length === 0) {
    console.log("applyLockedStyles - nothing to apply, early return");
    return;
  }

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  }

  // Apply locked/queued style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  document.getElementById("applyWaterFixBtn").disabled = true;
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now instead of reloading the entire page, the specific tile image will refresh in-place with a cache-busting `_t` timestamp parameter. This gives immediate visual feedback without a full page reload.

---

**User**

OK - now I need you to fix another bug with the fix button - after the first fix is applied, the button that would nromally say "fix" is stuck saying "applying"

---

**Cursor**

I see the issue - on success the button state isn't reset before `cancelWaterFix()` is called. Let me update `resetWaterFixState()` to also reset the button:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  if (locked.length === 0 && queued.length === 0) {
    console.log("applyLockedStyles - nothing to apply, early return");
    return;
  }

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  }

  // Apply locked/queued style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Fixed! The `resetWaterFixState()` function now properly resets the button text back to "Apply Fix" and removes the loading class. This will be called when `cancelWaterFix()` runs after a successful fix.

---

**User**

Another bug - the first queued outline for the generation after the generation that's currently going always disappears...

---

**Cursor**

Let me examine the queue handling code to find where the outline is being lost:

I see the issue. The code says "MERGE" but actually replaces the local queue with the server queue. When the server transitions between queue items, there's a brief moment where an item is removed from the queue but not yet showing as "generating". Let me fix this:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  if (locked.length === 0 && queued.length === 0) {
    console.log("applyLockedStyles - nothing to apply, early return");
    return;
  }

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  }

  // Apply locked/queued style to matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      // Actually merge: keep local items that aren't currently being generated
      const localQueued = getQueuedQuadrants();
      const locked = getLockedQuadrants();
      // Keep local items not in server queue and not currently locked (generating)
      const localOnly = localQueued.filter(
        ([lx, ly]) =>
          !serverQueued.some(([sx, sy]) => sx === lx && sy === ly) &&
          !locked.some(([gx, gy]) => gx === lx && gy === ly)
      );
      // Merge: server queue + local-only items (that aren't generating)
      const merged = [...serverQueued, ...localOnly];
      setQueuedQuadrants(merged);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Wait, there might be another issue. Let me also check the `removeLockedStyles` function - it might be removing queued styles too aggressively:

I see another potential issue. When `status.status === "complete"`, it calls `clearLockedQuadrants()` after the styles have been applied. And `removeLockedStyles()` removes ALL queued/locked classes before reapplying. Let me look at `applyLockedStyles` more carefully:

Let me check if there's any place clearing the queue during active generation:

```javascript
if (isLocked) {
  tile.classList.add("locked");
  tile.classList.remove("queued");
} else if (isQueued) {
```

```javascript
if (isLocked) {
  tile.classList.add("locked");
  tile.classList.remove("queued");
} else if (isQueued) {
  tile.classList.add("queued");
  tile.classList.remove("locked");
}
```

I found the issue. In `applyLockedStyles()`, if a tile is neither locked nor queued, the old classes aren't removed. And in several places, `applyLockedStyles()` is called without `removeLockedStyles()` first (like line 743). Let me fix this:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  } else {
    document.body.classList.remove("generating");
  }

  // Apply locked/queued style to matching tiles, and REMOVE from non-matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    } else {
      // Neither locked nor queued - remove both classes
      tile.classList.remove("locked");
      tile.classList.remove("queued");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      // Actually merge: keep local items that aren't currently being generated
      const localQueued = getQueuedQuadrants();
      const locked = getLockedQuadrants();
      // Keep local items not in server queue and not currently locked (generating)
      const localOnly = localQueued.filter(
        ([lx, ly]) =>
          !serverQueued.some(([sx, sy]) => sx === lx && sy === ly) &&
          !locked.some(([gx, gy]) => gx === lx && gy === ly)
      );
      // Merge: server queue + local-only items (that aren't generating)
      const merged = [...serverQueued, ...localOnly];
      setQueuedQuadrants(merged);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Fixed! The issue was that `applyLockedStyles()` only added classes to matching tiles but didn't remove classes from tiles that were no longer locked/queued. Now it explicitly removes both classes from tiles that are neither locked nor queued.

---

**User**

OK - now add a new button / route to the app that fills an entire selected quadrant with the water color - call it "water fill"

---

**Cursor**

I'll add a "Water Fill" feature that fills an entire quadrant with the water color. Let me implement this:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Generated Tiles Viewer</title>
    <link rel="stylesheet" href="/static/viewer.css" />
    <style>
      /* Dynamic grid sizing based on template variables */
      .grid {
        grid-template-columns: repeat({{ nx }}, {{ size_px }}px);
        grid-auto-rows: {{ size_px }}px;
      }
      .tile.placeholder {
        min-width: {{ size_px }}px;
        min-height: {{ size_px }}px;
      }
    </style>
  </head>
  <body>
    <!-- Config data for JS -->
    <script
      id="app-config"
      type="application/json"
      data-config='{"x": {{ x }}, "y": {{ y }}, "nx": {{ nx }}, "ny": {{ ny }}, "size_px": {{ size_px }}}'
    ></script>

    <!-- Toast notification container -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- Fixed toolbar -->
    <div class="toolbar">
      <div class="toolbar-row">
        <div class="controls">
          <label>X: <input type="number" id="x" value="{{ x }}" /></label>
          <label>Y: <input type="number" id="y" value="{{ y }}" /></label>
          <label
            >NX: <input type="number" id="nx" value="{{ nx }}" min="1" max="20"
          /></label>
          <label
            >NY: <input type="number" id="ny" value="{{ ny }}" min="1" max="20"
          /></label>
          <label
            >Size:
            <input type="number" id="sizePx" value="{{ size_px }}" step="32"
          /></label>
          <button onclick="goTo()">Go</button>
          <div class="toggle-group">
            <label
              ><input
                type="checkbox"
                id="showLines"
                {%
                if
                show_lines
                %}checked{%
                endif
                %}
                onchange="toggleLines()"
              />
              Lines</label
            >
            <label
              ><input
                type="checkbox"
                id="showCoords"
                {%
                if
                show_coords
                %}checked{%
                endif
                %}
                onchange="toggleCoords()"
              />
              Coords</label
            >
            <label
              ><input
                type="checkbox"
                id="showRender"
                {%
                if
                show_render
                %}checked{%
                endif
                %}
                onchange="toggleRender()"
              />
              Renders</label
            >
          </div>
          <div class="toggle-group tools-group">
            <span class="tools-label">Tools:</span>
            <button
              id="selectTool"
              class="tool-btn"
              onclick="toggleSelectTool()"
              title="Select quadrants (S)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
                <path d="M13 13l6 6"></path>
              </svg>
              Select
            </button>
            <button
              id="fixWaterTool"
              class="tool-btn"
              onclick="toggleFixWaterTool()"
              title="Fix water color (W)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
              </svg>
              Fix Water
            </button>
            <button
              id="waterFillTool"
              class="tool-btn"
              onclick="toggleWaterFillTool()"
              title="Fill quadrant with water (F)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                <path d="M12 8v8"></path>
                <path d="M8 12h8"></path>
              </svg>
              Water Fill
            </button>
          </div>
        </div>
        <div class="toolbar-info">
          <span>({{ x }}, {{ y }}) → ({{ x + nx - 1 }}, {{ y + ny - 1 }})</span>
          <span>{{ generation_dir }}</span>
        </div>
      </div>
      <div class="selection-status" id="selectionStatus">
        <span id="selectionCount">0 quadrants selected</span>
        <span class="selection-limit">(max 4)</span>
        <button
          id="deselectAllBtn"
          class="deselect-btn"
          onclick="deselectAll()"
          disabled
        >
          Deselect All
        </button>
        <button
          id="deleteBtn"
          class="delete-btn"
          onclick="deleteSelected()"
          disabled
        >
          Delete
        </button>
        <button
          id="renderBtn"
          class="render-btn"
          onclick="renderSelected()"
          disabled
        >
          Render
        </button>
        <button
          id="generateBtn"
          class="generate-btn"
          onclick="generateSelected()"
          disabled
        >
          Generate
        </button>
      </div>
      <div class="water-fix-status" id="waterFixStatus" style="display: none">
        <span class="water-fix-label">Target Color:</span>
        <span class="color-swatch" id="targetColorSwatch"></span>
        <span class="color-hex" id="targetColorHex"
          >Click a quadrant to pick color</span
        >
        <span class="water-fix-quadrant" id="waterFixQuadrant"></span>
        <button
          id="applyWaterFixBtn"
          class="apply-water-fix-btn"
          onclick="applyWaterFix()"
          disabled
        >
          Apply Fix
        </button>
        <button
          id="cancelWaterFixBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFix()"
        >
          Cancel
        </button>
      </div>
    </div>

    <div
      class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}"
      id="gridContainer"
    >
      <div class="grid">
        {% for dy in range(ny) %} {% for dx in range(nx) %} {% set qx = x + dx
        %} {% set qy = y + dy %} {% set has_gen = tiles.get((dx, dy), False) %}
        <div
          class="tile {% if not has_gen %}placeholder{% endif %}"
          data-coords="{{ qx }},{{ qy }}"
        >
          <span class="coords">({{ qx }}, {{ qy }})</span>
          {% if has_gen %}
          <img
            src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}"
            alt="Tile {{ qx }},{{ qy }}"
          />
          {% endif %}
        </div>
        {% endfor %} {% endfor %}
      </div>
    </div>

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

Now let me add the status bar for water fill in HTML:

```html
      <div class="water-fix-status" id="waterFixStatus" style="display: none;">
        <span class="water-fix-label">Target Color:</span>
        <span class="color-swatch" id="targetColorSwatch"></span>
        <span class="color-hex" id="targetColorHex">Click a quadrant to pick color</span>
        <span class="water-fix-quadrant" id="waterFixQuadrant"></span>
        <button
          id="applyWaterFixBtn"
          class="apply-water-fix-btn"
          onclick="applyWaterFix()"
          disabled
        >
          Apply Fix
        </button>
        <button
          id="cancelWaterFixBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFix()"
        >
          Cancel
        </button>
      </div>
      <div class="water-fill-status" id="waterFillStatus" style="display: none;">
        <span class="water-fill-label">Water Fill:</span>
        <span class="color-swatch water-color-preview"></span>
        <span class="water-fill-instruction" id="waterFillInstruction">Click a quadrant to fill with water</span>
        <button
          id="cancelWaterFillBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFill()"
        >
          Cancel
        </button>
      </div>
    </div>
```

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Generated Tiles Viewer</title>
    <link rel="stylesheet" href="/static/viewer.css" />
    <style>
      /* Dynamic grid sizing based on template variables */
      .grid {
        grid-template-columns: repeat({{ nx }}, {{ size_px }}px);
        grid-auto-rows: {{ size_px }}px;
      }
      .tile.placeholder {
        min-width: {{ size_px }}px;
        min-height: {{ size_px }}px;
      }
    </style>
  </head>
  <body>
    <!-- Config data for JS -->
    <script
      id="app-config"
      type="application/json"
      data-config='{"x": {{ x }}, "y": {{ y }}, "nx": {{ nx }}, "ny": {{ ny }}, "size_px": {{ size_px }}}'
    ></script>

    <!-- Toast notification container -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- Fixed toolbar -->
    <div class="toolbar">
      <div class="toolbar-row">
        <div class="controls">
          <label>X: <input type="number" id="x" value="{{ x }}" /></label>
          <label>Y: <input type="number" id="y" value="{{ y }}" /></label>
          <label
            >NX: <input type="number" id="nx" value="{{ nx }}" min="1" max="20"
          /></label>
          <label
            >NY: <input type="number" id="ny" value="{{ ny }}" min="1" max="20"
          /></label>
          <label
            >Size:
            <input type="number" id="sizePx" value="{{ size_px }}" step="32"
          /></label>
          <button onclick="goTo()">Go</button>
          <div class="toggle-group">
            <label
              ><input
                type="checkbox"
                id="showLines"
                {%
                if
                show_lines
                %}checked{%
                endif
                %}
                onchange="toggleLines()"
              />
              Lines</label
            >
            <label
              ><input
                type="checkbox"
                id="showCoords"
                {%
                if
                show_coords
                %}checked{%
                endif
                %}
                onchange="toggleCoords()"
              />
              Coords</label
            >
            <label
              ><input
                type="checkbox"
                id="showRender"
                {%
                if
                show_render
                %}checked{%
                endif
                %}
                onchange="toggleRender()"
              />
              Renders</label
            >
          </div>
          <div class="toggle-group tools-group">
            <span class="tools-label">Tools:</span>
            <button
              id="selectTool"
              class="tool-btn"
              onclick="toggleSelectTool()"
              title="Select quadrants (S)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
                <path d="M13 13l6 6"></path>
              </svg>
              Select
            </button>
            <button
              id="fixWaterTool"
              class="tool-btn"
              onclick="toggleFixWaterTool()"
              title="Fix water color (W)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
              </svg>
              Fix Water
            </button>
            <button
              id="waterFillTool"
              class="tool-btn"
              onclick="toggleWaterFillTool()"
              title="Fill quadrant with water (F)"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                <path d="M12 8v8"></path>
                <path d="M8 12h8"></path>
              </svg>
              Water Fill
            </button>
          </div>
        </div>
        <div class="toolbar-info">
          <span>({{ x }}, {{ y }}) → ({{ x + nx - 1 }}, {{ y + ny - 1 }})</span>
          <span>{{ generation_dir }}</span>
        </div>
      </div>
      <div class="selection-status" id="selectionStatus">
        <span id="selectionCount">0 quadrants selected</span>
        <span class="selection-limit">(max 4)</span>
        <button
          id="deselectAllBtn"
          class="deselect-btn"
          onclick="deselectAll()"
          disabled
        >
          Deselect All
        </button>
        <button
          id="deleteBtn"
          class="delete-btn"
          onclick="deleteSelected()"
          disabled
        >
          Delete
        </button>
        <button
          id="renderBtn"
          class="render-btn"
          onclick="renderSelected()"
          disabled
        >
          Render
        </button>
        <button
          id="generateBtn"
          class="generate-btn"
          onclick="generateSelected()"
          disabled
        >
          Generate
        </button>
      </div>
      <div class="water-fix-status" id="waterFixStatus" style="display: none">
        <span class="water-fix-label">Target Color:</span>
        <span class="color-swatch" id="targetColorSwatch"></span>
        <span class="color-hex" id="targetColorHex"
          >Click a quadrant to pick color</span
        >
        <span class="water-fix-quadrant" id="waterFixQuadrant"></span>
        <button
          id="applyWaterFixBtn"
          class="apply-water-fix-btn"
          onclick="applyWaterFix()"
          disabled
        >
          Apply Fix
        </button>
        <button
          id="cancelWaterFixBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFix()"
        >
          Cancel
        </button>
      </div>
      <div class="water-fill-status" id="waterFillStatus" style="display: none">
        <span class="water-fill-label">Water Fill:</span>
        <span class="color-swatch water-color-preview"></span>
        <span class="water-fill-instruction" id="waterFillInstruction"
          >Click a quadrant to fill with water</span
        >
        <button
          id="cancelWaterFillBtn"
          class="cancel-water-fix-btn"
          onclick="cancelWaterFill()"
        >
          Cancel
        </button>
      </div>
    </div>

    <div
      class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}"
      id="gridContainer"
    >
      <div class="grid">
        {% for dy in range(ny) %} {% for dx in range(nx) %} {% set qx = x + dx
        %} {% set qy = y + dy %} {% set has_gen = tiles.get((dx, dy), False) %}
        <div
          class="tile {% if not has_gen %}placeholder{% endif %}"
          data-coords="{{ qx }},{{ qy }}"
        >
          <span class="coords">({{ qx }}, {{ qy }})</span>
          {% if has_gen %}
          <img
            src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}"
            alt="Tile {{ qx }},{{ qy }}"
          />
          {% endif %}
        </div>
        {% endfor %} {% endfor %}
      </div>
    </div>

    <script src="/static/viewer.js"></script>
  </body>
</html>

```

Now let me add CSS for the water fill tool:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
  font-size: 12px;
  background: #1a1a2e;
  color: #eee;
  min-height: 100vh;
  padding: 15px;
  padding-top: 95px; /* Space for fixed toolbar */
}

.toolbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: #1a1a2e;
  padding: 10px 15px;
  z-index: 100;
  border-bottom: 1px solid #333;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.toolbar-row {
  display: flex;
  align-items: center;
  gap: 15px;
  flex-wrap: wrap;
}

.toolbar-info {
  color: #666;
  font-size: 0.75rem;
}

.toolbar-info span + span {
  margin-left: 12px;
}

.controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.controls label {
  color: #888;
  font-size: 0.8rem;
}

.controls input[type="number"] {
  width: 50px;
  padding: 5px 6px;
  border: 1px solid #333;
  border-radius: 3px;
  background: #16213e;
  color: #fff;
  font-family: inherit;
  font-size: 0.8rem;
}

.controls button {
  padding: 5px 12px;
  background: #00d9ff;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.75rem;
  font-weight: bold;
}

.controls button:hover {
  background: #00b8d4;
}

.toggle-group {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: 8px;
  padding-left: 12px;
  border-left: 1px solid #333;
}

.toggle-group label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  user-select: none;
  font-size: 0.75rem;
}

.toggle-group input[type="checkbox"] {
  width: 14px;
  height: 14px;
  accent-color: #00d9ff;
  cursor: pointer;
}

.grid-container {
  display: inline-block;
  border-radius: 8px;
  overflow: hidden;
}

.grid-container.show-lines {
  border: 2px solid #333;
}

.grid {
  display: grid;
  background: #333;
}

.grid-container.show-lines .grid {
  gap: 2px;
}

.grid-container:not(.show-lines) .grid {
  gap: 0;
  background: transparent;
}

.grid-container:not(.show-lines) {
  border: none;
}

.tile {
  position: relative;
  background: #2a2a4a;
  display: flex;
  align-items: center;
  justify-content: center;
}

.tile img {
  display: block;
  max-width: 100%;
  height: auto;
}

.tile.placeholder {
  background: #3a3a5a;
}

.tile .coords {
  position: absolute;
  top: 6px;
  left: 6px;
  background: rgba(0, 0, 0, 0.7);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.65rem;
  color: #00d9ff;
  transition: opacity 0.2s;
}

.tile.placeholder .coords {
  color: #666;
}

.grid-container:not(.show-coords) .tile .coords {
  opacity: 0;
}

/* Tool button styles */
.tools-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.tools-label {
  color: #666;
  font-size: 0.7rem;
}

.tool-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: #2a2a3e;
  color: #666;
  border: 1px solid #444;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  transition: all 0.15s ease;
}

.tool-btn:hover {
  background: #3a3a4e;
  color: #999;
  border-color: #555;
}

.tool-btn.active {
  background: #10b981;
  color: #fff;
  border-color: #10b981;
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.tool-btn.active:hover {
  background: #059669;
  border-color: #059669;
}

.tool-btn svg {
  width: 12px;
  height: 12px;
}

/* Selection styles */
.tile.selected {
  outline: 3px solid #ff3333;
  outline-offset: -3px;
  z-index: 10;
}

.grid-container.show-lines .tile.selected {
  outline-color: #ff3333;
}

.tile.selectable {
  cursor: pointer;
}

.tile.placeholder.selected {
  background: rgba(255, 51, 51, 0.15);
}

/* Locked tiles (during generation) */
.tile.locked {
  outline: 3px solid #a855f7;
  outline-offset: -3px;
  z-index: 10;
}

.grid-container.show-lines .tile.locked {
  outline-color: #a855f7;
}

.tile.placeholder.locked {
  background: rgba(168, 85, 247, 0.2);
}

/* Queued tiles (in queue, waiting to be generated) */
.tile.queued {
  outline: 3px dashed #c084fc !important;
  outline-offset: -3px;
  z-index: 9;
  animation: pulse-queue 1.5s ease-in-out infinite;
  /* Fallback: box-shadow for browsers that don't render dashed outlines well */
  box-shadow: inset 0 0 0 3px rgba(192, 132, 252, 0.5);
}

@keyframes pulse-queue {
  0%,
  100% {
    outline-color: #c084fc;
    box-shadow: inset 0 0 0 3px rgba(192, 132, 252, 0.5);
  }
  50% {
    outline-color: #8b5cf6;
    box-shadow: inset 0 0 0 3px rgba(139, 92, 246, 0.7);
  }
}

.grid-container.show-lines .tile.queued {
  outline-color: #c084fc;
}

.tile.placeholder.queued {
  background: rgba(168, 85, 247, 0.3) !important;
}

/* Allow selection while generating - only disable clicking on locked/queued tiles */
body.generating .tile.locked.selectable,
body.generating .tile.queued.selectable {
  cursor: not-allowed;
}

/* Selection status bar */
.selection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  background: rgba(255, 51, 51, 0.1);
  border: 1px solid rgba(255, 51, 51, 0.3);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #ff6666;
  margin-top: 8px;
}

.selection-status.empty {
  background: transparent;
  border-color: #333;
  color: #666;
}

.selection-status.generating {
  background: rgba(168, 85, 247, 0.1);
  border-color: rgba(168, 85, 247, 0.4);
  color: #a855f7;
}

.selection-limit {
  color: #888;
  font-size: 0.65rem;
}

.deselect-btn {
  padding: 3px 8px;
  background: #ff3333;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  margin-left: auto;
  transition: all 0.2s;
}

.deselect-btn:hover:not(:disabled) {
  background: #ff5555;
}

.deselect-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
}

.delete-btn {
  padding: 3px 8px;
  background: #dc2626;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  transition: all 0.2s;
}

.delete-btn:hover:not(:disabled) {
  background: #b91c1c;
}

.delete-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
}

.render-btn {
  padding: 4px 12px;
  background: #f59e0b;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
}

.render-btn:hover:not(:disabled) {
  background: #d97706;
}

.render-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

.render-btn.loading {
  background: #666;
  cursor: wait;
}

.render-btn .spinner {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-left: 4px;
  vertical-align: middle;
}

.generate-btn {
  padding: 4px 12px;
  background: #00d9ff;
  color: #1a1a2e;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
}

.generate-btn:hover:not(:disabled) {
  background: #00b8d4;
}

.generate-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

/* Toast notification system */
.toast-container {
  position: fixed;
  bottom: 15px;
  right: 15px;
  z-index: 1000;
  display: flex;
  flex-direction: column-reverse;
  gap: 8px;
  max-width: 320px;
}

.toast {
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 0.75rem;
  animation: slideIn 0.3s ease-out;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.toast.success {
  background: linear-gradient(135deg, #065f46 0%, #047857 100%);
  border: 1px solid #10b981;
  color: #d1fae5;
}

.toast.error {
  background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
  border: 1px solid #ef4444;
  color: #fee2e2;
}

.toast.info {
  background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%);
  border: 1px solid #3b82f6;
  color: #dbeafe;
}

.toast.loading {
  background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
  border: 1px solid #6b7280;
  color: #e5e7eb;
}

.toast-icon {
  flex-shrink: 0;
  font-size: 1rem;
}

.toast-content {
  flex: 1;
}

.toast-title {
  font-weight: 600;
  margin-bottom: 2px;
}

.toast-message {
  opacity: 0.9;
  font-size: 0.7rem;
}

.toast-close {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  opacity: 0.7;
  font-size: 1rem;
  padding: 0;
  line-height: 1;
}

.toast-close:hover {
  opacity: 1;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@keyframes slideOut {
  from {
    transform: translateX(0);
    opacity: 1;
  }
  to {
    transform: translateX(100%);
    opacity: 0;
  }
}

.toast.removing {
  animation: slideOut 0.3s ease-in forwards;
}

/* Loading spinner for generate button */
.generate-btn.loading {
  background: #666;
  color: #fff;
  cursor: wait;
}

.generate-btn .spinner {
  display: inline-block;
  width: 10px;
  height: 10px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-left: 4px;
  vertical-align: middle;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* Water fix tool styles */
.water-fix-status {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #60a5fa;
  margin-top: 8px;
}

.water-fix-label {
  color: #93c5fd;
  font-weight: 500;
}

.color-swatch {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.3), inset 0 0 0 1px rgba(0, 0, 0, 0.1);
  background: #333;
}

.color-swatch.has-color {
  cursor: pointer;
}

.color-hex {
  font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
  font-size: 0.75rem;
  color: #93c5fd;
  min-width: 180px;
}

.water-fix-quadrant {
  color: #60a5fa;
  font-size: 0.7rem;
}

.apply-water-fix-btn {
  padding: 4px 12px;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.7rem;
  font-weight: bold;
  transition: all 0.2s;
  margin-left: auto;
}

.apply-water-fix-btn:hover:not(:disabled) {
  background: #2563eb;
}

.apply-water-fix-btn:disabled {
  background: #444;
  color: #666;
  cursor: not-allowed;
  font-weight: normal;
}

.apply-water-fix-btn.loading {
  background: #666;
  cursor: wait;
}

.cancel-water-fix-btn {
  padding: 4px 8px;
  background: #4b5563;
  color: #d1d5db;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.65rem;
  transition: all 0.2s;
}

.cancel-water-fix-btn:hover {
  background: #6b7280;
}

/* Fix water tool active state */
.tile.fix-water-selectable {
  cursor: crosshair;
}

.tile.fix-water-selectable:hover {
  outline: 2px solid #3b82f6;
  outline-offset: -2px;
}

.tile.water-fix-selected {
  outline: 3px solid #3b82f6;
  outline-offset: -3px;
  z-index: 10;
}

/* Water fill tool styles */
.water-fill-status {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  background: rgba(74, 99, 114, 0.15);
  border: 1px solid rgba(74, 99, 114, 0.4);
  border-radius: 4px;
  font-size: 0.7rem;
  color: #7ba3b8;
  margin-top: 8px;
}

.water-fill-label {
  color: #93c5fd;
  font-weight: 500;
}

.water-color-preview {
  background: #4A6372 !important;
}

.water-fill-instruction {
  color: #7ba3b8;
}

.tile.water-fill-selectable {
  cursor: pointer;
}

.tile.water-fill-selectable:hover {
  outline: 2px solid #4A6372;
  outline-offset: -2px;
}

```

Now let me add the JavaScript for the water fill tool. First let me find where to add it:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  } else {
    document.body.classList.remove("generating");
  }

  // Apply locked/queued style to matching tiles, and REMOVE from non-matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    } else {
      // Neither locked nor queued - remove both classes
      tile.classList.remove("locked");
      tile.classList.remove("queued");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast("loading", "Filling with water...", `Processing quadrant (${qx}, ${qy})`);

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      // Actually merge: keep local items that aren't currently being generated
      const localQueued = getQueuedQuadrants();
      const locked = getLockedQuadrants();
      // Keep local items not in server queue and not currently locked (generating)
      const localOnly = localQueued.filter(
        ([lx, ly]) =>
          !serverQueued.some(([sx, sy]) => sx === lx && sy === ly) &&
          !locked.some(([gx, gy]) => gx === lx && gy === ly)
      );
      // Merge: server queue + local-only items (that aren't generating)
      const merged = [...serverQueued, ...localOnly];
      setQueuedQuadrants(merged);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now let me add the keyboard shortcut and update the tile click handler:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  } else {
    document.body.classList.remove("generating");
  }

  // Apply locked/queued style to matching tiles, and REMOVE from non-matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    } else {
      // Neither locked nor queued - remove both classes
      tile.classList.remove("locked");
      tile.classList.remove("queued");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast("loading", "Filling with water...", `Processing quadrant (${qx}, ${qy})`);

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      // Actually merge: keep local items that aren't currently being generated
      const localQueued = getQueuedQuadrants();
      const locked = getLockedQuadrants();
      // Keep local items not in server queue and not currently locked (generating)
      const localOnly = localQueued.filter(
        ([lx, ly]) =>
          !serverQueued.some(([sx, sy]) => sx === lx && sy === ly) &&
          !locked.some(([gx, gy]) => gx === lx && gy === ly)
      );
      // Merge: server queue + local-only items (that aren't generating)
      const merged = [...serverQueued, ...localOnly];
      setQueuedQuadrants(merged);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now update the tile click handler:

```javascript
// Get config from data attributes
const config = JSON.parse(document.getElementById("app-config").dataset.config);

// Locked quadrants storage key
const LOCKED_QUADRANTS_KEY = "generatingQuadrants";
const QUEUED_QUADRANTS_KEY = "queuedQuadrants";

function getLockedQuadrants() {
  try {
    const stored = localStorage.getItem(LOCKED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setLockedQuadrants(quadrants) {
  localStorage.setItem(LOCKED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearLockedQuadrants() {
  localStorage.removeItem(LOCKED_QUADRANTS_KEY);
}

function getQueuedQuadrants() {
  try {
    const stored = localStorage.getItem(QUEUED_QUADRANTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

function setQueuedQuadrants(quadrants) {
  localStorage.setItem(QUEUED_QUADRANTS_KEY, JSON.stringify(quadrants));
}

function clearQueuedQuadrants() {
  localStorage.removeItem(QUEUED_QUADRANTS_KEY);
}

function applyLockedStyles() {
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();

  console.log("applyLockedStyles - locked:", locked, "queued:", queued);

  // Add generating class to body only if actively generating
  if (locked.length > 0) {
    document.body.classList.add("generating");
  } else {
    document.body.classList.remove("generating");
  }

  // Apply locked/queued style to matching tiles, and REMOVE from non-matching tiles
  document.querySelectorAll(".tile").forEach((tile) => {
    const [qx, qy] = tile.dataset.coords.split(",").map(Number);
    const isLocked = locked.some(([lx, ly]) => lx === qx && ly === qy);
    const isQueued = queued.some(([lx, ly]) => lx === qx && ly === qy);

    if (isLocked) {
      tile.classList.add("locked");
      tile.classList.remove("queued");
    } else if (isQueued) {
      console.log("Applying queued style to tile:", qx, qy);
      tile.classList.add("queued");
      tile.classList.remove("locked");
    } else {
      // Neither locked nor queued - remove both classes
      tile.classList.remove("locked");
      tile.classList.remove("queued");
    }
  });
}

function removeLockedStyles() {
  console.log("removeLockedStyles called");
  document.body.classList.remove("generating");
  document.querySelectorAll(".tile.locked").forEach((tile) => {
    console.log("Removing locked from tile:", tile.dataset.coords);
    tile.classList.remove("locked");
  });
  document.querySelectorAll(".tile.queued").forEach((tile) => {
    console.log("Removing queued from tile:", tile.dataset.coords);
    tile.classList.remove("queued");
  });
}

function getParams() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const nx = document.getElementById("nx").value;
  const ny = document.getElementById("ny").value;
  const sizePx = document.getElementById("sizePx").value;
  const showLines = document.getElementById("showLines").checked ? "1" : "0";
  const showCoords = document.getElementById("showCoords").checked ? "1" : "0";
  const showRender = document.getElementById("showRender").checked ? "1" : "0";
  return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
}

function goTo() {
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

function navigate(dx, dy) {
  const params = getParams();
  const x = parseInt(params.x) + dx;
  const y = parseInt(params.y) + dy;
  window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
}

function toggleLines() {
  const container = document.getElementById("gridContainer");
  const showLines = document.getElementById("showLines").checked;
  container.classList.toggle("show-lines", showLines);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("lines", showLines ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleCoords() {
  const container = document.getElementById("gridContainer");
  const showCoords = document.getElementById("showCoords").checked;
  container.classList.toggle("show-coords", showCoords);

  // Update URL without reload
  const url = new URL(window.location);
  url.searchParams.set("coords", showCoords ? "1" : "0");
  history.replaceState({}, "", url);
}

function toggleRender() {
  // This requires a page reload to fetch different data
  const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } =
    getParams();
  window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
}

// Keyboard navigation
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;

  switch (e.key) {
    case "ArrowLeft":
      navigate(-1, 0);
      break;
    case "ArrowRight":
      navigate(1, 0);
      break;
    case "ArrowUp":
      navigate(0, -1);
      break;
    case "ArrowDown":
      navigate(0, 1);
      break;
    case "l":
    case "L":
      document.getElementById("showLines").click();
      break;
    case "c":
    case "C":
      document.getElementById("showCoords").click();
      break;
    case "g":
    case "G":
      document.getElementById("showRender").click();
      break;
    case "s":
    case "S":
      toggleSelectTool();
      break;
    case "w":
    case "W":
      toggleFixWaterTool();
      break;
    case "f":
    case "F":
      toggleWaterFillTool();
      break;
    case "Escape":
      if (selectToolActive) toggleSelectTool();
      if (fixWaterToolActive) cancelWaterFix();
      if (waterFillToolActive) cancelWaterFill();
      break;
  }
});

// Select tool state
let selectToolActive = false;
const selectedQuadrants = new Set();
const MAX_SELECTION = 4;

function toggleSelectTool() {
  // Deactivate fix water tool if active
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  selectToolActive = !selectToolActive;
  const btn = document.getElementById("selectTool");
  const tiles = document.querySelectorAll(".tile");

  if (selectToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => tile.classList.add("selectable"));
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => tile.classList.remove("selectable"));
  }
}

// Fix water tool state
let fixWaterToolActive = false;
let fixWaterTargetColor = null;
let fixWaterQuadrant = null;

function toggleFixWaterTool() {
  // Deactivate select tool if active
  if (selectToolActive) {
    toggleSelectTool();
  }

  fixWaterToolActive = !fixWaterToolActive;
  const btn = document.getElementById("fixWaterTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFixStatus = document.getElementById("waterFixStatus");

  if (fixWaterToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      // Only make tiles with images selectable
      if (tile.querySelector("img")) {
        tile.classList.add("fix-water-selectable");
      }
    });
    // Show water fix status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFixStatus.style.display = "flex";
    // Reset state
    resetWaterFixState();
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("fix-water-selectable");
      tile.classList.remove("water-fix-selected");
    });
    // Hide water fix status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFixStatus.style.display = "none";
  }
}

function resetWaterFixState() {
  fixWaterTargetColor = null;
  fixWaterQuadrant = null;
  document.getElementById("targetColorSwatch").style.background = "#333";
  document.getElementById("targetColorSwatch").classList.remove("has-color");
  document.getElementById("targetColorHex").textContent =
    "Click a quadrant to pick color";
  document.getElementById("waterFixQuadrant").textContent = "";
  // Reset button state
  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.remove("loading");
  btn.textContent = "Apply Fix";
  document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
    tile.classList.remove("water-fix-selected");
  });
}

function cancelWaterFix() {
  if (fixWaterToolActive) {
    toggleFixWaterTool();
  }
}

function rgbToHex(r, g, b) {
  return (
    "#" +
    [r, g, b]
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
      .toUpperCase()
  );
}

function getPixelColorFromImage(img, x, y) {
  // Create an off-screen canvas
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);

  // Get the pixel data at the clicked position
  const pixelData = ctx.getImageData(x, y, 1, 1).data;

  return {
    r: pixelData[0],
    g: pixelData[1],
    b: pixelData[2],
    a: pixelData[3],
  };
}

function handleFixWaterClick(tileEl, e) {
  if (!fixWaterToolActive) return;

  const img = tileEl.querySelector("img");
  if (!img) {
    showToast("error", "No image", "This quadrant has no generation to fix");
    return;
  }

  // Get coordinates
  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Calculate click position relative to the image
  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // Scale to natural image dimensions
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.floor(clickX * scaleX);
  const imgY = Math.floor(clickY * scaleY);

  // Ensure we're within bounds
  if (
    imgX < 0 ||
    imgX >= img.naturalWidth ||
    imgY < 0 ||
    imgY >= img.naturalHeight
  ) {
    console.log("Click outside image bounds");
    return;
  }

  try {
    // Get the pixel color
    const color = getPixelColorFromImage(img, imgX, imgY);
    const hex = rgbToHex(color.r, color.g, color.b);

    console.log(
      `Picked color at (${imgX}, ${imgY}) in quadrant (${qx}, ${qy}): RGB(${color.r}, ${color.g}, ${color.b}) = ${hex}`
    );

    // Update state
    fixWaterTargetColor = hex;
    fixWaterQuadrant = { x: qx, y: qy };

    // Update UI
    document.getElementById("targetColorSwatch").style.background = hex;
    document.getElementById("targetColorSwatch").classList.add("has-color");
    document.getElementById(
      "targetColorHex"
    ).textContent = `${hex} — RGB(${color.r}, ${color.g}, ${color.b})`;
    document.getElementById(
      "waterFixQuadrant"
    ).textContent = `Quadrant (${qx}, ${qy})`;
    document.getElementById("applyWaterFixBtn").disabled = false;

    // Update selected tile visual
    document.querySelectorAll(".tile.water-fix-selected").forEach((tile) => {
      tile.classList.remove("water-fix-selected");
    });
    tileEl.classList.add("water-fix-selected");

    showToast("info", "Color picked", `Target color: ${hex} at (${qx}, ${qy})`);
  } catch (error) {
    console.error("Error picking color:", error);
    showToast(
      "error",
      "Error picking color",
      "Could not read pixel color. Try again."
    );
  }
}

async function applyWaterFix() {
  if (!fixWaterTargetColor || !fixWaterQuadrant) {
    showToast("error", "No color selected", "Pick a color first");
    return;
  }

  // Default replacement color - a nice blue water color
  const replacementColor = "#2A4A5F";

  const btn = document.getElementById("applyWaterFixBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.textContent = "Applying...";

  showToast(
    "loading",
    "Applying water fix...",
    `Replacing ${fixWaterTargetColor} in (${fixWaterQuadrant.x}, ${fixWaterQuadrant.y})`
  );

  try {
    const response = await fetch("/api/fix-water", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: fixWaterQuadrant.x,
        y: fixWaterQuadrant.y,
        target_color: fixWaterTargetColor,
        replacement_color: replacementColor,
      }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast(
        "success",
        "Water fix applied!",
        result.message || "Color replaced successfully"
      );

      // Refresh the specific tile image immediately with cache-busting
      const { x, y } = fixWaterQuadrant;
      const tile = document.querySelector(`.tile[data-coords="${x},${y}"]`);
      if (tile) {
        const img = tile.querySelector("img");
        if (img) {
          // Add timestamp to bust browser cache
          const currentSrc = new URL(img.src);
          currentSrc.searchParams.set("_t", Date.now());
          img.src = currentSrc.toString();
        }
      }

      // Reset the tool after a short delay
      setTimeout(() => {
        cancelWaterFix();
      }, 1000);
    } else {
      showToast("error", "Water fix failed", result.error || "Unknown error");
      btn.disabled = false;
      btn.classList.remove("loading");
      btn.textContent = "Apply Fix";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fix error:", error);
    showToast("error", "Request failed", error.message);
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = "Apply Fix";
  }
}

// Water Fill tool - fills entire quadrant with water color
let waterFillToolActive = false;

function toggleWaterFillTool() {
  // Deactivate other tools
  if (selectToolActive) {
    toggleSelectTool();
  }
  if (fixWaterToolActive) {
    cancelWaterFix();
  }

  waterFillToolActive = !waterFillToolActive;
  const btn = document.getElementById("waterFillTool");
  const tiles = document.querySelectorAll(".tile");
  const selectionStatus = document.getElementById("selectionStatus");
  const waterFillStatus = document.getElementById("waterFillStatus");

  if (waterFillToolActive) {
    btn.classList.add("active");
    tiles.forEach((tile) => {
      tile.classList.add("water-fill-selectable");
    });
    // Show water fill status bar, hide selection status
    selectionStatus.style.display = "none";
    waterFillStatus.style.display = "flex";
  } else {
    btn.classList.remove("active");
    tiles.forEach((tile) => {
      tile.classList.remove("water-fill-selectable");
    });
    // Hide water fill status bar, show selection status
    selectionStatus.style.display = "flex";
    waterFillStatus.style.display = "none";
  }
}

function cancelWaterFill() {
  if (waterFillToolActive) {
    toggleWaterFillTool();
  }
}

async function handleWaterFillClick(tileEl) {
  if (!waterFillToolActive) return;

  const coords = tileEl.dataset.coords.split(",").map(Number);
  const [qx, qy] = coords;

  // Confirm action
  if (!confirm(`Fill quadrant (${qx}, ${qy}) entirely with water color?`)) {
    return;
  }

  const instruction = document.getElementById("waterFillInstruction");
  instruction.textContent = `Filling (${qx}, ${qy})...`;

  showToast("loading", "Filling with water...", `Processing quadrant (${qx}, ${qy})`);

  try {
    const response = await fetch("/api/water-fill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: qx, y: qy }),
    });

    const result = await response.json();
    clearLoadingToasts();

    if (result.success) {
      showToast("success", "Water fill complete!", result.message);

      // Refresh the tile image
      const img = tileEl.querySelector("img");
      if (img) {
        const currentSrc = new URL(img.src);
        currentSrc.searchParams.set("_t", Date.now());
        img.src = currentSrc.toString();
      }

      instruction.textContent = "Click a quadrant to fill with water";
    } else {
      showToast("error", "Water fill failed", result.error || "Unknown error");
      instruction.textContent = "Click a quadrant to fill with water";
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Water fill error:", error);
    showToast("error", "Request failed", error.message);
    instruction.textContent = "Click a quadrant to fill with water";
  }
}

function updateSelectionStatus() {
  const count = selectedQuadrants.size;
  const countEl = document.getElementById("selectionCount");
  const limitEl = document.querySelector(".selection-limit");
  const statusEl = document.getElementById("selectionStatus");
  const deselectBtn = document.getElementById("deselectAllBtn");
  const deleteBtn = document.getElementById("deleteBtn");
  const renderBtn = document.getElementById("renderBtn");
  const generateBtn = document.getElementById("generateBtn");

  // Check if we're generating/rendering
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isProcessing = isGenerating || isRendering;

  console.log(
    "updateSelectionStatus - locked:",
    locked,
    "queued:",
    queued,
    "isProcessing:",
    isProcessing
  );

  let statusParts = [];

  // Show current processing status
  if (locked.length > 0 && isProcessing) {
    const action = isRendering ? "Rendering" : "Generating";
    const coordsStr = locked.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`${action} ${coordsStr}`);
  }

  // Show queue count
  if (queued.length > 0) {
    const queueCoords = queued.map(([x, y]) => `(${x},${y})`).join(" ");
    statusParts.push(`📋 Queue: ${queued.length} - ${queueCoords}`);
  }

  // Show selection count
  if (count > 0) {
    statusParts.push(`${count} selected`);
  }

  let statusText;
  if (statusParts.length > 0) {
    statusText = statusParts.join(" • ");
  } else {
    statusText = "0 quadrants selected";
  }

  countEl.textContent = statusText;

  // Update status bar styling
  if (locked.length > 0 || queued.length > 0) {
    if (limitEl) limitEl.style.display = "none";
    statusEl.classList.remove("empty");
    statusEl.classList.add("generating");
  } else {
    if (limitEl) limitEl.style.display = "";
    statusEl.classList.toggle("empty", count === 0);
    statusEl.classList.remove("generating");
  }

  // Enable buttons for selection (can add to queue even during processing)
  deselectBtn.disabled = count === 0;
  deleteBtn.disabled = count === 0;
  renderBtn.disabled = count === 0;
  generateBtn.disabled = count === 0;
}

// Toast notification system
function showToast(type, title, message, duration = 5000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️",
    loading: "⏳",
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  container.appendChild(toast);

  // Auto-remove after duration (except for loading toasts)
  if (type !== "loading" && duration > 0) {
    setTimeout(() => {
      toast.classList.add("removing");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return toast;
}

function clearLoadingToasts() {
  document.querySelectorAll(".toast.loading").forEach((t) => t.remove());
}

// Generation/Render state
let isGenerating = false;
let isRendering = false;

async function deleteSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  // Confirm deletion
  const coordsStr = coords.map(([x, y]) => `(${x},${y})`).join(", ");
  if (!confirm(`Delete generation data for ${coordsStr}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    if (result.success) {
      showToast("success", "Deleted", result.message);
      // Deselect and refresh
      deselectAll();
      location.reload();
    } else {
      showToast("error", "Delete failed", result.error);
    }
  } catch (error) {
    console.error("Delete error:", error);
    showToast("error", "Delete failed", error.message);
  }
}

async function generateSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Generate requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log(
    "generateSelected - alreadyGenerating:",
    alreadyGenerating,
    "isGenerating:",
    isGenerating,
    "isRendering:",
    isRendering
  );

  if (alreadyGenerating) {
    console.log("Adding to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    setQueuedQuadrants(newQueued);
    console.log("Queue updated in localStorage:", newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued:", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s)`
    );

    updateSelectionStatus();
  } else {
    console.log("Starting new generation (not queued):", coords);
    // Set loading state BEFORE the request
    isGenerating = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const generateBtn = document.getElementById("generateBtn");
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Generating tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a minute.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Generation queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Generation queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - generation finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Generation complete!",
        result.message ||
          `Successfully generated ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetGenerateButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Generation failed",
        result.error || "Unknown error occurred."
      );
      resetGenerateButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Generation error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetGenerateButton();
  }
}

async function renderSelected() {
  if (selectedQuadrants.size === 0) return;

  const coords = Array.from(selectedQuadrants).map((s) => {
    const [x, y] = s.split(",").map(Number);
    return [x, y];
  });

  console.log("Render requested for:", coords);

  // Clear selection
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  selectedQuadrants.clear();

  // Check if already generating - if so, these will be queued
  const alreadyGenerating = isGenerating || isRendering;
  console.log("renderSelected - alreadyGenerating:", alreadyGenerating);

  if (alreadyGenerating) {
    console.log("Adding render to queue:", coords);
    // Add to queued quadrants visually BEFORE the request
    const currentQueued = getQueuedQuadrants();
    const newQueued = [...currentQueued, ...coords];
    console.log("Queue updated:", newQueued);
    setQueuedQuadrants(newQueued);

    // Mark tiles as queued (dashed purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      console.log("Marking tile as queued (render):", qx, qy, tile);
      if (tile) {
        tile.classList.add("queued");
        console.log("Tile classes after adding queued:", tile.className);
      }
    });

    // Show immediate feedback
    showToast(
      "info",
      "Adding to queue...",
      `Queueing ${coords.length} quadrant(s) for render`
    );

    updateSelectionStatus();
  } else {
    // Set loading state BEFORE the request
    isRendering = true;
    setLockedQuadrants(coords);
    document.body.classList.add("generating");

    // Mark tiles as locked (solid purple)
    coords.forEach(([qx, qy]) => {
      const tile = document.querySelector(`.tile[data-coords="${qx},${qy}"]`);
      if (tile) {
        tile.classList.add("locked");
      }
    });

    const renderBtn = document.getElementById("renderBtn");
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';

    updateSelectionStatus();

    showToast(
      "loading",
      "Rendering tiles...",
      `Processing ${coords.length} quadrant${
        coords.length > 1 ? "s" : ""
      }. This may take a moment.`
    );
  }

  // Start polling for status updates
  startStatusPolling();

  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ quadrants: coords }),
    });

    const result = await response.json();

    // Request was queued (202 Accepted)
    if (response.status === 202 && result.queued) {
      console.log("Render queued at position:", result.position);

      showToast(
        "info",
        "Added to queue",
        `Render queued at position ${result.position}.`
      );
      return;
    }

    // Request completed (200 OK) - render finished
    if (response.ok && result.success) {
      clearLoadingToasts();
      showToast(
        "success",
        "Render complete!",
        result.message ||
          `Successfully rendered ${coords.length} quadrant${
            coords.length > 1 ? "s" : ""
          }.`
      );

      // Check if there are more items in queue from the status
      const statusResponse = await fetch("/api/status");
      const status = await statusResponse.json();

      if (status.queue_length > 0) {
        // More items in queue, reset button but keep polling
        resetRenderButton();
        // Re-apply queued styles
        if (status.queue && status.queue.length > 0) {
          const serverQueued = status.queue.flatMap((item) => item.quadrants);
          setQueuedQuadrants(serverQueued);
          applyLockedStyles();
        }
      } else {
        // No more items, stop polling and reload
        stopStatusPolling();
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      }
    } else {
      clearLoadingToasts();
      showToast(
        "error",
        "Render failed",
        result.error || "Unknown error occurred."
      );
      resetRenderButton();
    }
  } catch (error) {
    clearLoadingToasts();
    console.error("Render error:", error);
    showToast(
      "error",
      "Request failed",
      error.message || "Could not connect to server."
    );
    resetRenderButton();
  }
}

function deselectAll() {
  selectedQuadrants.clear();
  document.querySelectorAll(".tile.selected").forEach((tile) => {
    tile.classList.remove("selected");
  });
  updateSelectionStatus();
  console.log("Deselected all quadrants");
}

function toggleTileSelection(tileEl, qx, qy) {
  if (!selectToolActive) return;

  // Check if this tile is currently being generated or in the queue
  const key = `${qx},${qy}`;
  const locked = getLockedQuadrants();
  const queued = getQueuedQuadrants();
  const isLockedOrQueued =
    locked.some(([lx, ly]) => lx === qx && ly === qy) ||
    queued.some(([lx, ly]) => lx === qx && ly === qy);
  if (isLockedOrQueued) {
    console.log(
      `Cannot select quadrant (${qx}, ${qy}) - currently generating or in queue`
    );
    return;
  }

  if (selectedQuadrants.has(key)) {
    selectedQuadrants.delete(key);
    tileEl.classList.remove("selected");
    console.log(`Deselected quadrant (${qx}, ${qy})`);
  } else {
    // Check if we've hit the max selection limit
    if (selectedQuadrants.size >= MAX_SELECTION) {
      console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
      return;
    }
    selectedQuadrants.add(key);
    tileEl.classList.add("selected");
    console.log(`Selected quadrant (${qx}, ${qy})`);
  }

  updateSelectionStatus();

  // Log current selection
  if (selectedQuadrants.size > 0) {
    console.log("Selected:", Array.from(selectedQuadrants).join("; "));
  }
}

// Setup tile click handlers
document.querySelectorAll(".tile").forEach((tile) => {
  tile.addEventListener("click", (e) => {
    // Handle fix water tool clicks
    if (fixWaterToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleFixWaterClick(tile, e);
      return;
    }

    // Handle water fill tool clicks
    if (waterFillToolActive) {
      e.preventDefault();
      e.stopPropagation();
      handleWaterFillClick(tile);
      return;
    }

    // Handle select tool clicks
    if (!selectToolActive) return;
    e.preventDefault();
    e.stopPropagation();

    const coords = tile.dataset.coords.split(",").map(Number);
    toggleTileSelection(tile, coords[0], coords[1]);
  });
});

// Initialize selection status
updateSelectionStatus();

// Status polling for generation progress
let statusPollInterval = null;

function startStatusPolling() {
  if (statusPollInterval) return;
  statusPollInterval = setInterval(checkGenerationStatus, 1000);
}

function stopStatusPolling() {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
    statusPollInterval = null;
  }
}

async function checkGenerationStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    console.log("Status poll:", status);

    // Sync locked quadrants from server (currently processing)
    if (
      status.is_generating &&
      status.quadrants &&
      status.quadrants.length > 0
    ) {
      setLockedQuadrants(status.quadrants);
    } else if (!status.is_generating) {
      clearLockedQuadrants();
    }

    // Sync queue state from server
    // MERGE with local state to avoid losing items in transit
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      // Actually merge: keep local items that aren't currently being generated
      const localQueued = getQueuedQuadrants();
      const locked = getLockedQuadrants();
      // Keep local items not in server queue and not currently locked (generating)
      const localOnly = localQueued.filter(
        ([lx, ly]) =>
          !serverQueued.some(([sx, sy]) => sx === lx && sy === ly) &&
          !locked.some(([gx, gy]) => gx === lx && gy === ly)
      );
      // Merge: server queue + local-only items (that aren't generating)
      const merged = [...serverQueued, ...localOnly];
      setQueuedQuadrants(merged);
    } else if (status.status === "idle" && !status.is_generating) {
      // Only clear queue when truly idle (no race condition with pending requests)
      clearQueuedQuadrants();
    }
    // If generating but queue is empty, keep local queue (items may be in transit)

    // Always re-apply styles to sync with server state
    removeLockedStyles();
    applyLockedStyles();

    if (status.is_generating) {
      // Update UI to show generation/render in progress
      setProcessingUI(status);
    } else {
      // Current operation finished

      const isRenderOp = status.status === "rendering" || isRendering;
      const opName = isRenderOp ? "Render" : "Generation";

      if (status.status === "complete") {
        clearLoadingToasts();
        showToast("success", `${opName} complete!`, status.message);

        // Clear the locked state for the completed operation
        clearLockedQuadrants();

        // Check if there are more items in the queue
        if (status.queue_length > 0) {
          showToast(
            "info",
            "Processing queue",
            `${status.queue_length} more item(s) in queue...`
          );
          // Keep button in loading state - next queue item will start processing
          // The next status poll will update UI when new item starts
        } else {
          // No more items, stop polling and reload
          stopStatusPolling();
          setTimeout(() => window.location.reload(), 1500);
        }
      } else if (status.status === "error" && status.error) {
        clearLoadingToasts();
        showToast("error", `${opName} failed`, status.error);

        // Clear locked state
        clearLockedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;

        // Continue polling if there are more items in queue
        if (status.queue_length === 0) {
          stopStatusPolling();
        }
      } else if (status.status === "idle" && status.queue_length === 0) {
        // Idle with no queue - fully clean up
        stopStatusPolling();
        clearLockedQuadrants();
        clearQueuedQuadrants();
        removeLockedStyles();

        const generateBtn = document.getElementById("generateBtn");
        const renderBtn = document.getElementById("renderBtn");
        generateBtn.classList.remove("loading");
        generateBtn.innerHTML = "Generate";
        renderBtn.classList.remove("loading");
        renderBtn.innerHTML = "Render";
        isGenerating = false;
        isRendering = false;
      }
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Status check failed:", error);
  }
}

function setProcessingUI(status) {
  // Determine if this is a render or generate operation
  const isRenderOp = status.status === "rendering";
  const generateBtn = document.getElementById("generateBtn");
  const renderBtn = document.getElementById("renderBtn");

  // Sync locked quadrants from server
  if (status.quadrants && status.quadrants.length > 0) {
    setLockedQuadrants(status.quadrants);
  }

  // Apply locked/queued styles to tiles
  applyLockedStyles();

  if (isRenderOp) {
    renderBtn.classList.add("loading");
    renderBtn.innerHTML = 'Rendering<span class="spinner"></span>';
    isRendering = true;
    isGenerating = false;
  } else {
    generateBtn.classList.add("loading");
    generateBtn.innerHTML = 'Generating<span class="spinner"></span>';
    isGenerating = true;
    isRendering = false;
  }

  // Add generating class to body
  document.body.classList.add("generating");

  // Show toast if not already showing
  if (document.querySelectorAll(".toast.loading").length === 0) {
    const opName = isRenderOp ? "Render" : "Generation";
    showToast(
      "loading",
      `${opName} in progress...`,
      status.message || "Please wait..."
    );
  }

  // Update the loading toast message
  const loadingToast = document.querySelector(".toast.loading .toast-message");
  if (loadingToast && status.message) {
    loadingToast.textContent = status.message;
  }
}

function resetGenerateButton() {
  const generateBtn = document.getElementById("generateBtn");
  generateBtn.classList.remove("loading");
  generateBtn.innerHTML = "Generate";
  isGenerating = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

function resetRenderButton() {
  const renderBtn = document.getElementById("renderBtn");
  renderBtn.classList.remove("loading");
  renderBtn.innerHTML = "Render";
  isRendering = false;

  // Clear locked state
  clearLockedQuadrants();
  removeLockedStyles();

  // Update selection status to show normal message
  updateSelectionStatus();
}

// Check status on page load
(async function initializeStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    // Sync queue state from server
    if (status.queue && status.queue.length > 0) {
      const serverQueued = status.queue.flatMap((item) => item.quadrants);
      setQueuedQuadrants(serverQueued);
    } else {
      clearQueuedQuadrants();
    }

    if (status.is_generating) {
      const opName = status.status === "rendering" ? "Render" : "Generation";
      console.log(`${opName} in progress, restoring UI state...`);
      // Store locked quadrants from server if we don't have them locally
      if (status.quadrants && status.quadrants.length > 0) {
        const localLocked = getLockedQuadrants();
        if (localLocked.length === 0) {
          setLockedQuadrants(status.quadrants);
        }
      }
      setProcessingUI(status);
      applyLockedStyles();
      startStatusPolling();
    } else if (status.queue_length > 0) {
      // Not currently generating but have items in queue
      console.log(`${status.queue_length} items in queue, starting polling...`);
      applyLockedStyles();
      startStatusPolling();
    } else {
      // Not generating/rendering and no queue - clear any stale locked state
      clearLockedQuadrants();
    }

    updateSelectionStatus();
  } catch (error) {
    console.error("Initial status check failed:", error);
  }
})();

```

Now let me add the API endpoint in `app.py`:

```python
"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import hashlib
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
}

# Generation queue for multiple requests
generation_queue: list[
  dict
] = []  # List of {"quadrants": [...], "type": "generate"|"render"}
queue_lock = threading.Lock()
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Ensure web server is running before generation
  ensure_web_server_running()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
  )


def process_queue_item(item: dict) -> dict:
  """Process a single queue item (generate or render)."""
  global generation_state

  quadrants = item["quadrants"]
  item_type = item["type"]

  # Convert to list of tuples
  selected_quadrants = [(q[0], q[1]) for q in quadrants]

  # Initialize generation state
  generation_state["is_generating"] = True
  generation_state["quadrants"] = selected_quadrants
  generation_state["status"] = "starting" if item_type == "generate" else "rendering"
  generation_state["message"] = f"Starting {item_type}..."
  generation_state["error"] = None
  generation_state["started_at"] = time.time()

  print(f"\n{'=' * 60}")
  print(
    f"{'🎯' if item_type == 'generate' else '🎨'} {item_type.title()} request from queue: {selected_quadrants}"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    if item_type == "generate":
      result = run_generation(conn, config, selected_quadrants)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
      return result
    else:  # render
      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def queue_worker():
  """Background worker that processes the generation queue."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    item = None

    # Get next item from queue
    with queue_lock:
      if generation_queue:
        item = generation_queue.pop(0)

    if item is None:
      # No items in queue, wait a bit and check again
      time.sleep(0.5)
      continue

    # Acquire the generation lock and process the item
    with generation_lock:
      try:
        process_queue_item(item)
      finally:
        generation_state["is_generating"] = False

    # Small delay between items
    time.sleep(0.5)

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue(quadrants: list[tuple[int, int]], item_type: str) -> dict:
  """Add a generation/render request to the queue."""
  with queue_lock:
    generation_queue.append(
      {
        "quadrants": quadrants,
        "type": item_type,
      }
    )
    queue_position = len(generation_queue)

  # Ensure the queue worker is running
  start_queue_worker()

  return {
    "success": True,
    "queued": True,
    "position": queue_position,
    "message": f"Added to queue at position {queue_position}",
  }


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  with queue_lock:
    queue_info = [
      {"quadrants": item["quadrants"], "type": item["type"]}
      for item in generation_queue
    ]
  return jsonify(
    {
      **generation_state,
      "queue": queue_info,
      "queue_length": len(queue_info),
    }
  )


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""
  from PIL import Image
  import io

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is None:
      return jsonify(
        {"success": False, "error": f"No generation found for quadrant ({x}, {y})"}
      ), 400

    # Get dimensions from existing image
    existing_img = png_bytes_to_image(generation_bytes)
    width, height = existing_img.size
    mode = existing_img.mode

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating/rendering - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "render")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state (reuse for rendering)
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "rendering"
    generation_state["message"] = "Starting render..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎨 Render request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)

      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return jsonify(
        {
          "success": True,
          "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
          "quadrants": selected_quadrants,
        }
      ), 200

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """API endpoint to generate tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "generate")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting"
    generation_state["message"] = "Starting generation..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎯 Generation request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)
      result = run_generation(conn, config, selected_quadrants)

      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        return jsonify(result), 200
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        return jsonify(result), 400

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


def ensure_web_server_running() -> None:
  """Ensure the web server for rendering is running."""
  global WEB_SERVER_PROCESS

  if WEB_SERVER_PROCESS is not None:
    # Check if still running
    if WEB_SERVER_PROCESS.poll() is None:
      return  # Still running

  # Start the web server
  print(f"🌐 Starting web server for rendering on port {WEB_SERVER_PORT}...")
  WEB_SERVER_PROCESS = start_web_server(WEB_DIR, WEB_SERVER_PORT)


def main():
  global GENERATION_DIR, WEB_SERVER_PORT

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())

```

The "Water Fill" feature is now implemented. Here's a summary:

## Changes Made

**1. `viewer.html`** - Added "Water Fill" button with grid/plus icon and status bar

**2. `viewer.css`** - Added styles for:
   - Water fill status bar with water color preview swatch
   - Hover state for selectable tiles

**3. `viewer.js`** - Added water fill tool logic:
   - `toggleWaterFillTool()` - Toggle the tool on/off
   - `cancelWaterFill()` - Cancel and reset
   - `handleWaterFillClick()` - Handle click on a tile to fill it
   - Keyboard shortcut: `F` to toggle
   - Confirmation dialog before filling

**4. `app.py`** - Added `/api/water-fill` endpoint:
   - Takes `x` and `y` coordinates
   - Creates a solid color image matching the original dimensions
   - Fills with `#4A6372` (the water color)
   - Preserves RGBA mode if original had alpha

## How to Use

1. Click **Water Fill** tool (or press `F`)
2. Click any quadrant - a confirmation dialog will appear
3. Confirm to fill the entire quadrant with the water color (`#4A6372`)
4. The tile refreshes immediately to show the result

Restart the app to pick up the changes.

---

**User**

Water fill failed



No generation found for quadrant (14, 1) - if no generation data is in the db, create a png quadrant of just the water color

---

**Cursor**

I'll update the water fill endpoint to create a new water-filled quadrant if none exists:

```python
"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
  S          - Toggle select tool
"""

import argparse
import hashlib
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import (
  render_quadrant,
  run_generation_for_quadrants,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  WEB_DIR,
  get_generation_config,
  start_web_server,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)

# Generation lock - only one generation at a time
generation_lock = threading.Lock()
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
}

# Generation queue for multiple requests
generation_queue: list[
  dict
] = []  # List of {"quadrants": [...], "type": "generate"|"render"}
queue_lock = threading.Lock()
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
WEB_SERVER_PROCESS = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Returns dict with success status and message/error.
  """
  # Ensure web server is running before generation
  ensure_web_server_running()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
  )


def process_queue_item(item: dict) -> dict:
  """Process a single queue item (generate or render)."""
  global generation_state

  quadrants = item["quadrants"]
  item_type = item["type"]

  # Convert to list of tuples
  selected_quadrants = [(q[0], q[1]) for q in quadrants]

  # Initialize generation state
  generation_state["is_generating"] = True
  generation_state["quadrants"] = selected_quadrants
  generation_state["status"] = "starting" if item_type == "generate" else "rendering"
  generation_state["message"] = f"Starting {item_type}..."
  generation_state["error"] = None
  generation_state["started_at"] = time.time()

  print(f"\n{'=' * 60}")
  print(
    f"{'🎯' if item_type == 'generate' else '🎨'} {item_type.title()} request from queue: {selected_quadrants}"
  )
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    if item_type == "generate":
      result = run_generation(conn, config, selected_quadrants)
      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
      return result
    else:  # render
      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def queue_worker():
  """Background worker that processes the generation queue."""
  global generation_state, queue_worker_running

  print("🔄 Queue worker started")

  while queue_worker_running:
    item = None

    # Get next item from queue
    with queue_lock:
      if generation_queue:
        item = generation_queue.pop(0)

    if item is None:
      # No items in queue, wait a bit and check again
      time.sleep(0.5)
      continue

    # Acquire the generation lock and process the item
    with generation_lock:
      try:
        process_queue_item(item)
      finally:
        generation_state["is_generating"] = False

    # Small delay between items
    time.sleep(0.5)

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue(quadrants: list[tuple[int, int]], item_type: str) -> dict:
  """Add a generation/render request to the queue."""
  with queue_lock:
    generation_queue.append(
      {
        "quadrants": quadrants,
        "type": item_type,
      }
    )
    queue_position = len(generation_queue)

  # Ensure the queue worker is running
  start_queue_worker()

  return {
    "success": True,
    "queued": True,
    "position": queue_position,
    "message": f"Added to queue at position {queue_position}",
  }


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  with queue_lock:
    queue_info = [
      {"quadrants": item["quadrants"], "type": item["type"]}
      for item in generation_queue
    ]
  return jsonify(
    {
      **generation_state,
      "queue": queue_info,
      "queue_length": len(queue_info),
    }
  )


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""
  from PIL import Image
  import io

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating/rendering - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "render")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state (reuse for rendering)
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "rendering"
    generation_state["message"] = "Starting render..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎨 Render request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)

      # Ensure web server is running
      update_generation_state("rendering", "Starting web server...")
      ensure_web_server_running()

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant(conn, config, qx, qy, WEB_SERVER_PORT)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      update_generation_state("complete", f"Rendered {rendered_count} quadrant(s)")
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")

      return jsonify(
        {
          "success": True,
          "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
          "quadrants": selected_quadrants,
        }
      ), 200

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """API endpoint to generate tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Check if already generating - if so, add to queue
  if not generation_lock.acquire(blocking=False):
    # Add to queue instead of rejecting
    result = add_to_queue(selected_quadrants, "generate")
    return jsonify(result), 202  # 202 Accepted

  try:
    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = "starting"
    generation_state["message"] = "Starting generation..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎯 Generation request: {selected_quadrants}")
    print(f"{'=' * 60}")

    # Connect to database
    conn = get_db_connection()
    try:
      config = get_generation_config(conn)
      result = run_generation(conn, config, selected_quadrants)

      if result["success"]:
        print(f"✅ Generation complete: {result['message']}")
        generation_state["status"] = "complete"
        generation_state["message"] = result["message"]
        return jsonify(result), 200
      else:
        print(f"❌ Generation failed: {result['error']}")
        generation_state["status"] = "error"
        generation_state["error"] = result["error"]
        return jsonify(result), 400

    except Exception as e:
      traceback.print_exc()
      generation_state["status"] = "error"
      generation_state["error"] = str(e)
      return jsonify(
        {
          "success": False,
          "error": str(e),
        }
      ), 500
    finally:
      conn.close()

  finally:
    generation_state["is_generating"] = False
    generation_lock.release()


def ensure_web_server_running() -> None:
  """Ensure the web server for rendering is running."""
  global WEB_SERVER_PROCESS

  if WEB_SERVER_PROCESS is not None:
    # Check if still running
    if WEB_SERVER_PROCESS.poll() is None:
      return  # Still running

  # Start the web server
  print(f"🌐 Starting web server for rendering on port {WEB_SERVER_PORT}...")
  WEB_SERVER_PROCESS = start_web_server(WEB_DIR, WEB_SERVER_PORT)


def main():
  global GENERATION_DIR, WEB_SERVER_PORT

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web server on exit
    if WEB_SERVER_PROCESS is not None:
      print("🛑 Stopping web server...")
      WEB_SERVER_PROCESS.terminate()
      WEB_SERVER_PROCESS.wait()

  return 0


if __name__ == "__main__":
  exit(main())

```

Fixed! Now if no generation exists for a quadrant, it will create a new water-filled image using the quadrant dimensions from the config (half the tile size). Restart the app to pick up the change.

