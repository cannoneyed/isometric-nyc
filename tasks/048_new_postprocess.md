# New Postprocess

I want to update the postprocessing script in src/isometric_nyc/generation/export_dzi.py to use the logic in https://jenissimo.itch.io/unfaker - it has settings to generate a "fixed grid" pixel art from a raw image. First, can you figure out how to add this functionality to the script? These are the parameters I'm using:


Auto-detect Colors
Max Colors: 128
Snap to Grid

Downscaling Method
dominant (picks the most frequent color in each block)

Advanced Settings
Grid Detection
Algorithm: Simple Grid

Downscaling: Threshold
Dominant Threshold: 0.15

Pixel Size
2

Image Cleanup
Morphological Cleanup
Jaggy Cleanup

Enable Alpha Binarization
Alpha Threshold: 128


Can you extract the logic from this itch.io page and implement it in a `bun` (or `python` if possible) script in the same directory as the export script? The only big difference is that I want to be able to precompute a fixed color palette for the entire tiled city (by sampling from a set of ~100 quadrants).

First, explain the plan below:

## Plan

### Overview

I'll implement the pixel art postprocessing logic from [unfake.js](https://github.com/jenissimo/unfake.js) as a Python script. There's already an official Python port called [unfake.py](https://github.com/painebenjamin/unfake.py) (`pip install unfake`) that provides the exact algorithms you need. We can either use this library directly or implement a streamlined version that integrates with your existing `export_dzi.py` pipeline.

### Approach: Hybrid Integration

Given your requirement for a precomputed fixed color palette across all tiles, I recommend a **two-phase approach**:

1. **Phase 1: Palette Extraction** - Sample ~100 quadrants, apply unfake-style processing to extract dominant colors, then build a unified palette
2. **Phase 2: Tile Processing** - Process each tile using the precomputed palette and unfake algorithms

### Algorithm Breakdown

Based on your parameters, here's what each setting does:

| Parameter | Algorithm |
|-----------|-----------|
| **Simple Grid Detection** | Analyzes horizontal/vertical color runs to find the GCD of consistent pixel sizes |
| **Dominant Downscaling (threshold=0.15)** | For each 2×2 block, pick the most frequent color if it appears in ≥15% of pixels; otherwise blend |
| **Morphological Cleanup** | OpenCV erosion/dilation to remove single-pixel noise |
| **Jaggy Cleanup** | Detects and removes isolated diagonal pixels that break clean edges |
| **Alpha Binarization (threshold=128)** | Converts alpha channel to binary: `alpha = 255 if alpha >= 128 else 0` |
| **Max 128 Colors** | Uses median-cut or k-means quantization with the fixed palette |

### Implementation Plan

#### Step 1: Create `src/isometric_nyc/generation/pixel_art_postprocess.py`

A new Python module containing:

```python
# Core functions to implement:

def detect_pixel_grid(image: Image, method: str = "simple") -> int:
    """Detect the native pixel size using run-length analysis."""

def dominant_downscale(image: Image, pixel_size: int, threshold: float = 0.15) -> Image:
    """Downscale using dominant color selection per block."""

def morphological_cleanup(image: Image) -> Image:
    """Remove single-pixel noise using erosion/dilation."""

def jaggy_cleanup(image: Image) -> Image:
    """Remove isolated diagonal pixels for cleaner edges."""

def binarize_alpha(image: Image, threshold: int = 128) -> Image:
    """Convert alpha to binary (fully opaque or fully transparent)."""

def quantize_to_palette(image: Image, palette: list[tuple[int,int,int]]) -> Image:
    """Map each pixel to the nearest color in the fixed palette."""
```

#### Step 2: Create `src/isometric_nyc/generation/build_city_palette.py`

A script to precompute the city-wide palette:

```python
def sample_quadrants_for_palette(
    db_path: Path,
    sample_size: int = 100,
    pixels_per_quadrant: int = 10000
) -> list[tuple[int, int, int]]:
    """Sample colors from quadrants across the city."""

def build_palette(
    colors: list[tuple[int, int, int]],
    max_colors: int = 128
) -> list[tuple[int, int, int]]:
    """Build unified palette using median-cut quantization."""

def save_palette(palette: list, output_path: Path) -> None:
    """Save palette as JSON and as a visual PNG swatch."""
```

#### Step 3: Update `export_dzi.py`

Modify the existing `postprocess_image()` function to use the new unfake-style processing:

```python
def postprocess_image_unfake(
    img: Image.Image,
    palette: list[tuple[int, int, int]],
    pixel_size: int = 2,
    dominant_threshold: float = 0.15,
    morph_cleanup: bool = True,
    jaggy_cleanup: bool = True,
    alpha_threshold: int = 128,
) -> Image.Image:
    """Apply unfake-style pixel art postprocessing."""
```

### Dependencies

The implementation will use:
- **OpenCV** (`cv2`) - for morphological operations (already available via pyvips workflow)
- **NumPy** - for efficient array operations
- **Pillow** - for image I/O (already used)
- **scipy** or **sklearn** - for k-means/median-cut palette extraction (optional, can implement manually)

Alternatively, we could add `unfake` as a dependency (`uv add unfake`) and use its algorithms directly, though this requires a Rust toolchain for full performance.

### New CLI Flags for `export_dzi.py`

```bash
# New postprocessing flags:
--unfake              # Enable unfake-style processing (replaces --postprocess)
--pixel-size 2        # Native pixel size (default: 2)
--dominant-threshold 0.15
--morph-cleanup       # Enable morphological cleanup
--jaggy-cleanup       # Enable jaggy cleanup
--alpha-threshold 128
--palette-file PATH   # Use precomputed palette JSON
```

### Workflow

1. **Build palette once:**
   ```bash
   uv run python src/isometric_nyc/generation/build_city_palette.py \
       generations/v01 \
       --sample-quadrants 100 \
       --max-colors 128 \
       --output palettes/city_v01.json
   ```

2. **Export with unfake processing:**
   ```bash
   uv run python src/isometric_nyc/generation/export_dzi.py \
       generations/v01 \
       --unfake \
       --palette-file palettes/city_v01.json \
       --pixel-size 2 \
       --morph-cleanup \
       --jaggy-cleanup
   ```

### File Structure

```
src/isometric_nyc/generation/
├── export_dzi.py                 # (modified)
├── pixel_art_postprocess.py      # NEW - core algorithms
├── build_city_palette.py         # NEW - palette extraction
└── palettes/                     # NEW - saved palette files
    └── .gitkeep
```

### Decisions

1. **Use `unfake` library directly** - Need the speed for processing many tiles
2. **PNG swatch for palette** - Keep existing logic from `build_unified_palette()`
3. **Fixed pixel size as parameter** - Default to 2, applied uniformly to all tiles

---

## Implementation Complete ✅

### Files Modified/Created

1. **`src/isometric_nyc/generation/pixel_art_postprocess.py`** (NEW)
   - `process_tile_unfake()` - Wraps unfake library with our parameters
   - `extract_palette_from_image()` - Extracts RGB colors from palette image
   - `postprocess_image_unfake()` - High-level interface matching existing API

2. **`src/isometric_nyc/generation/export_dzi.py`** (MODIFIED)
   - Added `--unfake` flag to enable pixel art postprocessing
   - Added `--pixel-size`, `--unfake-colors`, `--dominant-threshold` options
   - Added `--morph-cleanup`, `--jaggy-cleanup`, `--alpha-threshold` options
   - Integrated unfake processing into tile assembly pipeline

3. **`pyproject.toml`** - Added `unfake` dependency

### Usage

```bash
# Basic unfake export to default directory (public/dzi/)
uv run python src/isometric_nyc/generation/export_dzi.py \
    generations/v01 --unfake

# Export to custom directory (public/unfake_v1/)
uv run python src/isometric_nyc/generation/export_dzi.py \
    generations/v01 --unfake --export-dir unfake_v1

# With custom settings
uv run python src/isometric_nyc/generation/export_dzi.py \
    generations/v01 \
    --unfake \
    --export-dir unfake_custom \
    --pixel-size 2 \
    --unfake-colors 128 \
    --dominant-threshold 0.15 \
    --morph-cleanup \
    --jaggy-cleanup

# With existing palette file
uv run python src/isometric_nyc/generation/export_dzi.py \
    generations/v01 \
    --unfake \
    --palette palettes/city_v01.png

# Test mode (20x20 subset, exports to dzi_test/)
uv run python src/isometric_nyc/generation/export_dzi.py \
    generations/v01 --unfake --test
```

### Export Directory Structure

All exports go to `src/app/public/{export_dir}/`:
```
public/{export_dir}/
├── tiles.dzi          # DZI descriptor
├── tiles_files/       # Tile pyramid
└── metadata.json      # Metadata sidecar
```

### Viewing Different Exports

The web app supports loading different exports via URL parameter:

```bash
# Start dev server
cd src/app && npm run dev

# View different exports:
# - Default (dzi): http://localhost:3000/
# - Custom export:  http://localhost:3000/?export=unfake_v1
# - Test export:    http://localhost:3000/?export=dzi_test
```

### Processing Pipeline

1. **Alpha Binarization** - Convert alpha to binary (≥128 = opaque, <128 = transparent)
2. **Grid Snapping** - Crop image to align with pixel grid
3. **Color Quantization** - Map to fixed palette using median-cut
4. **Dominant Downscale** - Pick most frequent color per 2×2 block (threshold=0.15)
5. **Jaggy Cleanup** - Remove isolated diagonal pixels
6. **Morphological Cleanup** - Remove single-pixel noise
7. **Upscale** - Resize back to original 512×512 using nearest neighbor
