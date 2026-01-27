# Postprocessing

Let's take the functionality of src/isometric*nyc/pixelate.py and apply it to
the entire exported tile set in src/app/public/tiles. The only big change we
need to make is to ensure we use the same reduced color set for \_all* tiles

## Implementation Complete ✅

Created `src/isometric_nyc/postprocess_tiles.py` which:

1. Samples colors from a representative subset of tiles
2. Builds a unified 32-color palette
3. Applies pixelation (2x scale) and quantization to all tiles using the shared
   palette
4. Processes tiles in parallel for efficiency (8 workers)

### Output

- **Processed tiles**: `src/app/public/tiles_processed/` (556MB vs 2.5GB
  original)
- **Unified palette**: `src/app/public/palette.png`

### Usage

```bash
# Full processing (build palette + process all tiles)
uv run python -m isometric_nyc.postprocess_tiles

# Build palette only
uv run python -m isometric_nyc.postprocess_tiles --build-palette-only

# Use existing palette
uv run python -m isometric_nyc.postprocess_tiles --palette path/to/palette.png

# Customize settings
uv run python -m isometric_nyc.postprocess_tiles --scale 2 --colors 32 --workers 8

# Process in place (overwrites originals!)
uv run python -m isometric_nyc.postprocess_tiles --in-place
```

### To use processed tiles in the app

Replace `tiles/` with `tiles_processed/` or update the app's tile URL pattern
