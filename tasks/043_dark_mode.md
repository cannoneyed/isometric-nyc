# Dark Mode

## Step 1 ✅

Let's to create a synthetic dataset for a "dark mode", where we re-render all tiles at night-time (with an optional light mask for a shader). We want to follow the same "infill / omni" template approach that we use for src/isometric_nyc/synthetic_data/create_omni_dataset.py - that is, for n "tiles" with a "render" and a "generation", we'll want to create m variants for each of the specified infill types (quadrants, rectangles, etc)

To make this dataset, we'll need a new script called "generate_dark_mode.py" in the src/generation - for now, copy the logic of `generate_water_mask_tile` since we'll be calling a nano banana model with a modified prompt (which I'll manuall add later).

**Implementation:** `src/isometric_nyc/generation/generate_dark_mode.py`

Usage:
```bash
# Generate dark mode for a single tile
uv run python src/isometric_nyc/generation/generate_dark_mode.py \
  generations/v01 0 0 --output-dir synthetic_data/datasets/dark_mode

# Generate for multiple tiles
uv run python src/isometric_nyc/generation/generate_dark_mode.py \
  generations/v01 --quadrants "(0,0),(2,0)" --output-dir synthetic_data/datasets/dark_mode
```


## Step 2 - debug app ✅

We need to fork the src/water_shader_demo app to be able to load various x,y tiles with the corresponding generation and do a side-by-side comparison with the input. Remove all of the cotnrols on the left and just show the left/right comparison of the day(input) and night (generation) images

**Implementation:** `src/dark_mode_demo/`

Usage:
```bash
# 1. Start the backend server
uv run python src/dark_mode_demo/server.py \
  --input_dir synthetic_data/datasets/dark_mode/inputs \
  --generation_dir synthetic_data/datasets/dark_mode/generations

# 2. Start the frontend (in another terminal)
cd src/dark_mode_demo
bun install
bun run dev

# 3. Open http://localhost:5173/?x=0&y=0
```

## Step 3 - Automatic Generation

