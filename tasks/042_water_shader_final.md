# Water Shader

## Step 1: ✅ COMPLETE

For the water shader, we need to create a synthetic dataset for a black/white water mask (as specified in tasks/021_water_shader.md). We want to follow the same "infill / omni" template approach that we use for src/isometric_nyc/synthetic_data/create_omni_dataset.py - that is, for n "tiles" with a "render" and a "generation", we'll want to create m variants for each of the specified infill types (quadrants, rectangles, etc)

To make this dataset, we'll need a new script called "generate_water_mask_tile.py" in the src/generation - for now, copy the logic of `generate_nano_banana` since we'll be calling a nano banana model with a modified prompt. Ensure that the script takes the following extra params

--output-dir: A directory to save the render to a `renders` subdir and the generation to a `generations`, with the name `<x>_<y>.png`.

### Implementation

Script created at: `src/isometric_nyc/generation/generate_water_mask_tile.py`

**Usage Examples:**

```bash
# Generate water mask for tile at (0,0) using pixel art (default):
uv run python src/isometric_nyc/generation/generate_water_mask_tile.py \
  generations/v01 0 0 --output-dir synthetic_data/datasets/water_masks

# Generate for multiple tiles:
uv run python src/isometric_nyc/generation/generate_water_mask_tile.py \
  generations/v01 --quadrants "(0,0),(2,0),(0,2)" --output-dir synthetic_data/datasets/water_masks

# Use 3D renders as input instead of pixel art (rare):
uv run python src/isometric_nyc/generation/generate_water_mask_tile.py \
  generations/v01 0 0 --output-dir synthetic_data/datasets/water_masks --use-render
```

**Output Structure:**
```
<output-dir>/
  inputs/<x>_<y>.png        # The input image (pixel art by default, or 3D render with --use-render)
  generations/<x>_<y>.png   # The generated binary water mask (white=water, black=land)
  debug/<x>_<y>/            # Debug images (input, prompt, generated)
```


## Step 2 - debug app: ✅ COMPLETE

We need to modify the src/water_shader_demo app to be able to load various x,y tiles with the corresponding generation - we need a python backend that can be booted using two params:

--mask_dir - a directory containing the <x>_<y>.png images
--generations_dir - a directory containing the SQLite generations db

The app needs to load the specified x,y 2x2 quadrant tile (starting at top left) and the accompanying mask image, then compose them into a debug view of the applied shader. We need to be able to input an x/y coordinates in a numerical inputs, and have those coordinates be tracked in the url.

### Implementation

**Backend Server:** `src/water_shader_demo/server.py`

**Usage:**

```bash
# Start the backend server:
uv run python src/water_shader_demo/server.py \
  --mask_dir synthetic_data/datasets/water_masks/generations \
  --generations_dir generations/v01

# Then open: http://localhost:5001/?x=0&y=0
```

**Development Mode (with hot reload):**

```bash
# Terminal 1: Start the backend
uv run python src/water_shader_demo/server.py \
  --mask_dir synthetic_data/datasets/water_masks/generations \
  --generations_dir generations/v01

# Terminal 2: Start Vite dev server (proxies API requests to backend)
cd src/water_shader_demo && bun run dev

# Open: http://localhost:5173/?x=0&y=0
```

**Features:**
- X/Y coordinate inputs in the control panel
- URL tracking (coordinates synced to `?x=0&y=0` query params)
- Quick navigation buttons for tiles that have both generation and mask
- Status indicator showing if mask is available
- Loading and error states
- Lists available tiles and masks from the database

**API Endpoints:**
- `GET /api/tile/<x>/<y>` - Get 2x2 tile image from SQLite db
- `GET /api/mask/<x>/<y>` - Get water mask image from disk
- `GET /api/available-tiles` - List all tiles with 4 quadrants
- `GET /api/available-masks` - List all available masks
- `GET /api/status` - Server configuration status

