# Snow Mode Demo

A side-by-side comparison viewer for snow/blizzard versions of Isometric NYC tiles.

## Quick Start

### 1. Generate Snow Data

First, generate some snow tiles using the generation script:

```bash
uv run python src/isometric_nyc/generation/generate_layer_snow.py \
  generations/nyc 0 0 \
  --output-dir synthetic_data/datasets/snow
```

This will create:
- `synthetic_data/datasets/snow/inputs/<x>_<y>.png` - Daytime tiles
- `synthetic_data/datasets/snow/generations/<x>_<y>.png` - Nighttime tiles

### 2. Start the Backend Server

```bash
uv run python src/layers/snow/app/server.py \
  --dataset_dir synthetic_data/datasets/snow
```

The server runs on http://localhost:5002 by default.

The dataset directory should contain:
- `inputs/` - daytime tile images (`<x>_<y>.png`)
- `generations/` - nighttime tile images (`<x>_<y>.png`)

### 3. Start the Frontend (Development)

In a separate terminal:

```bash
cd src/layers/snow/app
bun install
bun run dev
```

The frontend runs on http://localhost:5173 by default.

### 4. View the Demo

Open http://localhost:5173/?x=0&y=0 in your browser.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/input/<x>/<y>` | Get daytime (input) image |
| `GET /api/generation/<x>/<y>` | Get nighttime (generation) image |
| `GET /api/available-tiles` | List available tile coordinates |
| `GET /api/status` | Server status and configuration |

## Building for Production

```bash
cd src/layers/snow/app
bun run build
```

The built files will be in `dist/`. The Flask server will automatically serve these when available.
