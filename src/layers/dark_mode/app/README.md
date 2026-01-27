# Dark Mode Demo

A side-by-side comparison viewer for day/night versions of Isometric NYC tiles.

## Quick Start

### 1. Generate Dark Mode Data

First, generate some dark mode tiles using the generation script:

```bash
uv run python src/isometric_nyc/generation/generate_dark_mode.py \
  generations/v01 0 0 \
  --output-dir synthetic_data/datasets/dark_mode
```

This will create:
- `synthetic_data/datasets/dark_mode/inputs/<x>_<y>.png` - Daytime tiles
- `synthetic_data/datasets/dark_mode/generations/<x>_<y>.png` - Nighttime tiles

### 2. Start the Backend Server

```bash
uv run python src/dark_mode_demo/server.py \
  --dataset_dir synthetic_data/datasets/dark_mode
```

The server runs on http://localhost:5002 by default.

The dataset directory should contain:
- `inputs/` - daytime tile images (`<x>_<y>.png`)
- `generations/` - nighttime tile images (`<x>_<y>.png`)

### 3. Start the Frontend (Development)

In a separate terminal:

```bash
cd src/dark_mode_demo
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
cd src/dark_mode_demo
bun run build
```

The built files will be in `dist/`. The Flask server will automatically serve these when available.

