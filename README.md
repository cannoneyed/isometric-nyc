# Isometric NYC

This project aims to generate a massive isometric pixel art view of New York
City using the latest and greatest AI tools available.

## Project Structure

```
isometric-nyc/
├── src/
│   ├── isometric_nyc/    # Python tile generation pipeline
│   ├── app/              # Web viewer (React + Deck.gl)
│   └── web/              # 3D tiles viewer (Three.js)
├── synthetic_data/       # Training datasets
├── tile_plans/           # Tile generation plans
└── tasks/                # Task documentation
```

---

## Web Viewer

A high-performance tiled map viewer for exploring the isometric pixel art map.

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **Deck.gl** — WebGL tile rendering with OrthographicView
- **Bun** — Package management

### Running the Web App

```bash
cd src/app

# Install dependencies
bun install

# Start development server
bun run dev
```

The app will open at **http://localhost:3000**

### Adding Tiles

Place your tile images in the `src/app/public/tiles/` directory:

```
src/app/public/tiles/
├── 0/                    # Zoom level 0 = native resolution (max zoom in)
│   ├── 0_0.png          # Tile at x=0, y=0
│   ├── 0_1.png          # Tile at x=0, y=1
│   ├── 1_0.png          # Tile at x=1, y=0
│   └── ...
├── 1/                    # Zoom level 1 = 2× zoomed out
├── 2/                    # Zoom level 2 = 4× zoomed out
└── info.json
```

**Tile naming:** `{zoom_level}/{x}_{y}.png`

**Zoom convention:** | Level | Resolution | Grid Size (20×20 base) |
|-------|------------|------------------------| | 0 | Native (512×512 per tile)
| 20×20 tiles | | 1 | 2× zoomed out | 10×10 tiles | | 2 | 4× zoomed out | 5×5
tiles |

---

## Python Pipeline

### Setup

Create a `.env` file in the root directory with API credentials:

```bash
GOOGLE_MAPS_API_KEY=...
NYC_OPENDATA_APP_TOKEN=...
```

Install dependencies using `uv`:

```bash
uv sync
```

### Scripts

```bash
# Generate tiles for an address
uv run isometric-nyc "350 5th Ave, New York, NY"

# Run the e2e generation viewer
uv run python src/isometric_nyc/e2e_generation/app.py

# Export tiles to the web app
**uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19**
```

### Exporting Tiles for the Web App

The `export_tiles_for_app.py` script exports quadrants from the SQLite database
to the web app's tile directory:

```bash
# Export a 20x20 region, normalizing coordinates so (0,0) is top-left
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19

# Export a specific region (e.g., quadrants 10-29 → tiles 0-19)
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 10,10 --br 29,29

# Export render images instead of generations
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --render

# Custom output directory
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --output-dir ./my-tiles/0

# Dry run to see what would be exported
uv run python src/isometric_nyc/e2e_generation/export_tiles_for_app.py \
  generations/v01 --tl 0,0 --br 19,19 --dry-run
```

**Coordinate normalization:** The `--tl` coordinate becomes `0_0.png` in the
output. For example, exporting `--tl 10,15 --br 20,25` produces tiles `0_0.png`
through `10_10.png`.

---

## Development

| Task               | Command                       |
| ------------------ | ----------------------------- |
| Run Python tests   | `uv run pytest`               |
| Format Python code | `uv run ruff format .`        |
| Lint Python code   | `uv run ruff check .`         |
| Run web app        | `cd src/app && bun run dev`   |
| Build web app      | `cd src/app && bun run build` |
