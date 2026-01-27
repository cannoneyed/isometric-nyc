# isometric.nyc web app

A high-performance tiled map viewer for exploring the isometric pixel art map.

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **OpenSeaDragon** — Tile viewer library
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

---

### Tile Formats

#### DZI

[Deep Zoom Image](https://openseadragon.github.io/examples/tilesource-dzi/) is
OpenSeadragon's native format, providing the best performance with zero custom
tile loading code.

```
src/app/public/dzi
├── tiles.dzi              # DZI descriptor (XML)
├── metadata.json          # Custom metadata (grid dimensions, origin)
└── tiles_files/           # Tile pyramid
    ├── 9/                 # Lowest zoom level
    │   └── 0_0.webp
    ├── 10/
    ├── ...
    └── 17/                # Highest zoom level (full resolution)
        ├── 0_0.webp
        ├── 0_1.webp
        └── ...
```

**Benefits of DZI:**

- **Native OpenSeadragon support** - No custom tile loading code required
- **WebP format** - 25-35% smaller than PNG
- **Standard format** - Works with any static file server or CDN
- **Fast** - Browser handles caching and prefetching natively

---

### Exporting Tiles for the Web App

#### DZI Format

[DZI (Deep Zoom Image)](https://openseadragon.github.io/examples/tilesource-dzi/)
is OpenSeadragon's native format, providing the best performance. Export uses
[libvips](https://www.libvips.org/) for efficient pyramid generation.

**Prerequisites:**

```bash
# Install libvips
brew install vips

# Add pyvips dependency (already in pyproject.toml)
uv sync
```

**Export commands:**

```bash
# Export all tiles to DZI format (WebP, with postprocessing)
DYLD_LIBRARY_PATH=/opt/homebrew/lib uv run python \
  src/isometric_nyc/generation/export_dzi.py generations/nyc

# Export with PNG instead of WebP
DYLD_LIBRARY_PATH=/opt/homebrew/lib uv run python \
  src/isometric_nyc/generation/export_dzi.py generations/nyc --png

# Export with bounds clipping (NYC outline)
DYLD_LIBRARY_PATH=/opt/homebrew/lib uv run python \
  src/isometric_nyc/generation/export_dzi.py generations/nyc --bounds v1.json

# Export without postprocessing (raw tiles)
DYLD_LIBRARY_PATH=/opt/homebrew/lib uv run python \
  src/isometric_nyc/generation/export_dzi.py generations/nyc --no-postprocess

# Dry run to see what would be exported
DYLD_LIBRARY_PATH=/opt/homebrew/lib uv run python \
  src/isometric_nyc/generation/export_dzi.py generations/nyc --dry-run
```

**Output files:**

- `src/app/public/tiles.dzi` - DZI descriptor (XML)
- `src/app/public/tiles_metadata.json` - Custom metadata for frontend
- `src/app/public/tiles_files/` - Tile pyramid directory

**Benefits of DZI:**

- **Native OpenSeadragon support** - Zero custom tile loading code
- **WebP format** - 25-35% smaller than PNG
- **Fast generation** - libvips is highly optimized for large images
- **Standard format** - Works with any static file server
