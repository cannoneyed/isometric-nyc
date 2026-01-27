# Isometric NYC Viewer

A high-performance tiled map viewer for exploring isometric pixel art of New
York City.

## Tech Stack

- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool (via Bun)
- **Deck.gl** - WebGL-powered tile rendering with OrthographicView
- **Bun** - Package management and script running

## Getting Started

```bash
# Install dependencies
bun install

# Start development server
bun run dev

# Build for production
bun run build

# Preview production build
bun run preview

# Deploy to github pages
uv run python deploy.py
```

The app will open at http://localhost:3000

## Features

- **Tile-based rendering** - Efficiently loads only visible tiles
- **Smooth pan & zoom** - Hardware-accelerated WebGL rendering
- **Orthographic projection** - Flat 2D view perfect for pixel art (no
  perspective distortion)
- **Placeholder tiles** - Visual grid while actual tiles load
- **Light controls** - UI for future shader-based lighting effects

## Tile Configuration

The viewer is configured for a 20×20 grid of 512×512 pixel tiles:

```
/public/tiles/
  0/              # Zoom level 0 = native resolution (max zoom in)
    0_0.png       # Tile at coordinate (0, 0)
    0_1.png       # Tile at coordinate (0, 1)
    ...
    19_19.png     # Tile at coordinate (19, 19)
  1/              # Zoom level 1 = 2x zoomed out (10×10 grid)
  2/              # Zoom level 2 = 4x zoomed out (5×5 grid)
  info.json       # Tile metadata
```

**Zoom level convention:**

- `z=0` — Native pixel resolution (maximum zoom in), full 20×20 grid
- `z=1` — 2× zoomed out, 10×10 grid
- `z=2` — 4× zoomed out, 5×5 grid
- etc.

To add your own tiles, place PNG images at `public/tiles/0/{x}_{y}.png` where:

- `x` ranges from 0 to 19
- `y` ranges from 0 to 19

## Controls

| Action           | Description   |
| ---------------- | ------------- |
| **Scroll**       | Zoom in/out   |
| **Drag**         | Pan the view  |
| **Double-click** | Zoom to point |

## Architecture

```
src/
├── main.tsx              # Entry point
├── App.tsx               # Main app component
├── components/
│   ├── IsometricMap.tsx  # Deck.gl tile viewer
│   ├── ControlPanel.tsx  # Zoom/light controls
│   └── TileInfo.tsx      # Tile hover info
└── styles/
    └── global.css        # Dark theme styling
```

## Future Enhancements

- [ ] Multi-texture rendering (color + normal maps)
- [ ] Custom shaders for dynamic lighting
