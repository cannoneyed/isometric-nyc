# Water Shader (🚧 WIP)

A WebGL shader demo for adding animated water effects to the isometric map
tiles.

### Features

- **Shoreline foam**: Subtle animated foam effect at water edges using a
  distance mask
- **Water ripples**: Gentle darkening ripples across open water areas
- **Adjustable parameters**: Real-time sliders for tweaking all effects
- **Pixel art aesthetic**: Quantized effects that maintain the retro look

### Tech Stack

- **React** + **TypeScript** — UI framework
- **Vite** — Build tool
- **WebGL** — Custom GLSL shaders for water animation
- **Bun** — Package management

### Running the Water Shader Demo

```bash
cd src/demos/water_shader

# Install dependencies
bun i

# Start development server
bun dev
```

The app will open at **http://localhost:5173** (or next available port)

### How It Works

The shader uses a **distance mask** approach:

- **Black pixels** in the mask = land (pass through unchanged)
- **White pixels** = deep water (receives ripple effects)
- **Gradient transition** = shoreline (receives foam effects)

The mask is a blurred version of the land/water boundary, providing smooth
distance information for wave animation.

### Shader Controls

| Control         | Description                        |
| --------------- | ---------------------------------- |
| Wave Speed      | Animation speed of shoreline waves |
| Wave Frequency  | Density of wave bands              |
| Foam Threshold  | How far from shore foam appears    |
| Water Darkness  | Overall brightness of water pixels |
| Ripple Darkness | Intensity of deep water ripples    |

### Adding Tile Assets

Place tile images and corresponding masks in:

```
src/water_shader_demo/public/
├── tiles/
│   └── 0_0.png          # The tile image
└── masks/
    └── 0_0.png          # Distance mask (black=land, white=water)
```
