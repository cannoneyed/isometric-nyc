# Water Shader

The next phase of the isometric NYC pixel art map project is implementing a
water shader on the water tiles in the final tile viewer web app at
src/isometric_nyc/app.

Using the idea in 021_water_shader.md, we'll be using a blurred distance mask to
power the shader. We're going to split this work into a few steps:

## 1. Web app shader ✅ COMPLETED

First, we're going to get a working water shader renderer working in the web
app. The shader mask tiles will follow the exact same naming scheme as the
regular tiles (e.g. public/tiles/<zoom*level>/<x>*<y>.png).

For the mask tiles, water will be indicated using white, and land will be
indicated with black. For any non-existent tiles, we need to just assume land
(e.g. black).

We've exported one 2x2 quadrant set of water masks to the `public/water_masks/0`
dir (tl at 1,16).

We also have a working water shader in src/water_shader_demo. Let's get a new
layer on the deck.gl tile viewer app to implement the water shader for tiles!

### Implementation Details

The following files were created/modified:

- `src/app/src/shaders/water.ts` - WebGL shader code (vertex + fragment) and
  utility functions for creating textures and initializing WebGL context
- `src/app/src/components/WaterShaderOverlay.tsx` - React component that renders
  the water shader effect as a transparent WebGL canvas overlay on top of the
  deck.gl tile layer
- `src/app/src/components/IsometricMap.tsx` - Updated to include the
  WaterShaderOverlay component
- `src/app/src/components/ControlPanel.tsx` - Added water shader controls
  (enable/disable, show mask toggle, wave speed, wave frequency, ripple darkness
  sliders)
- `src/app/src/App.tsx` - Added water shader state management

The water shader works as an overlay that:

- Loads mask tiles from `public/water_masks/0/{x}_{y}.png`
- For tiles without masks, assumes land (no water effect)
- Renders animated wave/ripple effects only on water pixels (white in mask)
- Is fully transparent on land pixels (black in mask)

## 2. Generate water masks for all tiles

TODO: Create a script to generate blurred water masks for all tiles based on the
water color detection in the pixel art tiles. The masks should be saved to
`public/water_masks/0/` following the same naming convention as tiles.
