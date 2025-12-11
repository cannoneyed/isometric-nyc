# Web App

I want to start building a web application for the isometric pixel art NYC map -
let's build it out in an app dir in the root directory. We'll need to build it
with the following stack:

- Bun
- Typescript
- Vite (for building/serving)
- React (for frontend organization / UI)
- Deck.gl (for tiled web viewer)
- Static asset server for tiles
- ~10k 512x512 image tiles
- Separate "layers" for the map itsel and for rendering effects via shaders
  (e.g. a normal map, distance field map)

We have ~100x100 512x512 pixel "tiles" that we'll need to serve at multiple zoom
levels. These are currently stored in a sqlite db but we can export them as
static png images to serve via some static serving infra.

Eventually we'll want to deploy this to the world, but for now we can focus on
local development and getting the UX right.

---

Here are the responses from a Gemini Agent that was tasked with helping to plan
this out:

This is a fantastic project. Rendering a 2.6 gigapixel (50k × 50k) image is well
beyond the limits of a single GPU texture (typically capped at 16k × 16k), so
your "tiled" approach is mandatory.

The complexity here lies in the intersection of **Tiling Logic** (loading only
what is visible), **Coordinate Systems** (isometric pixel art requires a flat
Cartesian plane, not the Mercator projection used by Google Maps), and **Custom
Shaders** (requiring multi-texture sampling per tile).

Here is a breakdown of the best architectural approach for serving and rendering
this map.

### Part A: Serving the Images (The Asset Pipeline)

You should not serve these images dynamically. You want to pre-process them into
a "pyramid" of static tiles.

**The Tool:** **libvips** This is the industry standard for high-performance
image processing. It is fast and memory-efficient.

**The Strategy:**

1.  **Generate XYZ Tiles:** You need to slice your massive master image into
    standard XYZ tiles (Z/X/Y.png).
2.  **Dual Pipelines:** Run this process twice: once for your **Color Map** and
    once for your **Normal Map**. Ensure the tiling grid aligns perfectly (same
    tile size, same zoom levels).

**Command Example:**

```bash
# Generate standard Google-style tiles layout
vips dzsave massive_nyc_color.png ./tiles_color --layout google --suffix .png --tile-size 512
vips dzsave massive_nyc_normal.png ./tiles_normal --layout google --suffix .png --tile-size 512
```

- `--layout google`: Produces a standard `z/x/y.png` directory structure.
- `--tile-size 512`: Matches your requirements.

**Serving:** Upload the resulting folders to an S3 bucket or serve them via
Nginx/Node. No active server logic is required; they are just static files.

---

### Part B: Rendering on the Client

Most mapping libraries (Leaflet, Mapbox) enforce a **Web Mercator** projection,
which distorts images as you move "north." For isometric pixel art, this is
fatal—your straight lines will curve. You need a library that supports
**Orthographic (flat)** projection and allows deep access to the WebGL pipeline
for your shaders.

#### Top Recommendation: Deck.gl (with OrthographicView)

Deck.gl is likely your best option. It is optimized for WebGL, handles the
complex math of "which tiles are currently on screen," and allows you to inject
custom shader code easily.

- **Why it fits:**
  - **OrthographicView:** It supports non-geospatial, infinite 2D coordinate
    systems natively.
  - **TileLayer:** It has a built-in layer that handles loading/unloading tiles
    based on zoom level.
  - **Shader Injection:** It allows you to hook into the fragment shader to add
    your normal map logic.

**Implementation Sketch:**

You would use the `TileLayer` to manage the grid. The "secret sauce" is loading
_two_ textures for every single tile request.

```javascript
import { Deck } from "@deck.gl/core";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";
import { OrthographicView } from "@deck.gl/core";

// Custom shader to apply lighting using the normal map
const lightingFragmentShader = `
#define SHADER_NAME lighting-bitmap-layer-fragment-shader

uniform sampler2D bitmapTexture;
uniform sampler2D normalTexture; // Your secondary texture
uniform float lightingIntensity;

varying vec2 vTexCoord;

void main(void) {
  vec4 color = texture2D(bitmapTexture, vTexCoord);
  vec4 normal = texture2D(normalTexture, vTexCoord);

  // Unpack normal from [0,1] to [-1,1]
  vec3 n = normalize(normal.rgb * 2.0 - 1.0);
  
  // Calculate simple lighting (dot product with light direction)
  vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
  float light = dot(n, lightDir);
  
  gl_FragColor = vec4(color.rgb * light * lightingIntensity, color.a);
}
`;

new Deck({
  views: [new OrthographicView({ id: "ortho" })],
  initialViewState: { target: [25000, 25000, 0], zoom: 0 },
  layers: [
    new TileLayer({
      id: "pixel-art-layer",
      tileSize: 512,

      // 1. Fetch BOTH textures for this tile
      getTileData: ({ x, y, z }) => {
        return Promise.all([
          loadTexture(`https://myserver/tiles_color/${z}/${x}/${y}.png`),
          loadTexture(`https://myserver/tiles_normal/${z}/${x}/${y}.png`),
        ]);
      },

      // 2. Render a custom BitmapLayer for each tile
      renderSubLayers: (props) => {
        const {
          bbox: { left, bottom, right, top },
        } = props.tile;
        const [colorTex, normalTex] = props.data; // Data from getTileData

        return new BitmapLayer(props, {
          data: null, // No data needed, we are rendering a single quad
          image: colorTex,
          bounds: [left, bottom, right, top],

          // 3. Inject custom shader and uniforms
          shaders: {
            fs: lightingFragmentShader,
          },
          uniforms: {
            normalTexture: normalTex,
            lightingIntensity: 1.2,
          },
          updateTriggers: {
            uniforms: [
              /* Add dynamic variables here if light moves */
            ],
          },
        });
      },
    }),
  ],
});
```

#### Runner Up: Pixi.js (v8)

If you prefer a "Game Engine" feel over a "Map" feel, Pixi.js is excellent. It
is strictly 2D and very fast.

- **Pros:** Native handling of sprite batches, very easy to implement "pixel
  perfect" scaling.
- **Cons:** It does not have a built-in "Deep Zoom" quadtree loader out of the
  box. You would likely need to use a community plugin like `pixi-tilemap` or
  write a simple utility to calculate `(viewport / tileSize)` to decide which
  tiles to add to the stage.
- **Shader:** Pixi.js has a very clean `Filter` API where you can pass the
  normal map sprite as a uniform.

#### What to avoid

- **Leaflet / OpenLayers:** While capable, applying a custom fragment shader
  that samples _two_ distinct tile sources (color + normal) simultaneously
  requires fighting their abstraction layers significantly. They are designed
  for single-layer blending, not multi-texture lighting calculations.
- **Three.js:** Great for the shader, but you will have to write the entire
  "Quadtree" logic (calculating which XYZ tiles to load based on frustum) from
  scratch. That is mathematically non-trivial to get bug-free.

### Summary Recommendation

1.  **Use `libvips`** to generate static `z/x/y` tile pyramids for both Color
    and Normal maps.
2.  **Use `Deck.gl`** with `OrthographicView` and `TileLayer`.
3.  **Override `getTileData`** to fetch both images in parallel.
4.  **Override `renderSubLayers`** to pass both textures into a `BitmapLayer`
    with your custom shader.

---

### 1\. Do I need high-res and low-res versions?

**Yes, absolutely.** You must generate the full "pyramid" of tiles.

If you only generate the high-res (Zoom 10) tiles and try to view the whole map
(Zoom 0), deck.gl would attempt to download all 10,000 high-res tiles
simultaneously to fill your screen. This would crash the browser instantly.

You need a structure where **Zoom 0** is a single 512x512 image of the whole
city, **Zoom 1** is 4 images, etc.

**Fortunately, `vips dzsave` does this for you automatically.** When you run the
command I recommended previously:

```bash
vips dzsave massive_nyc.png ./tiles --layout google
```

It will automatically create folders `0/`, `1/`, `2/` ... all the way down to
your maximum resolution. You don't need to do this manually.

---

### 2\. Will deck.gl handle "crossfading"?

**Out of the box: No.** **With configuration: It handles "refinement," not
cinematic crossfading.**

By default, deck.gl behaves like this:

1.  You zoom in.
2.  The old low-res tile disappears.
3.  The screen is empty (background color) for 200ms while the new high-res
    tiles download.
4.  The new tiles "pop" in.

**The Fix: `refinementStrategy: 'best-available'`** You can tell deck.gl to keep
the low-res parent tile on screen (stretched out and blurry) _until_ the new
high-res children have finished loading. Once they are ready, it swaps them
instantly.

There is no built-in "opacity dissolve" animation (where the crisp image slowly
fades in over the blurry one). It is a hard swap.

**How to enable the smooth swap:**

```javascript
new TileLayer({
  // ... other props

  // ESSENTIAL: Prevents "flashing" white background while loading
  refinementStrategy: "best-available",

  // Optional: Pre-load tiles just outside the viewport for smoother panning
  maxCacheSize: 200,

  // ...
});
```

#### If you _absolutely must_ have a cinematic cross-fade:

You would need to write a custom `renderSubLayers`. You would have to:

1.  Load the parent tile _and_ the child tile simultaneously.
2.  Pass a `opacity` uniform to your shader.
3.  Animate that uniform from 0.0 to 1.0 based on a timer.

**Recommendation:** Stick to `refinementStrategy: 'best-available'`. It is what
Google Maps and standard tools use. The "snap" from blurry to sharp is usually
satisfying enough for users and saves you significant engineering headache.

---

# Final instructions

We'll want to build this app out in the `src` dir, under `app` subdir. For
testing, let's assume a 20x20 grid of tiles, indexed starting at 0,0
