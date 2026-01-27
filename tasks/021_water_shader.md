# Water shader

I'd like to build a water shader for the tile data that we've composed for our
isometric pixel art map of NYC.

We'll use a 4x4 quadrant grid to debug the shader.

Here's what a Gemini agent advised us to do to build the shader:

---

Since you are building a giant isometric map (and likely using a game engine
like Godot, given your interest in it), **yes, you absolutely need a way to
identify the edges.**

You cannot easily mathematically detect "shoreline" inside a single pixel shader
run without data telling the shader where the land sits.

Here is the breakdown of the simplest, most effective way to achieve "crashing
waves" in a pixel-art style, keeping your `#4A6372` palette in mind.

### The Technique: The "Distance Field" Mask

A simple binary mask (White = Water, Black = Land) creates hard edges, but it
doesn't tell the water **which direction** the shore is or how far away it is.
This makes it impossible to animate a wave _traveling toward_ the shore.

Instead, you want a **Gradient Mask (or Signed Distance Field - SDF)**.

- **Deep Water:** Black (0.0)
- **Shoreline:** White (1.0)
- **Land:** Transparent or Masked out.

The gradient tells the shader "how close am I to the shore?" You use this value
to drive the wave animation.

---

### Step-by-Step Implementation

#### 1\. Prepare the Mask

You don't need to hand-paint a complex gradient.

- Create a binary mask of your landmass.
- Apply a **Gaussian Blur** to it in Photoshop/GIMP.
- The blurry edge around the land effectively becomes your distance data. The
  lighter the pixel, the closer it is to land.

#### 2\. The Shader Logic

You will use the mask value as a "time offset."

- If `Mask_Value` is 0.1 (far from shore), the wave arrives later.
- If `Mask_Value` is 0.9 (near shore), the wave arrives sooner.

To keep it "pixel art" (crunchy), you must use **Step** or **Threshold**
functions to prevent smooth HD gradients.

#### 3\. The Code (Godot / GLSL style)

Here is a shader conceptualized for a 2D sprite. This assumes you have your main
water texture and the blurred mask we discussed.

```glsl
shader_type canvas_item;

uniform vec4 water_color : source_color = vec4(0.29, 0.38, 0.44, 1.0); // Your #4A6372
uniform vec4 foam_color : source_color = vec4(0.8, 0.9, 0.9, 1.0);     // White/Blue foam
uniform sampler2D land_mask; // The blurred mask texture
uniform float wave_speed = 2.0;
uniform float wave_freq = 10.0;
uniform float pixel_size = 256.0; // Higher = crunchier pixels

void fragment() {
    // 1. Sample the mask (How close are we to land?)
    float dist_to_land = texture(land_mask, UV).r;

    // 2. Create the Wave Driver
    // We combine Time and Distance. The "- dist_to_land" makes waves move inward.
    float wave_driver = sin((TIME * wave_speed) - (dist_to_land * wave_freq));

    // 3. Pixelate / Quantize
    // This snaps the smooth sine wave into hard pixel bands
    // round() or floor() helps achieve the retro look
    float pixel_wave = round(wave_driver * 2.0) / 2.0;

    // 4. Determine Foam
    // If we are very close to land (high dist) AND the wave is high
    float foam_mix = step(0.8, dist_to_land + (pixel_wave * 0.1));

    // 5. Output
    vec4 final_color = mix(water_color, foam_color, foam_mix);

    // Optional: Cut out the actual land if the mask is purely for water
    if (dist_to_land > 0.95) {
        final_color.a = 0.0;
    }

    COLOR = final_color;
}
```

### Why this works for Pixel Art

1.  **The `step` function:** This is your best friend. It turns smooth gradients
    into hard lines. It ensures you don't get "blurry" generic game water, but
    hard bands of color.
2.  **The Mask:** By using the blurred mask, the waves automatically contour to
    the shape of your isometric NYC islands. If you have a pier jutting out, the
    blur will wrap around it, and the shader waves will naturally curl around
    the pier.

### A Simpler Alternative (Noise-Based)

If creating a mask for a "giant" map is too difficult (e.g., the map is
procedurally generated or too large for a single texture):

You can use **Simplex Noise** to generate the waves, and use a simple **Neighbor
Check** in the shader (if your resolution allows) to detect edges, but this is
computationally heavier and often looks glitchy.

**Recommendation:** Stick to the "Blurred Mask" texture approach. It is
performant, gives you artistic control over where the waves are, and allows for
that specific "crashing against the shore" movement.

---

I'd like to try the "blurred mask" approach.

1. Let's build a simple web app in src/water_shader_demo

- use bun for building/package management
- use vite for serving
- use react for web scaffolding

The app will serve a 4x4 grid of "quadrants" which are 512x512 pngs. We'll
eventually export both the image tiles and the corresponding mask tiles. We'll
want to create a webgl shader to display the waves on top of the water pixels.

For now, let's just scaffold out the basic web app and we can figure out how to
get the assets.
