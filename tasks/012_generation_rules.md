# Generation Rules

**Status: ✅ COMPLETE** - Library created at
`src/isometric_nyc/generation/generate_template.py`

## Overview

Let's extend out the building tile generation system that uses a fine-tuned
model to "inpaint" regions of a 1024x1024 pixel image.

Each image tile is broken down into four "quadrants", which each have a unique x
y index - so a tile might be composed of the four quadrants (0,0), (0,1), (1,0),
and (1,1).

The plan is to gradually generate a massive grid of tiles, but the key is that
we want there to be quadrant overlap so that there won't be any "seams" of
pixels that were generated without having any of the adjacent quadrant's pixels
in the "template" image to be infilled. For example:

(Quadrants - G means previously Generated, x means empty, and S means selected /
to be generated)

```
x x x x
G G S x
G G S x
x x x x
```

The above case is good, since the selected quadrants will be generated using a
template image with the left half previously generated and the right half left
blank in order to be painted in by the infilling model. Since the generated
pixels will be extended seamlessly, there won't be any seams.

```
x x x x x x
G G x S S x
G G x S S x
x x x x x x
```

The above case is also good, since the selected quadrants don't have any
neighbors and can be cleanly generated without any seams forming.

```
x x x x x x
G G S G G x
G G S G G x
x x x x x x
```

The above case is also good, since we can create a template image with a
vertical band 50% wide in the middle of the image (corresponding) to the S
quadrants and the left 25% from the left G and the right 25% from the right G.
Since there's generated pixel information from both the left and right, no seams
will be present.

```
G G G G G x
G G S G G x
G G S G G x
x x x x x x
```

The above case is ILLEGAL - this is because we can't generate bot the top and
bottom S quadrants without the top S qudrant forming a "context-less" border
with the G quadrant above. In other words, since a template image can be a
maximum of 2 x 2 quadrants, we can't have seamless infill generation with
content from above, left, and right because we've selected 2 quadrants.

```
G G G G G x
G G S G G x
G G x G G x
x x x x x x
```

However the above case is LEGAL, since we can create a template image with a
square quadrant on the bottom middle - with the top half containing the
generated pixels above the selected quadrant and the left/right halves both 25%
of the generated pixels to the left/right.

---

First, I want to create a library that formalizes these rules and creates a
template image for generation - let's create a new script / library in
src/isometric_nyc/generation called `generate_template.py` that extends the
logic in `generate_tile_omni.py`. It needs to create the template image with the
selected quadrants filled with the corresponding "render" pixels and the
remaining pixels from the "generated" pixels, and the rendered part outlined in
a 2px solid red border (that goes on top, no shifting of the pixels)

Second, we need the script to be able to extract the selected quadrant generated
pixel data from a generated image.

Please use a nice modular format, since the next step will be to make the
"infill" shape generic instead of exactly a quadrant.

We don't want to do any generation yet, just be able to test that we can
generate the correct infill templates and parse the corresponding generations.

---

## Implementation Summary

Created `src/isometric_nyc/generation/generate_template.py` with:

### Core Classes

- **`QuadrantState`**: Enum for quadrant states (EMPTY, GENERATED, SELECTED)
- **`QuadrantPosition`**: Immutable position class with neighbor utilities
- **`BoundingBox`**: Pixel coordinate bounding box
- **`QuadrantGrid`**: Manages grid state and validates selection legality

### Key Functions

- **`create_template_image()`**: Creates infill template with:

  - Selected quadrants → render pixels
  - Generated neighbors → generation pixels
  - 2px red border around render region

- **`extract_generated_quadrants()`**: Extracts quadrant images from generated
  result

- **`draw_red_border()`**: Draws border on top of image (no pixel displacement)

### Validation Rules Implemented

The library correctly validates:

- 1x1 selection: Can have up to 3 generated neighbors
- 1x2 (tall) selection: Cannot have generated neighbors on BOTH left AND right
- 2x1 (wide) selection: Cannot have generated neighbors on BOTH top AND bottom
- 2x2 selection: Cannot have ANY generated neighbors (fills entire template)

### Testing

Run tests with:

```bash
uv run python src/isometric_nyc/generation/generate_template.py
uv run python src/isometric_nyc/generation/generate_template.py --output-dir /tmp/tests
```

### Convenience Functions

- `create_half_template()`: For half-tile generation
- `create_single_quadrant_template()`: For single quadrant with neighbors
- `create_test_grid_state()`: Creates common test scenarios
