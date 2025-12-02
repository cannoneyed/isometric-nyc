# Tile generation

Next we're going to implement the remaining pieces to do step-by-step tile
generation. The goal is to progressively march tile-by-tile to generate the
entire map of NYC in the isometric pixel art style.

Each generation will be a 1024 x 1024 pixel square and will be guided by assets
generated using the `export_views` python script. These two assets are:

1. An isometric whitebox guide of the building geometry (generated from the
   logic in `whitebox.py`)
2. An isometric rendered view of the 3D building data using Google 3D tiles API
   (generated in a web view)

These two assets will be saved in individual directories, along with a JSON file
that defines the geometry of the generation (`view.json`)

```
/<view-id>
  whitebox.png
  render.png
  view.json
```

The <view-id> parameter will be a hash of the centroid

---

Generation will use a (to be implemented) script called `generate.py` that will
be based off of the marimo notebook in `notebooks/nano-banana.py`.

Generation will be done in one of two ways:

1. Guided - one or more "quadrants" of the tile will be present (from previously
   generated tiles) and the image model will be asked to generate the missing
   content of the image based on the reference images (whitebox.png, render.png,
   and one or more "style" references).

2. Unguided - the tile will be generated from scratch, with no previously
   generated tile content with which to fill.

---

[x] Modify `export_views.py` and the `web/main.js` script to be fully
parameterized by a specific `view.json`.

[x] Create a `plan_tiles.py` script that generates a plan for generating an m by
n set of tiles. The user should specificy initial params (e.g. name, lat/lng,
camera height, view angles, and m/n tiles) and then the planner should create
mXn tile directories in the `tile_plans/<name>` directory, each one
corresponding to a tile. Each tile MUST be offset and overlap by HALF of the
tile width/height depending on if it's above/below or left/right of the previous
tile. That is, if we have a 2 x 2 grid, then we'd have something like the
following tile definitions:

A B C D E F G H I

Tile 1: A B | D E Tile 2: B C | E F Tile 3: D E | G H Tile 4: E F | H I

The first tile (top left) should be centered on the specified lat/lng, and then
we need to calculate the lat/lng offset for each of the subsequent tiles based
on the isometric view frustum.

Each one of the tile directories (e.g. `tile_plans/<name>/001`) must be
populated with the `view.json` containing all of the camera metadata described
above and needed in `export_views.py`

[x] Create a `validate_plan.py` script that stitches together the generated
images from `export_views.py` into a single large image to verify that the
tiling logic is correct. This script should look for `render.png` and
`whitebox.png` in each tile directory and stitch them into `full_render.png` and
`full_whitebox.png` in the parent plan directory.

[x] Implement `generate_tile.py` script to generate isometric pixel art for a
single tile using the nano-banana notebook logic.

[ ] Update `generate_tile.py` to automatically generate templates from neighbors
using `create_template.py` logic before generation.
