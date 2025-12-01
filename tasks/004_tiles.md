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
