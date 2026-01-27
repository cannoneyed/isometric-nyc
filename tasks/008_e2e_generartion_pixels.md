# Isometric NYC E2E Generation

Alright - next we need to add a new script `generate_tile.py` that relies on
some of the logic from `render_tile.py` - if any of that logic is common, factor
it out into a `shared.py` file

First, we need to figure out the logic for generatiing pixel art versions of the
rendered tiles. Heres' how the logic needs to work:

1. The generate tile script takes a given x y coordinate, e.g. 0 0
2. If no `render` entry exists for any of the four quadrants in that tile, use
   the logic in `render_tile.py` to render those tile quadrants and save them to
   the db
3. If no `generated` neighbor quadrants exist, then we're generating from
   scratch. Use the logic in `src/isometric_nyc/generate_tile_oxen.py` to
   generate the pixel art generation, and save those quadrants to the generated
   table. You'll need to stitch together the four rendered quadrants and save it
   to a `renders/<x>_<y>.png` file, and also save the generation to
   `generations/<x>_<y>.png`.
4. If there are "neighboring" tiles/quadrants, use the logic in
   `src/isometric_nyc/generate_tile_oxen.py` to generate the pixel art
   generation. Likewise, save those quadrants to the generated table. You'll
   need to stitch together the four rendered quadrants and save it to a
   `renders/<x>_<y>.png` file, and also save the generation to
   `generations/<x>_<y>.png`.

Please implement all of this logic, extracting out common/shared logic (or
importing them from the respective source scripts)
