# Infill Dataset

For every tile subdir in synthetic_data/tiles/v04, we're going to create
synthetic data for infilled tile generation.

Each tile will have four quadrants:

A B C D

When generating a neighboring tile, we'll use sections of the tile that has
previously been generated in the neighboring quadrants. For example, if I have a
2x2 grid of tiles, then my tile grid will look like

A B E C D F G H I

Where tiles: 000_000 will contain quadrants A B C D 000_001 will contain
quadrants B E D F 001_000 will contain quadrants C D G H 001_001 will contain
quadrants D F H I

We'll progressively generate more and more tiles by "walking" one half tile in
any straight direction. For example, assuming that tile 000_000 has already been
generated:

Step 1 (Tile 000_001):

We'll need a "template" image where the left half will be the generated pixels
from the neighboring tile (000_000 - B and D) and the right half will be the
pixels from the 000_001 `render` image. This will generate pixels for (E and F)

Step 2 (Tile 001_000)

We'll need a "template" image where the top half will be the generated pixels
from the neighboring tile (000_000 - C and D) and the bottom half will be the
pixels from the 001_000 `render` image. This will generate pixels for (G and H)

Step 3 (Tile 001_001)

We'll need a "template" image where the top and left halves will be the
generated pixels from the neighboring tiles (000_001 - G and H) and (and 001_000
F), and we'll need a template where only the bottom right quadrant is from the
001_001 `render` image. This will generate pixels for (I)

# Dataset generation

Therefore, we need to generate 8 variants of data for fine-tuning a model on
this task:

1. Left half generated, right half rendered
2. Right half generated, left half rendered
3. Top half generated, bottom half rendered
4. Bottom half generated, top half rendered
5. Top left quadrant rendered, everything else generated
6. Top right quadrant rendered, everything else generated
7. Bottom left quadrant rendered, everything else generated
8. Bottom right quadrant rendered, everything else generated

For every tile subdir in synthetic_data/tiles/v04, we're going to create one of
each of these variations. For clarity, outline the RENDERED pixels with a
single-pixel-width red line.

The naming scheme should be as follows:

x = g if generated, r if rendered
infill*<top left quadrant x>*<top right quadrant x>_<bottom left x>_<bottom right x>.png

# Execution

Generate a script that will generate these infill variants for a given tile_dir,
or for all subdirectories in a tile_dir
