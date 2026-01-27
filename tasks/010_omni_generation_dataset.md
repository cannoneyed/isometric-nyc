# Omni Generation Dataset

We want to combine all of the training datasets for full-tile generation,
quadrant-based generation, and inpainting generation (described in
006_infill_dataset and 009_inpainting.md)

We'll need a mixture of the following types of input data templates:

1. Full generation - the entire tile is `render` image pixels.
2. Quadrant generation - the tile is one quarter `render` image pixels with the
   remainder `generation` pixels. We need all variants for all four quadrants.
3. Half generation - the tile is one half `render` pixels and one half
   `generation` pixels. We need variants for all four halves.
4. Middle generation - the tile is one half `render` pixels in either a vertical
   or horizontal strip in the middle of the image, with the remainder
   `generation` pixels.
5. Rectangle strips - the tile contains either a full horizontal or full
   vertical strip of `render` pixels, with the remainder `generation` pixels,
   these can be be anywhere between 25 and 60 percent of the image and can be
   anywhere in the image.
6. Rectangle infills - each image contains a rectangle of between 25 and 60
   percent of the image area somewhere in the image bounds filled with `render`
   pixels, with the remainder `generation` pixels.

We want a distribution somewhere like the following:

Full - 20% Quadrant - 20% Half - 20% Middle - 15% Rectangle strips - 10%
Rectangle infills - 15%

Base the script to generate the data on
`src/isometric_nyc/synthetic_data/create_inpainting_examples.py`. Use the
render/generation images in `synthetic_data/datasets/v04` and create a directory
called `omni`.

All `render` pixels in the images must be outlined by a 1px red solid line
border that's on top of the image and does not displace any pixels.

Make sure you also generate a training data csv with the prompt "Fill in the
outlined section with the missing pixels corresponding to the
<isometric nyc pixel art> style, removing the border and exactly following the
shape/style/structure of the surrounding image (if present)."

Also create a small test set in the `test` directory of each of the variants
(using either of the two generation/render pairs)
