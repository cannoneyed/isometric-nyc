## Data joining

We will implement a system to join the data from the different sources. The
goal is to provide a generative model, in this case - Google's Nano Banana
(Gemini Flash 2.5 Image) with a number of reference images to generate a pixel-
perfect isometric image of the building.

These isometric pixel art images need to be scaled **precisely** to the size
of the building footprint. So we need to generate a few intermediate images:

1. An **overhead** view of the building, outlined by the footprint. We'll generate this by
   grabbing a satellite image of a tile of fixed width and height centered on
   the building footprint. We'll then overlay an outline of the building
   footprint on top of this image.

### Plan
