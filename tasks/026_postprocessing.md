# Postprocessing

Let's take the functionality of src/isometric_nyc/pixelate.py and apply it to
the entire exported tile set in src/app/public/tiles. The only big change we
need to make is to ensure we use the same reduced color set for _all_ tiles
