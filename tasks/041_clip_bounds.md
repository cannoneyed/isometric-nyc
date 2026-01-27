# Clip Bounds

I want to add a new parameter to src/isometric_nyc/generation/export_pmtiles.py - `bounds`, which lets the user specify a bounds json file (by default looking up the filename in the `src/isometric_nyc/generation/bounds` directory) and *truncate* the bounds of the tiles using the geometry defined in the bounds json. That is, if the edge of the json bounds intersects a tile, that tile should have generation image pixels *inside* the bounds and black pixels outside of it.
