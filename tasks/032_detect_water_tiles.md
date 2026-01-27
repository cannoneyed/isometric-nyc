# Detect Water tiles

I want to write a script that marks all "water tiles" in the generation dir database. It should iterate over all tiles in the DB and then add a column flag called "is_water_tile" or something like that if the tile contains the color 4A6372. Make the script in `src/isometric_nyc/generation` and call it 'detect_water_tiles.py'.

Then, I want to add two features:

To the generation app in `src/isometric_nyc/generation/app.py` (and JS/HTML) - add a toggle that lets the user see or not see "water flag" tiles. If the toggle is on, mark the water flagged tile with a blue bounding box. This will require a new API route to get that tile/quadrant metadata from the db/server.

Also, to the `debug_map.py` script, make the water tiles blue squares instead of red squares in the map.
