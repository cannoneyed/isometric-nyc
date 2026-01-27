# Debug Map

We need to create a `debug_map` script in `src/isometric_nyc/generation`.

The idea is very simple - the script should have a single parameter -
`generation_dir` (which should default to `generations/v01`). A python script
should scrape all of the x,y coordinates for generated quadrants in the sqlite
db. Then, we need to overlay an indicator (25% alpha red) rectangle of these
(isometric) quadrants over a real standard map of NYC so we can see how much
progress has been made.

This can be a python script that outputs an image.

Ensure that the isometric/standard projections are accounted for, as well as the
camera params that define a quadrant geo in `generation_config.json`.
