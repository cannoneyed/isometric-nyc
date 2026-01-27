# Nano Banana Generation

We need a new script in src/isometric_nyc/generation called `generate_tile_nano_banana.py` that uses almost all of the same logic as `generate_tile_omni.py`, except we use a nano banana generation (using the nano banana generation logic in src/isometric_nyc/generate_tile_v2.py)

We should accept the following parameters:

generation_dir: Path to the generation directory containing quadrants.db
x: Quadrant x coordinate (tile anchor x, or target x if --target-position is used)
y: Quadrant y coordinate (tile anchor y, or target y if --target-position is used)
target-position: Position of target quadrant (x,y) within the 2x2 tile context. tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right. If specified, (x,y) is the target to generate and surrounding quadrants provide context. If not specified, (x,y) is the tile anchor and all
prompt: An additional prompt to give to nano banana generation model.
save: true by default, but if false, doesn't save the generated quadrants to the database.
references: A list of "(x,y)" coordinates for previously generated tl quadrants that will be loaded from the db, spliced into 2x2 image tiles, and attached as references to the prompt.

We need to do the same template construction and generated pixel splicing as in the generate_tile_omni.py script. In addition, we need to ensure that the generated image is resized to 1024x1024 (so we have 512x512 pixel quadrants). 

We should also supply 2-3 reference images to the model - these should be specified in the params.
