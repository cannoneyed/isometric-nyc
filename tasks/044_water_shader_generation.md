# Water shader generation

We need to automatically generate in a step-by-step manner the water shader mask tiles for all water+shore quadrants in the quadrants db. Using the script defined in src/isometric_nyc/generation/detect_water_tiles.py, we need to be able to do two things:

## Part 1

1. Modify the generations db to have a "water_mask" image column that stores the png contents of the water mask, and a "water_type" column that contains one of the three conditions: ALL_WATER, ALL_LAND, WATER_EDGE.
2. For all tiles that are pure water (e.g. all the water color #4A6372), we should add a pure white png image to the "water_mask" column and set ALL_WATER in the water_type column
3. For all tiles that are pure land (e.g. none of the water color), we shoiuld add a pure black png image to the "water_mask" column and set ALL_LAND in the water_type column.
4. Update the src/generation/app.py to be able to show the water tiles, by adding a new drop-down menu that determines what tile to show (either render, generations, or water_mask)


## Part 2 - Manual water mask generation

We need to modify the src/isometric_nyc/generation/app.py to be able to generate water mask tiles - we need to add a new entry key "is_water_mask" in the app_config.json that, if that model is used to generate, will save the generated output quadrant data to the `water_mask` column in the generation db.


## Part 3 - automated water mask generation

After running the `populate_water_masks.py` script, we're going to try to write a script that automatically generate the missing quadrants from the database. Let's keep things very simple and do something like a random sample approach. The algorithm goes something like:

- Select a random quadrant of type `water_mask` that doesn not have any water mask generation data.
- If, by the rules of quadrant generation (enumerated  in generate_tile_omni and others) we can generate that quadrant using an omni model, then generate it. If not, continue to the next step
- Select the closest non-generated `water_mask` quadrant and generate it.

The script should have an optional x,y params that override the initial random selection.
