# Dark Mode generation

We need to automatically generate in a step-by-step manner the dark mode quadrant tiles for all quadrants in the quadrants db.

## Part 1

1. Modify the generations db to have a "dark_mode" image column that stores the png contents of the dark mode generation.
2. Update the src/generation/app.py to be able to show the dark mode tiles, by adding a new item to the drop-down menu that determines what tile to show (either render, generations, dark_mode, or water_mask)


## Part 2 - Manual dark mode generation

We need to modify the src/isometric_nyc/generation/app.py to be able to generate dark mode tiles - we need to add a new entry key "is_dark_mode" in the app_config.json that, if that model is used to generate, will save the generated output quadrant data to the `dark_mode` column in the generation db.


## Part 3 - automated dark mode generation

After running the `populate_dark_mode.py` script, we're going to try to write a script that automatically generate the missing quadrants from the database. Let's keep things very simple and use the logic for the `generate_rectangle` command in the app.

- The script should take a tl and br "(x,y)" params, then use the generation logic in generation/app.py to enqueue and then generate all tiles/quadrants for that rectangle.
- Track the progress of these generations and save the queued generations in the database
