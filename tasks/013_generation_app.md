# Generation App

Let's wire up generation to the view_generations.py web app! If the user has one
or more quadrants selected, they can hit "generate" button to send a request to
the python app.

We'll use the logic in `src/isometric_nyc/generation/generate_template.py`
to a) determine if the generation is valid, b) generate the template (using the
logic in generate_tile_omni.py) and c) extract the image tile data and save to
the database.

We need to handle load/error states in the web app, and when the request
succeeds, we need to automatically refresh the page to load the new tiles.

Let's use a simple notification/toast type system to notify the user if the
generation succeeds or fails - also, only one generation can happen at a time.
