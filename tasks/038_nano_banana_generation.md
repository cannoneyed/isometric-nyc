# Generation with nano banana

We need to add a new feature to the src/isometric_nyc/generation/app.py (and viewer.js) app - the ability to generate quadrants with nano banana. We should add a new drop down model option called "Nano Banana" - if that model is selected, then we need to make a different kind of queued request to the app, and use the logic in  src/isometric_nyc/generation/generate_nano_banana.py

We also need to be able to mark certain quadrants as "references" for the nano banana generation - we need another button / "layer" to the app that can toggle on/off "reference" status (and save this in the database) - these references will be supplied to the nano banana generation step.
