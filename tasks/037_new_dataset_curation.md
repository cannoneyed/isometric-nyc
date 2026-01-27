# Dataset Curation

1. We need to add a new feature to the app defined in src/isometric_nyc/generation/app.py (and static/viewer.js). We need to define a new boolean attribute called "starred" which we can apply, just like flagged. This needs to save the "starred" entry as a column to the database. This should be toggleable on/off via a button on the frontend toolbar, and *only one* quadrant is allowed to be selected and starred (if multiple quadrants are selected, then the "star" button must be disabled). In addition, we need to render the starred quadrants with a yellow outline and star icon in the top right.

2. We also need a feature in the client app that lets the user cycle through starred entries. We should be able to open a dialog that contains all starred entries (with their x,y coordinates listed) and click on the entry to go to that location (it should be centered in the viewport calculated by the nx and ny number of displayed quadrants).

3. Finally, we'll generate a new script called `export_starred_from_db.py` that reads all starred quadrants from the database, and does the following: given a `name` flag, creates a new dir in `synthetic_data/datasets` and populates two directories: `generations` and `renders` with 2x2 quadrant tile images based on the x,y starred quadrant in the top left of the tile.
