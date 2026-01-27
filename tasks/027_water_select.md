# Water Select

Next we're going to implement a new tool for the generation/app.py in order
to select water quadrantss. We'll need to

a) add a new "water tile" control to static/viewer.js that toggles whether or
not a quadrants is a water quadrants b) save this flag in the sqlite database
(likely as an is_water column) c) fetch this data from the server for the
currently visible quadrants and, when the "water tile" control is selected, show
an icon in the bottom right of the quadrant if it's a water quadrant
