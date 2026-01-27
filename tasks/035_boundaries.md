# Boundaries

We need to add a new "boundaries" function to the app - both src/isometric_nyc/generation/app.py (and static/viewer.js) and the debug_map.py.

First, let's extract the NYC GeoJSON boundary into a new generation/bounds/nyc.json file. This should be the default boundary for both the debug map and the generation app.py.

Second, we should be able to specify an alternate bounds.json as a CLI argument to both of these apps.

Finally, we need a new app for creating these bounds using a polygon editor.

## Bounds app

Let's create a new app in generation called `create_bounds.py`.

Let's use the same basic HTML setup as the debug_map.py html page - we should be able to view all generated and pending tiles projected onto a real map of NYC. We should also by default show the NYC.json. However we'll now add a new "bounds editor" layer that lets you do the following. 

- Start with a rectangular boundary, roguhly 50% of the screen size.
- any vertex is draggable and movable
- double clicking on a line between vertices adds a new vertex
- double clicking on any existing vertex deletes it
- vertices cannot be dragged through an existing line (e.g. must be regular polygons)
- there can be a minimum of three vertices.
- There should be a "save" button that saves the geojson of the bounding polygon to "generation/bounds" dir with a new name (prompted from the user).

We should be able to load an existing bounds by launching the app with that json as a param.
