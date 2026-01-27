### Debug & bounds map

The `debug_map.py` script generates a visualization showing generation progress
by overlaying generated quadrants on a real map of NYC.

```bash
# Generate debug map for the default generation directory
uv run python src/isometric_nyc/generation/debug_map.py

# Generate for a specific generation directory
uv run python src/isometric_nyc/generation/debug_map.py generations/nyc

# Custom output path
uv run python src/isometric_nyc/generation/debug_map.py generations/nyc -o ./my_debug_map.png
```

**Features:**

- Overlays 25% alpha red rectangles for each generated quadrant
- Shows the seed point (green marker) as the generation origin
- Accounts for the isometric projection — quadrants appear as parallelograms
  matching the camera azimuth angle
- Uses CartoDB basemap tiles for geographic context
- Outputs to `debug_map.png` in the generation directory by default

**Output example:** Shows coverage gaps, generation progress, and geographic
extent of the current generation.

### Bounds Editor

The `create_bounds.py` script provides an interactive polygon editor for
defining custom generation boundaries.

```bash
# Create new bounds
uv run python src/isometric_nyc/generation/create_bounds.py generations/nyc

# Edit existing bounds file
uv run python src/isometric_nyc/generation/create_bounds.py generations/nyc --load bounds/my-region.json

# Custom port
uv run python src/isometric_nyc/generation/create_bounds.py generations/nyc --port 8888
```

**Features:**

- Interactive polygon editor on a Leaflet map
- View generated and pending tiles overlaid on real NYC geography
- NYC boundary displayed by default for reference
- Drag vertices to reshape the boundary
- Double-click on an edge to add a new vertex
- Double-click on a vertex to delete it (minimum 3 vertices)
- Self-intersection validation with warnings
- Save boundaries to `generation/bounds/` directory

**Controls:**

| Action             | How                       |
| ------------------ | ------------------------- |
| Move vertex        | Drag the vertex marker    |
| Add vertex         | Double-click on edge      |
| Delete vertex      | Double-click on vertex    |
| Reset to rectangle | Click "Reset" button      |
| Clear all          | Click "Clear" button      |
| Save               | Enter name, click "Save"  |

### Custom Boundaries

Both `debug_map.py` and `app.py` support custom boundary files:

```bash
# Debug map with custom bounds
uv run python src/isometric_nyc/generation/debug_map.py generations/nyc --bounds bounds/my-region.json

# Generation app with custom bounds
uv run python src/isometric_nyc/generation/app.py generations/nyc --bounds bounds/my-region.json
```

Boundary files are stored in `src/isometric_nyc/generation/bounds/`:

```
bounds/
├── nyc.json          # Default NYC borough boundaries
└── my-region.json    # Custom region (created with bounds editor)
```
