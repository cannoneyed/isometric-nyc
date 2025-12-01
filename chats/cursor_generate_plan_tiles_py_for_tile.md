# Generate plan_tiles.py for tile directories
_Exported on 12/1/2025 at 16:55:45 EST from Cursor (2.1.39)_

---

**User**

Let's move on to the next step in @tasks/004_tiles.md - please generate a file `plan_tiles.py` that creates the tile directories according to the spec

---

**Cursor**

```markdown
[x] Create a `plan_tiles.py` script that generates a plan for generating an m by n set of tiles. The user should specificy initial params (e.g. name, lat/lng, camera height, view angles, and m/n tiles) and then the planner should create mXn tile directories in the `tile_plans/<name>` directory, each one corresponding to a tile. Each tile MUST be offset and overlap by HALF of the tile width/height depending on if it's above/below or left/right of the previous tile. That is, if we have a 2 x 2 grid, then we'd have something like the following tile definitions:

A B C
D E F
G H I

Tile 1: A B | D E
Tile 2: B C | E F
Tile 3: D E | G H
Tile 4: E F | H I

The first tile (top left) should be centered on the specified lat/lng, and then we need to calculate the lat/lng offset for each of the subsequent tiles based on the isometric view frustum.
```

I have created the `plan_tiles.py` script as requested.

Here are the details of the implementation:
*   **File:** `src/isometric_nyc/plan_tiles.py`
*   **Functionality:**
    *   Accepts parameters: `name`, `lat`, `lon`, `rows`, `cols`, `azimuth`, `elevation`, `view-height` (coverage), and dimensions.
    *   Calculates the correct lat/lon offset for each tile based on the isometric view frustum (derived from the logic in `whitebox.py`).
    *   Ensures 50% overlap by shifting subsequent tiles by half the viewport width/height in screen space.
    *   Creates the directory structure `tile_plans/<name>/<view_id>/` where `view_id` is a hash of the centroid.
    *   Generates a `view.json` file in each directory with all necessary metadata.

I verified the logic by running a test plan with a 2x2 grid and inspecting the output coordinates and file structure. I've also updated `tasks/004_tiles.md` to mark this step as complete.

You can run the script like this:
```bash
uv run python src/isometric_nyc/plan_tiles.py --name my_plan --lat 40.7505 --lon -73.9934 --rows

```shellscript
uv run python src/isometric_nyc/plan_tiles.py --name my_plan --lat 40.7505 --lon -73.9934 --rows
```

---

**User**

Please update @src/isometric_nyc/plan_tiles.py to a) name the directories <n_m> (e.g 000_000 for the top left corner) and add the row/col numbers to the view.json

---

**Cursor**

```python
      # Generate ID
      # Use row/col format: "rrr_ccc"
      view_id = f"{r:03d}_{c:03d}"

      # Create directory
      tile_dir = base_dir / view_id
      tile_dir.mkdir(exist_ok=True)

      # Create view.json
      view_data = {
        "name": f"{name} - {r}x{c}",
        "lat": lat,
        "lon": lon,
        "camera_azimuth_degrees": azimuth,
        "camera_elevation_degrees": elevation,
        "width_px": width_px,
        "height_px": height_px,
        "view_height_meters": view_height_meters,
        "grid_pos": {"row": r, "col": c},
        "row": r,
        "col": c,
      }

      with open(tile_dir / "view.json", "w") as f:
        json.dump(view_data, f, indent=2)

      print(f"  Tile [{r},{c}] -> {view_id}: {lat:.6f}, {lon:.6f}")
```

I have updated `src/isometric_nyc/plan_tiles.py`.

Changes made:
*   **Directory Naming:** Changed from a hash to `rrr_ccc` format (e.g., `000_000`, `000_001`) for easier identification of grid position.
*   **View JSON:** Added `row` and `col` as top-level keys in the `view.json` file, in addition to keeping them in the `grid_pos` object.

You can now generate the plan with the new naming convention:
```bash
uv run python src/isometric_nyc/plan_tiles.py --name test_plan --lat 40.7505 --lon -73.9934 --rows 2 --cols 2
```

```shellscript
uv run python src/isometric_nyc/plan_tiles.py --name test_plan --lat 40.7505 --lon -73.9934 --rows 2 --cols 2
```

---

**User**

Can you next update @src/isometric_nyc/export_views.py to use a headless chrome?

---

**Cursor**

```python
    # Launch browser in headless mode (ensure WebGL works)
    browser = p.chromium.launch(
      headless=True,
      args=[
        "--enable-webgl",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
      ],
    )
```

I've updated `src/isometric_nyc/export_views.py` to launch the browser in headless mode (`headless=True`), while keeping the WebGL-

related flags (`--enable-webgl`, etc.) to ensure rendering still works correctly.

---

**User**

OK - next, can we add a parameter to the @src/isometric_nyc/export_views.py to be able to process ALL tile directories in a planned out tile dir? If I pass in a tile_dir with multiple child directories, it should iterate over all of them and process all of the views inside each subdir

---

**Cursor**

I have updated `src/isometric_nyc/export_views.py`.

Changes made:
*   **Multi-tile support:** The script now checks if the provided `--tile_dir` contains a `view.json`. If not, it treats it as a parent directory and scans all subdirectories for `view.json` files.
*   **Batch Processing:**
    *   The main logic now builds a list of `tasks` (tuples of `output_dir` and `view_config`).
    *   Whitebox rendering iterates through this list.
    *   Web rendering uses a new `process_web_views` function that takes the entire list of tasks.
*   **Efficient Web Export:**
    *   The web server is started **once** (if not disabled).
    *   The Playwright browser instance is launched **once**.
    *   The script iterates through all views within the single browser session, creating new pages/contexts as needed for each view.
*   **Limit:** Added a `--limit` argument to limit the number of tiles processed (useful for testing).

You can now run the export on the entire plan directory:
```bash
uv run python src/isometric_nyc/export_views.py --tile_dir tile_plans/test_plan
```

```shellscript
uv run python src/isometric_nyc/export_views.py --tile_dir tile_plans/test_plan
```

---

**User**

OK, final step - add a `validate_plan` script that takes a `tile_dir` and stitches together all of the row/col images into one single image to validate that it works - you can save the image in the tile_dir as "full.png"

---

**Cursor**

call:default_api:EditNotebook{cell_idx:0,cell_language:मनdown,is_new_cell:false,new_string:mn[x] Create a `plan_tiles.py` script that generates a plan for generating an m by
n set of tiles. The user should specificy initial params (e.g. name, lat/lng,
camera height, view angles, and m/n tiles) and then the planner should create
mXn tile directories in the `tile_plans/<name>` directory, each one
corresponding to a tile. Each tile MUST be offset and overlap by HALF of the
tile width/height depending on if it's above/below or left/right of the previous
tile. That is, if we have a 2 x 2 grid, then we'd have something like the
following tile definitions:

A B C D E F G H I

Tile 1: A B | D E Tile 2: B C | E F Tile 3: D E | G H Tile 4: E F | H I

The first tile (top left) should be centered on the specified lat/lng, and then
we need to calculate the lat/lng offset for each of the subsequent tiles based
on the isometric view frustum.

Each one of the tile directories (e.g. `tile_plans/<name>/001`) must be
populated with the `view.json` containing all of the camera metadata described
above and needed in `export_views.py`

[x] Create a `validate_plan.py` script that stitches together the generated images from `export_views.py` into a single large image to verify that the tiling logic is correct. This script should look for `render.png` and `whitebox.png` in each tile directory and stitch them into `full_render.png` and `full_whitebox.png` in the parent plan directory.mn,old_string:mn[x] Create a `plan_tiles.py` script that generates a plan for generating an m by
n set of tiles. The user should specificy initial params (e.g. name, lat/lng,
camera height, view angles, and m/n tiles) and then the planner should create
mXn tile directories in the `tile_plans/<name>` directory, each one
corresponding to a tile. Each tile MUST be offset and overlap by HALF of the
tile width/height depending on if it's above/below or left/right of the previous
tile. That is, if we have a 2 x 2 grid, then we'd have something like the
following tile definitions:

A B C D E F G H I

Tile 1: A B | D E Tile 2: B C | E F Tile 3: D E | G H Tile 4: E F | H I

The first tile (top left) should be centered on the specified lat/lng, and then
we need to calculate the lat/lng offset for each of the subsequent tiles based
on the isometric view frustum.

Each one of the tile directories (e.g. `tile_plans/<name>/001`) must be
populated with the `view.json` containing all of the camera metadata described
above and needed in `export_views.py`mn,target_notebook:mntasks/004_tiles.mdmn}


```markdown
[x] Create a `plan_tiles.py` script that generates a plan for generating an m by
n set of tiles. The user should specificy initial params (e.g. name, lat/lng,
camera height, view angles, and m/n tiles) and then the planner should create
mXn tile directories in the `tile_plans/<name>` directory, each one
corresponding to a tile. Each tile MUST be offset and overlap by HALF of the
tile width/height depending on if it's above/below or left/right of the previous
tile. That is, if we have a 2 x 2 grid, then we'd have something like the
following tile definitions:

A B C D E F G H I

Tile 1: A B | D E Tile 2: B C | E F Tile 3: D E | G H Tile 4: E F | H I

The first tile (top left) should be centered on the specified lat/lng, and then
we need to calculate the lat/lng offset for each of the subsequent tiles based
on the isometric view frustum.

Each one of the tile directories (e.g. `tile_plans/<name>/001`) must be
populated with the `view.json` containing all of the camera metadata described
above and needed in `export_views.py`

[x] Create a `validate_plan.py` script that stitches together the generated images from `export_views.py` into a single large image to verify that the tiling logic is correct. This script should look for `render.png` and `whitebox.png` in each tile directory and stitch them into `full_render.png` and `full_whitebox.png` in the parent plan directory.
```

I have implemented the `validate_plan.py` script.

**Details:**
*   **File:** `src/isometric_nyc/validate_plan.py`
*   **Functionality:**
    *   Accepts a `tile_dir` path as an argument.
    *   Scans the directory for subdirectories matching the `rrr_ccc` pattern.
    *   Loads `render.png` and `whitebox.png` from each valid subdirectory.
    *   Stitches them together into a single large canvas, accounting for the 50% overlap (stride = 50% of width/height).
    *   Saves the output as `full_render.png` and `full_whitebox.png` in the `tile_dir`.

You can run the validation script like this:
```bash
uv run python src/isometric_nyc/validate_plan.py tile_plans/test_plan
```

I've also updated `tasks/004_tiles.md` to mark this task as complete.

```shellscript
uv run python src/isometric_nyc/validate_plan.py tile_plans/test_plan
```

