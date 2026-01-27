# Generation playbook

> **Status: COMPLETED** ✅

## Task Description

Next, I want to create a script that works like the logic in
`src/isometric_nyc/generation/view_generations.py`, except it generates
quadrants programatically - please extract the logic out of that web server into
a "generate_omni.py" library, and use that logic for both the `view_generations`
script and a new `generate_tiles_omni` script.

The script should take a few parameters:

- generation_dir - the directory containing the generation config

- quadrants - a list of comma separated quadrant tuples to generate (e.g.
  "(0,1),(0,2)")
- quadrants_json - a json file containing a list of quadrant generations to
  generate in order

If provided a quadrants_json parameter, the JSON schema will look like this:

```
[{
  "quadrants": "(x,y),(x,y)",
  "status": "pending"|"done"|"error"
}, ...]
```

If given a quadrants json, the script should generate the quadrants in order
until done or an error is encountered, and update the json file accordingly
if/when an entry is finished. It should also be able to pick up where it left
off.

## Implementation

### Files Created/Modified

1. **`src/isometric_nyc/generation/generate_omni.py`** (NEW)

   - Reusable library with core generation logic
   - `parse_quadrant_tuple()` / `parse_quadrant_list()` - parsing quadrant
     strings
   - `call_oxen_api()` - calling the Oxen API
   - `download_image_to_pil()` - downloading images
   - `render_quadrant()` - rendering a quadrant via Playwright
   - `run_generation_for_quadrants()` - main generation pipeline

2. **`src/isometric_nyc/generation/generate_tiles_omni.py`** (NEW)

   - Command-line script for batch generation
   - Supports `--quadrants "(0,1),(0,2)"` for direct specification
   - Supports `--quadrants-json path/to/file.json` for batch processing
   - Updates JSON file after each entry for resume capability

3. **`src/isometric_nyc/generation/view_generations.py`** (MODIFIED)
   - Now imports from `generate_omni.py`
   - Uses shared `run_generation_for_quadrants()` function
   - Removed duplicated code

### Usage

```bash
# Generate specific quadrants:
uv run python src/isometric_nyc/generation/generate_tiles_omni.py \
  <generation_dir> \
  --quadrants "(0,1),(0,2)"

# Process a batch JSON file:
uv run python src/isometric_nyc/generation/generate_tiles_omni.py \
  <generation_dir> \
  --quadrants-json path/to/quadrants.json
```
