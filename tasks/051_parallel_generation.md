# Parallel Generation

Rather than use the app in src/isometric_nyc/generation/app.py to generate quadrants, let's use a programmatic approach to generating an entire map's worth of quadrants.

The idea is as follows:

- create a script that initializes a new "layer" directory that has a json config file for the generation and a SQLite database for keeping track of the queue of quadrants to be generated.
- the config defines the generation_dir (which has the source map), the source image layer (either `renders` or `generations`)
- given a pool of model inference URLs (registered in the config file)
- first, generate a plan for how to generate quadrants. The algorithm for planning should "tile" in the following way:
  1. generate 2x2 quadrant tiles that have one quadrant between them, starting at the topmost row and extending down through the entire map
  2. generate the 1x2 and 2x1 gaps between the 2x2 quadrant tiles - note, that the 2x2 tiles on both side of the gap *must* be present in order for the 1x2 generation to be generated
  3. generate the 1x1 quadrants in the corners of the generations.
- this plan should be stored in a `progress.db` sqlite DB, with the order being ALL of step 1, followed by ALL of step 2, followed by ALL of step 3
- A script called `generate_full_map_layer.py` should be created in src/isometric_nyc/generation - it should take a --layer_dir directory and connect to the sqlite progress db. it should use the generation API endpoints in round-robin fashion with the given parameters in the config json, and go through each of the 3 steps until ALL quadrants have been generated.

- There must also be a `debug_generate_full_map_layer.py` that works very much like `src/isometric_nyc/generation/generate_debug_map.py` that takes a --layer_dir param that shows all TO BE GENERATED quadrants overlayed on the map, with the following color code: red - step 1 (2x2), green - step 2 (1x2 and 2x1) and blue - step 3 (1x1)

Decide on a plan (NO CODE) and write it below:

## Plan

### Overview

This plan describes a system for generating an entire map layer's worth of quadrants using multiple parallel model inference endpoints. The key insight is using a 3-step tiling strategy that avoids seams by generating 2x2 blocks with gaps between them, then filling the gaps.

### File Structure

```
layers/<layer_name>/
├── layer_config.json      # Configuration for this layer generation
├── progress.db            # SQLite database tracking generation progress
├── generations/           # Generated quadrant images (if stored as files)
│   ├── 0_0.png
│   ├── 1_0.png
│   └── ...
└── debug_map.html         # Visualization of generation progress
```

### Configuration (`layer_config.json`)

```json
{
  "name": "snow_layer",
  "generation_dir": "generations/nyc",
  "source_layer": "generations",  // or "renders"
  "model_endpoints": [
    {
      "name": "model_1",
      "url": "https://api.example.com/generate",
      "api_key_env": "MODEL_1_API_KEY"
    },
    {
      "name": "model_2",
      "url": "https://api.example.com/generate",
      "api_key_env": "MODEL_2_API_KEY"
    }
  ],
  "generation_params": {
    "prompt": "Transform to snowy winter scene...",
    "num_inference_steps": 14
  }
}
```

### Database Schema (`progress.db`)

```sql
-- Generation plan table
CREATE TABLE generation_plan (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step INTEGER NOT NULL,              -- 1, 2, or 3
  block_type TEXT NOT NULL,           -- '2x2', '1x2', '2x1', '1x1'
  top_left_x INTEGER NOT NULL,        -- Top-left X coordinate of the block
  top_left_y INTEGER NOT NULL,        -- Top-left Y coordinate of the block
  width INTEGER NOT NULL,             -- Block width in quadrants (1 or 2)
  height INTEGER NOT NULL,            -- Block height in quadrants (1 or 2)
  status TEXT NOT NULL DEFAULT 'pending',  -- pending, in_progress, complete, error
  model_name TEXT,                    -- Which model processed this (for tracking)
  started_at REAL,                    -- Timestamp when generation started
  completed_at REAL,                  -- Timestamp when generation completed
  error_message TEXT,                 -- Error message if status = 'error'

  -- Ordering ensures steps are processed in correct order
  -- All step 1 items first, then step 2, then step 3
  UNIQUE(step, top_left_x, top_left_y)
);

-- Index for efficient status queries
CREATE INDEX idx_plan_status ON generation_plan(status, step);

-- Metadata table
CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT
);
```

### Tiling Algorithm

The algorithm generates quadrants in 3 steps to avoid seams between generated regions:

#### Step 1: 2x2 Tiles with Gaps

Generate 2x2 quadrant blocks with a 1-quadrant gap between them. This creates a pattern like:

```
 x:  0  1  2  3  4  5  6  7  8
y:0  [A A] .  [B B] .  [C C] .
  1  [A A] .  [B B] .  [C C] .
  2   .   .   .   .   .   .   .
  3  [D D] .  [E E] .  [F F] .
  4  [D D] .  [E E] .  [F F] .
  5   .   .   .   .   .   .   .
```

**Start positions:** For a map with bounds (min_x, min_y) to (max_x, max_y):
- x positions: min_x, min_x+3, min_x+6, ... (while x+1 <= max_x)
- y positions: min_y, min_y+3, min_y+6, ... (while y+1 <= max_y)

Each 2x2 block covers quadrants: (x, y), (x+1, y), (x, y+1), (x+1, y+1)

#### Step 2: 1x2 and 2x1 Gaps

Fill the gaps between 2x2 tiles:

**Vertical 1x2 strips** (filling horizontal gaps):
- Located at x positions: min_x+2, min_x+5, min_x+8, ...
- For each x, generate strips at y: min_y, min_y+3, min_y+6, ...
- Each 1x2 covers: (x, y), (x, y+1)
- **Dependency:** Both adjacent 2x2 tiles (left and right) must be complete

**Horizontal 2x1 strips** (filling vertical gaps):
- Located at y positions: min_y+2, min_y+5, min_y+8, ...
- For each y, generate strips at x: min_x, min_x+3, min_x+6, ...
- Each 2x1 covers: (x, y), (x+1, y)
- **Dependency:** Both adjacent 2x2 tiles (above and below) must be complete

```
After Step 2:
 x:  0  1  2  3  4  5  6  7  8
y:0  [A A] v  [B B] v  [C C] .
  1  [A A] v  [B B] v  [C C] .
  2  [h h] .  [h h] .  [h h] .
  3  [D D] v  [E E] v  [F F] .
  4  [D D] v  [E E] v  [F F] .
  5  [h h] .  [h h] .  [h h] .
```

#### Step 3: 1x1 Corner Quadrants

Fill the remaining 1x1 gaps at the intersections:
- Located at positions where x = min_x+2, min_x+5, ... AND y = min_y+2, min_y+5, ...
- **Dependency:** All 4 surrounding blocks (step 1 and step 2) must be complete

```
After Step 3 (complete):
 x:  0  1  2  3  4  5  6  7  8
y:0  [A A] v  [B B] v  [C C] .
  1  [A A] v  [B B] v  [C C] .
  2  [h h] X  [h h] X  [h h] .
  3  [D D] v  [E E] v  [F F] .
  4  [D D] v  [E E] v  [F F] .
  5  [h h] X  [h h] X  [h h] .
```

Remember - the generated quadrants in this process are overlayed on top of existing quadrants, which may have irregular shape. The algorithm needs to pack the 2x2 quadrants as efficiently as possible within the already generated quadrants for the source layer.

### Scripts

#### 1. `init_layer.py`

**Location:** `src/isometric_nyc/generation/init_layer.py`

**Purpose:** Initialize a new layer directory with config and progress database.

**Usage:**
```bash
uv run python src/isometric_nyc/generation/init_layer.py \
  --name snow_layer \
  --generation-dir generations/nyc \
  --source-layer generations \
  --bounds "-10,-10,50,50" \
  --model-endpoints model_endpoints.json \
  --output layers/snow
```

**Behavior:**
1. Create the layer directory structure
2. Create `layer_config.json` with provided parameters
3. Query the source generation_dir to get map bounds (or use provided bounds)
4. Generate the tiling plan using the algorithm above
5. Populate `progress.db` with all generation items in order (step 1, step 2, step 3)
6. Print summary of plan (total items per step)

#### 2. `generate_full_map_layer.py`

**Location:** `src/isometric_nyc/generation/generate_full_map_layer.py`

**Purpose:** Execute the generation plan using multiple model endpoints.

**Usage:**
```bash
uv run python src/isometric_nyc/generation/generate_full_map_layer.py \
  --layer-dir layers/snow \
  [--max-concurrent 4] \
  [--resume]
```

**Behavior:**
1. Load `layer_config.json` and connect to `progress.db`
2. Process items in strict step order:
   - Complete ALL step 1 items before starting step 2
   - Complete ALL step 2 items before starting step 3
3. For each step:
   - Get all pending items for that step
   - Use round-robin assignment to distribute items across model endpoints
   - For each item:
     a. Mark as `in_progress` with current model
     b. Fetch source quadrant images from source generation_dir
     c. Stitch into appropriate block size (2x2, 1x2, 2x1, or 1x1)
     d. Call model API with stitched image
     e. Save generated quadrant(s) to layer's generations directory
     f. Mark as `complete` or `error` in progress.db
   - If an item fails, log error and continue with next item
4. Print progress periodically (items complete / total)
5. Support `--resume` to continue from where it left off

**Round-Robin Logic:**
```python
model_index = 0
for item in pending_items:
    model = model_endpoints[model_index % len(model_endpoints)]
    process_item(item, model)
    model_index += 1
```

#### 3. `debug_generate_full_map_layer.py`

**Location:** `src/isometric_nyc/generation/debug_generate_full_map_layer.py`

**Purpose:** Visualize the generation plan overlayed on the map.

**Usage:**
```bash
uv run python src/isometric_nyc/generation/debug_generate_full_map_layer.py \
  --layer-dir layers/snow \
  [--output debug_layer_plan.html]
```

**Behavior:**
1. Load layer config and progress database
2. Load the source generation_dir's config (for coordinate system)
3. For each item in the generation plan:
   - Calculate geographic corners using `calculate_quadrant_corners()`
   - Assign color based on step:
     - **Red (rgba(255, 99, 71, 0.4)):** Step 1 (2x2 blocks)
     - **Green (rgba(50, 205, 50, 0.4)):** Step 2 (1x2 and 2x1 strips)
     - **Blue (rgba(30, 144, 255, 0.4)):** Step 3 (1x1 corners)
   - Add status indicator (pending vs complete) via opacity/border
4. Generate Leaflet.js HTML map similar to `debug_map.py`
5. Include legend with color coding
6. Include stats panel showing progress per step

### Implementation Notes

1. **Source Layer Selection:**
   - If `source_layer = "generations"`: Use AI-generated pixel art from quadrants.generation column
   - If `source_layer = "renders"`: Use 3D renders from quadrants.render column

2. **Error Handling:**
   - Items with errors should be logged but not block other items
   - Provide a `--retry-errors` flag to reprocess failed items
   - Store error message in progress.db for debugging

3. **Parallelization (Future):**
   - The current design uses sequential round-robin
   - Future enhancement: Use ThreadPoolExecutor for concurrent API calls
   - Each model endpoint can process one item at a time

4. **Progress Tracking:**
   - Update progress.db atomically to support resume
   - Print periodic status: `[Step 1] 45/100 complete (45%), 2 errors`

5. **Output Storage:**
   - For now, store generated images in layer's `generations/` directory as `{x}_{y}.png`
   - Alternatively, store in a separate SQLite database similar to quadrants.db
