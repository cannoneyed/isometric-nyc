# Synchronizing zoom levels in render scripts
_Exported on 11/30/2025 at 16:16:15 EST from Cursor (2.1.39)_

---

**User**

OK - I'm trying to debug an annoying issue - I have two render scripts for an orthographic view of NYC - @src/web/main.js uses google 3d tiles and @src/isometric_nyc/whitebox.py uses NYC geo data to generate a whitebox. I need them to be the exact same zoom level, but I can't get them to work - the web version seems to be a bit zoomed out compared to the whitebox output

---

**Cursor**

```python
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified
VIEW_HEIGHT_METERS = view_json.get("view_height_meters", 200)

# Viewport settings
VIEWPORT_WIDTH = view_json["width_px"]
VIEWPORT_HEIGHT = view_json["height_px"]

# Camera settings from view.json
CAMERA_ZOOM = VIEW_HEIGHT_METERS / 2  # Parallel scale - lower = more zoomed in, higher = more zoomed out
```

```javascript
  // Match whitebox.py visually
  // view.view_height_meters determines the vertical extent of the view in world units (meters)
  const frustumHeight = view.view_height_meters || 200;
  const halfHeight = frustumHeight / 2;
  const halfWidth = halfHeight * aspect;

  console.log(
    `📐 Frustum: height=${frustumHeight}m`
  );

  // Set frustum with calculated values
  ortho.top = halfHeight;
  ortho.bottom = -halfHeight;
  ortho.left = -halfWidth;
  ortho.right = halfWidth;
  
  // Reset zoom to 1.0 to ensure strict 1:1 scale with world units
  ortho.zoom = 1.0;
  ortho.updateProjectionMatrix();
```

```javascript
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
    zoom: camera.zoom,
  };
}

// Debounced logging of camera state
function logCameraState() {
  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1 ||
    Math.abs(info.zoom - lastCameraState.zoom) > 0.01;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
      zoom: info.zoom,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m Zoom=${info.zoom.toFixed(2)} | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}
```

```javascript
// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0, zoom: 1 };

init();
```

I've synchronized the zoom levels by strictly defining the **view height in meters** as the source of truth in `view.json`.

Here is what I changed:

1.  **Updated `src/view.json`**: Added `"view_height_meters": 200`. This serves as the single source of truth for the zoom level (vertical extent of the visible area).
2.  **Updated `src/isometric_nyc/whitebox.py`**: It now calculates `CAMERA_ZOOM` (parallel scale) as `view_height_meters / 2`. This ensures the static render covers exactly 200m vertically.
3.  **Updated `src/web/main.js`**:
    *   It now reads `view.view_height_meters` to set the orthographic frustum height.
    *   **Crucial Fix**: I forced `ortho.zoom = 1.0` during camera positioning. `GlobeControls` or the transition manager might have been applying a slight zoom-out factor (zoom < 1.0) initially, causing the web view to look "zoomed out" even if the frustum math was correct.
    *   Added `Zoom` to the debug logs so you can verify it stays at 1.00.

Now, if you want to change the zoom for *both* tools, just edit `"view_height_meters"` in 

`src/view.json`. A smaller number = more zoomed in.

---

**User**

Amazing! Can you add some way to wait for all of the tiles to be rendered in @src/web/main.js  in @src/isometric_nyc/export_views.py instead of waiting 10s to take the screenshot?

---

**Cursor**

```javascript
// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0, zoom: 1 };

// Tile loading tracking
let tilesStableStartTime = 0;
window.TILES_LOADED = false;

init();
animate();
```

```javascript
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Check for tile loading stability
  // We consider tiles loaded if downloading and parsing count is 0 for at least 1 second
  if (tiles.stats.downloading === 0 && tiles.stats.parsing === 0) {
    if (tilesStableStartTime === 0) {
      tilesStableStartTime = performance.now();
    } else if (performance.now() - tilesStableStartTime > 1000) {
      if (!window.TILES_LOADED) {
        window.TILES_LOADED = true;
        console.log("✅ Tiles fully loaded and stable");
      }
    }
  } else {
    tilesStableStartTime = 0;
    window.TILES_LOADED = false;
  }

  // Log camera state (debounced)
```

```python
    # Wait for tiles to load
    print("   ⏳ Waiting for tiles to stabilize (window.TILES_LOADED)...")
    try:
      page.wait_for_function("window.TILES_LOADED === true", timeout=60000)
      print("   ✅ Tiles loaded signal received")
    except Exception as e:
      print(f"   ⚠️  Timeout waiting for tiles to load: {e}")
      print("   📸 Taking screenshot anyway...")

    # Take screenshot
```

I've updated both `src/web/main.js` and `src/isometric_nyc/export_views.py` to implement a smarter wait mechanism:

1.  **`src/web/main.js`**: Added logic in the `animate()` loop to track `tiles.stats`. It sets a global variable `window.TILES_LOADED = true` only after the tile downloading and parsing counts have remained at 0 for 1 full second (to ensure stability).
2.  **`src/isometric_nyc/export_views.py`**: Replaced the hardcoded 10s sleep with `page.wait_for_function("window.TILES_LOADED === true", timeout=60000)`. This will now take the screenshot as soon as the view is ready, or wait up to 60s if the network is slow, defaulting to a screenshot if it times out.

You can run the export as usual:
```bash
uv run python src/isometric_nyc/export_views.py
```

```shellscript
uv run python src/isometric_nyc/export_views.py
```

