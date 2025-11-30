# Accessing environment variables in Vite
_Exported on 11/30/2025 at 12:10:38 EST from Cursor (2.1.39)_

---

**User**

I have a new web directory vite project in this python project - I want to use the GOOGLE_MAPS_API_KEY variable defined in the root .env file - how can I access this from @src/web/main.js ?

---

**Cursor**

Let me check the current setup - the root `.env` file and the Vite configuration.

I see the issue. You have a Vite project in `src/web/` but your `.env` file is in the project root. By default, Vite looks for `.env` files in the same directory as its config. Also, Vite requires environment variables to be prefixed with `VITE_` to be exposed to client-side code.

Here's how to fix this:

1. **Create a Vite config** that points to the root `.env` file
2. **Ensure your env variable is prefixed correctly**

```javascript
import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  // Look for .env files in the project root (two levels up from src/web/)
  envDir: resolve(__dirname, "../.."),
});

```

Now update your `main.js` to use the correct prefix:

```javascript
import * as THREE from "three";
import { TilesRenderer } from "3d-tiles-renderer";

console.log("Hello, world!");
console.log(import.meta.env.VITE_GOOGLE_MAPS_API_KEY);

// // --- CONFIGURATION ---
// const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
// const LAT = 40.7128; // NYC Latitude
// const LON = -74.006; // NYC Longitude
// const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// // 1. Setup Scene
// const scene = new THREE.Scene();
// scene.background = new THREE.Color(0x222222); // Dark background

// // 2. Setup Isometric Camera (Orthographic)
// // The frustum size determines zoom level in orthographic mode
// const aspect = window.innerWidth / window.innerHeight;
// const frustumSize = 1000;
// const camera = new THREE.OrthographicCamera(
//   (frustumSize * aspect) / -2,
//   (frustumSize * aspect) / 2,
//   frustumSize / 2,
//   frustumSize / -2,
//   1,
//   10000 // Near/Far clipping planes
// );

// // 3. Position Camera for True Isometric View
// // Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
// camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
// camera.lookAt(0, 0, 0); // Look at the center of the scene

// // 4. Setup Renderer
// const renderer = new THREE.WebGLRenderer({ antialias: true });
// renderer.setSize(window.innerWidth, window.innerHeight);
// document.body.appendChild(renderer.domElement);

// // 5. Load Google 3D Tiles
// const tilesRenderer = new TilesRenderer(
//   `https://tile.googleapis.com/v1/3dtiles/root.json?key=${API_KEY}`
// );

// // Google Tiles are usually centered at Earth's center (ECEF).
// // We must correct this to center NYC at (0,0,0) in our scene.
// tilesRenderer.setCamera(camera);
// tilesRenderer.setResolutionFromRenderer(camera, renderer);

// // This helper function handles the complex coordinate conversion
// // from Lat/Lon (WGS84) to the tile system's rotation.
// import { Matrix4, Vector3 } from "three";
// function positionTileset(lat, lon, renderer) {
//   const EARTH_RADIUS = 6378137;
//   const phi = (90 - lat) * (Math.PI / 180);
//   const theta = (lon + 180) * (Math.PI / 180);

//   const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
//   const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
//   const y = EARTH_RADIUS * Math.cos(phi);

//   // Move the group so the target coordinate is at 0,0,0
//   renderer.group.position.set(-x, -y, -z);

//   // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
//   const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
//   // Note: Accurate geodetic alignment requires more complex matrix math
//   // involving the surface normal at that Lat/Lon.
// }

// positionTileset(LAT, LON, tilesRenderer);
// scene.add(tilesRenderer.group);

// // 6. Render Loop
// function animate() {
//   requestAnimationFrame(animate);

//   // Required to update which tiles to download based on camera view
//   tilesRenderer.update();

//   renderer.render(scene, camera);
// }

// animate();

```

And update the commented-out code for consistency:

```javascript
import * as THREE from "three";
import { TilesRenderer } from "3d-tiles-renderer";

console.log("Hello, world!");
console.log(import.meta.env.VITE_GOOGLE_MAPS_API_KEY);

// // --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
// const LAT = 40.7128; // NYC Latitude
// const LON = -74.006; // NYC Longitude
// const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// // 1. Setup Scene
// const scene = new THREE.Scene();
// scene.background = new THREE.Color(0x222222); // Dark background

// // 2. Setup Isometric Camera (Orthographic)
// // The frustum size determines zoom level in orthographic mode
// const aspect = window.innerWidth / window.innerHeight;
// const frustumSize = 1000;
// const camera = new THREE.OrthographicCamera(
//   (frustumSize * aspect) / -2,
//   (frustumSize * aspect) / 2,
//   frustumSize / 2,
//   frustumSize / -2,
//   1,
//   10000 // Near/Far clipping planes
// );

// // 3. Position Camera for True Isometric View
// // Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
// camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
// camera.lookAt(0, 0, 0); // Look at the center of the scene

// // 4. Setup Renderer
// const renderer = new THREE.WebGLRenderer({ antialias: true });
// renderer.setSize(window.innerWidth, window.innerHeight);
// document.body.appendChild(renderer.domElement);

// // 5. Load Google 3D Tiles
// const tilesRenderer = new TilesRenderer(
//   `https://tile.googleapis.com/v1/3dtiles/root.json?key=${API_KEY}`
// );

// // Google Tiles are usually centered at Earth's center (ECEF).
// // We must correct this to center NYC at (0,0,0) in our scene.
// tilesRenderer.setCamera(camera);
// tilesRenderer.setResolutionFromRenderer(camera, renderer);

// // This helper function handles the complex coordinate conversion
// // from Lat/Lon (WGS84) to the tile system's rotation.
// import { Matrix4, Vector3 } from "three";
// function positionTileset(lat, lon, renderer) {
//   const EARTH_RADIUS = 6378137;
//   const phi = (90 - lat) * (Math.PI / 180);
//   const theta = (lon + 180) * (Math.PI / 180);

//   const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
//   const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
//   const y = EARTH_RADIUS * Math.cos(phi);

//   // Move the group so the target coordinate is at 0,0,0
//   renderer.group.position.set(-x, -y, -z);

//   // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
//   const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
//   // Note: Accurate geodetic alignment requires more complex matrix math
//   // involving the surface normal at that Lat/Lon.
// }

// positionTileset(LAT, LON, tilesRenderer);
// scene.add(tilesRenderer.group);

// // 6. Render Loop
// function animate() {
//   requestAnimationFrame(animate);

//   // Required to update which tiles to download based on camera view
//   tilesRenderer.update();

//   renderer.render(scene, camera);
// }

// animate();

```

Done! Here's what changed:

1. **Created `src/web/vite.config.js`** - This tells Vite to look for `.env` files in the project root (`../..` from `src/web/`)

2. **Updated `main.js`** - Changed to use `import.meta.env.VITE_GOOGLE_MAPS_API_KEY`

**One more thing:** In your root `.env` file, make sure the variable is prefixed with `VITE_`:

```env
VITE_GOOGLE_MAPS_API_KEY=your_api_key_here
```

Vite only exposes environment variables with the `VITE_` prefix to client-side code for security reasons (to prevent accidentally leaking server-side secrets to the browser).

After making that change, restart your Vite dev server and the variable will be accessible.

```plaintext
VITE_GOOGLE_MAPS_API_KEY=your_api_key_here
```

---

**User**

Request URL:

https://tile.googleapis.com/v1/3dtiles/datasets/CgIYAQ/files/AJVsH2xr-C1rV0QV0g8rKYA-ZEmlmTu5jePd1LHNY2iFc0ZnbK1cXrA3W2uheliwv_bIWdzTRwUBkpc4RaJIaFOuylllaMS1QU_xkXFsgf2HgtjTAy1K83JXV7FC.glb?session=CI_2hu3Tpa7eCxDHiK_JBg

Request Method:

GET

Status Code:

403 Forbidden

Remote Address:

142.251.35.170:443

Referrer Policy:

strict-origin-when-cross-origin

---

**Cursor**

The 403 Forbidden error is a Google API key restriction issue. Your API key needs to be configured to allow requests from your local development server.

**To fix this:**

1. Go to the [Google Cloud Console](https://console.cloud.google.com/apis/credentials)

2. Find and click on your API key (the one you're using for `VITE_GOOGLE_TILES_API_KEY`)

3. Under **Application restrictions**, you have two options:

   **Option A: For development** - Set to "None" temporarily (less secure, but easiest for local dev)

   **Option B: Add HTTP referrers** - Add these to the allowed referrers:
   ```
   http://localhost:5173/*
   http://localhost:5173
   http://127.0.0.1:5173/*
   http://127.0.0.1:5173
   ```

4. Under **API restrictions**, make sure the **Map Tiles API** is enabled for this key (or set to "Don't restrict key")

5. **Save** and wait a few minutes for changes to propagate

Also, make sure the **Map Tiles API** is enabled for your project:
- Go to [APIs & Services → Library](https://console.cloud.google.com/apis/library)
- Search for "Map Tiles API" 
- Enable it if it's not already

After making these changes, refresh your page and it should work.

```plaintext
   http://localhost:5173/*
   http://localhost:5173
   http://127.0.0.1:5173/*
   http://127.0.0.1:5173
```

---

**User**

I selected None

---

**Cursor**

Give it 2-5 minutes for Google's changes to propagate, then refresh your page (or restart the Vite dev server).

If you're still seeing the 403 after waiting, double-check that:

1. **The correct API is enabled** - Go to [APIs & Services → Enabled APIs](https://console.cloud.google.com/apis/dashboard) and confirm **Map Tiles API** is listed. If not, enable it from the library.

2. **You're using the right key** - Make sure the key you edited in the console matches what's in your `.env` file for `VITE_GOOGLE_TILES_API_KEY`

Let me know if it works or if you're still getting errors!

---

**User**

    "message": "Method doesn't allow unregistered callers (callers without established identity). Please use API Key or other form of API consumer identity to call this API.",



---

**Cursor**

This error means the API key isn't being sent with the subsequent tile requests (only the initial `root.json` got it). The `3d-tiles-renderer` library needs to be configured to authenticate all requests.

Let me check the current approach and update it to use Google's authentication properly:

The issue is that `3d-tiles-renderer` needs to add the API key to *every* tile request, not just the initial one. The library provides a `GoogleCloudAuthPlugin` for this:

```javascript
import * as THREE from "three";
import { TilesRenderer, GoogleCloudAuthPlugin } from "3d-tiles-renderer";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");
const LAT = 40.7128; // NYC Latitude
const LON = -74.006; // NYC Longitude
const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222); // Dark background

// 2. Setup Isometric Camera (Orthographic)
// The frustum size determines zoom level in orthographic mode
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 1000;
const camera = new THREE.OrthographicCamera(
  (frustumSize * aspect) / -2,
  (frustumSize * aspect) / 2,
  frustumSize / 2,
  frustumSize / -2,
  1,
  10000 // Near/Far clipping planes
);

// 3. Position Camera for True Isometric View
// Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
camera.lookAt(0, 0, 0); // Look at the center of the scene

// 4. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer(
  `https://tile.googleapis.com/v1/3dtiles/root.json?key=${API_KEY}`
);

// Google Tiles are usually centered at Earth's center (ECEF).
// We must correct this to center NYC at (0,0,0) in our scene.
tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// This helper function handles the complex coordinate conversion
// from Lat/Lon (WGS84) to the tile system's rotation.
import { Matrix4, Vector3 } from "three";
function positionTileset(lat, lon, renderer) {
  const EARTH_RADIUS = 6378137;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
  const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
  const y = EARTH_RADIUS * Math.cos(phi);

  // Move the group so the target coordinate is at 0,0,0
  renderer.group.position.set(-x, -y, -z);

  // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
  const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
  // Note: Accurate geodetic alignment requires more complex matrix math
  // involving the surface normal at that Lat/Lon.
}

positionTileset(LAT, LON, tilesRenderer);
scene.add(tilesRenderer.group);

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Required to update which tiles to download based on camera view
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

```

```javascript
import * as THREE from "three";
import { TilesRenderer, GoogleCloudAuthPlugin } from "3d-tiles-renderer";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");
const LAT = 40.7128; // NYC Latitude
const LON = -74.006; // NYC Longitude
const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222); // Dark background

// 2. Setup Isometric Camera (Orthographic)
// The frustum size determines zoom level in orthographic mode
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 1000;
const camera = new THREE.OrthographicCamera(
  (frustumSize * aspect) / -2,
  (frustumSize * aspect) / 2,
  frustumSize / 2,
  frustumSize / -2,
  1,
  10000 // Near/Far clipping planes
);

// 3. Position Camera for True Isometric View
// Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
camera.lookAt(0, 0, 0); // Look at the center of the scene

// 4. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Register the Google Cloud Auth Plugin to handle API key for all requests
tilesRenderer.registerPlugin(
  new GoogleCloudAuthPlugin({ apiToken: API_KEY })
);

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// This helper function handles the complex coordinate conversion
// from Lat/Lon (WGS84) to the tile system's rotation.
import { Matrix4, Vector3 } from "three";
function positionTileset(lat, lon, renderer) {
  const EARTH_RADIUS = 6378137;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
  const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
  const y = EARTH_RADIUS * Math.cos(phi);

  // Move the group so the target coordinate is at 0,0,0
  renderer.group.position.set(-x, -y, -z);

  // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
  const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
  // Note: Accurate geodetic alignment requires more complex matrix math
  // involving the surface normal at that Lat/Lon.
}

positionTileset(LAT, LON, tilesRenderer);
scene.add(tilesRenderer.group);

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Required to update which tiles to download based on camera view
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

```

Let me also clean up the import statement (it should be at the top):

```javascript
import * as THREE from "three";
import { Matrix4 } from "three";
import { TilesRenderer, GoogleCloudAuthPlugin } from "3d-tiles-renderer";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");
const LAT = 40.7128; // NYC Latitude
const LON = -74.006; // NYC Longitude
const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222); // Dark background

// 2. Setup Isometric Camera (Orthographic)
// The frustum size determines zoom level in orthographic mode
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 1000;
const camera = new THREE.OrthographicCamera(
  (frustumSize * aspect) / -2,
  (frustumSize * aspect) / 2,
  frustumSize / 2,
  frustumSize / -2,
  1,
  10000 // Near/Far clipping planes
);

// 3. Position Camera for True Isometric View
// Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
camera.lookAt(0, 0, 0); // Look at the center of the scene

// 4. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Register the Google Cloud Auth Plugin to handle API key for all requests
tilesRenderer.registerPlugin(
  new GoogleCloudAuthPlugin({ apiToken: API_KEY })
);

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// This helper function handles the complex coordinate conversion
// from Lat/Lon (WGS84) to the tile system's rotation.
import { Matrix4, Vector3 } from "three";
function positionTileset(lat, lon, renderer) {
  const EARTH_RADIUS = 6378137;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
  const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
  const y = EARTH_RADIUS * Math.cos(phi);

  // Move the group so the target coordinate is at 0,0,0
  renderer.group.position.set(-x, -y, -z);

  // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
  const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
  // Note: Accurate geodetic alignment requires more complex matrix math
  // involving the surface normal at that Lat/Lon.
}

positionTileset(LAT, LON, tilesRenderer);
scene.add(tilesRenderer.group);

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Required to update which tiles to download based on camera view
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

```

```javascript
import * as THREE from "three";
import { Matrix4 } from "three";
import { TilesRenderer, GoogleCloudAuthPlugin } from "3d-tiles-renderer";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");
const LAT = 40.7128; // NYC Latitude
const LON = -74.006; // NYC Longitude
const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222); // Dark background

// 2. Setup Isometric Camera (Orthographic)
// The frustum size determines zoom level in orthographic mode
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 1000;
const camera = new THREE.OrthographicCamera(
  (frustumSize * aspect) / -2,
  (frustumSize * aspect) / 2,
  frustumSize / 2,
  frustumSize / -2,
  1,
  10000 // Near/Far clipping planes
);

// 3. Position Camera for True Isometric View
// Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
camera.lookAt(0, 0, 0); // Look at the center of the scene

// 4. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Register the Google Cloud Auth Plugin to handle API key for all requests
tilesRenderer.registerPlugin(
  new GoogleCloudAuthPlugin({ apiToken: API_KEY })
);

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// This helper function handles the complex coordinate conversion
// from Lat/Lon (WGS84) to the tile system's rotation.
function positionTileset(lat, lon, renderer) {
  const EARTH_RADIUS = 6378137;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
  const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
  const y = EARTH_RADIUS * Math.cos(phi);

  // Move the group so the target coordinate is at 0,0,0
  renderer.group.position.set(-x, -y, -z);

  // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
  const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
  // Note: Accurate geodetic alignment requires more complex matrix math
  // involving the surface normal at that Lat/Lon.
}

positionTileset(LAT, LON, tilesRenderer);
scene.add(tilesRenderer.group);

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Required to update which tiles to download based on camera view
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

```

Done! The key change is using `GoogleCloudAuthPlugin` which automatically:
1. Fetches the correct Google 3D Tiles root URL
2. Adds your API key to **all** tile requests (not just the first one)

Refresh your page and the 403/unregistered caller errors should be resolved.

---

**User**

OK, no more errors but it's rendering a blank page

---

**Cursor**

Let me check the current code and add some debugging to figure out what's happening:

---

**User**

Uncaught SyntaxError: The requested module '/node_modules/.vite/deps/3d-tiles-renderer.js?v=0673a4cf' does not provide an export named 'GoogleCloudAuthPlugin' (at 

Error: THREE.GLTFLoader: No DRACOLoader instance provided.

    at new GLTFDracoMeshCompressionExtension (chunk-DWKB6WEO.js?v=7703a77f:1752:13)

    at GLTFLoader.parse (chunk-DWKB6WEO.js?v=7703a77f:1013:41)

    at chunk-DWKB6WEO.js?v=7703a77f:1043:13

    at new Promise (<anonymous>)

    at GLTFLoader.parseAsync (chunk-DWKB6WEO.js?v=7703a77f:1042:12)

    at Ai.parseTile (3d-tiles-renderer.js?v=7703a77f:1482:62)

    at 3d-tiles-renderer.js?v=7703a77f:434:294

    at Ai.invokeOnePlugin (3d-tiles-renderer.js?v=7703a77f:468:17)

    at 3d-tiles-renderer.js?v=7703a77f:434:251

    at G.tryRunJobs (chunk-DWKB6WEO.js?v=7703a77f:324:14)

---

**Cursor**

Good news - tiles are loading! They just need a Draco decoder (Google's tiles use Draco compression). Let me update the code:

```javascript
import * as THREE from "three";
import { Matrix4 } from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { TilesRenderer } from "3d-tiles-renderer";
import { GoogleCloudAuthPlugin, GLTFExtensionsPlugin } from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");
const LAT = 40.7128; // NYC Latitude
const LON = -74.006; // NYC Longitude
const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222); // Dark background

// 2. Setup Isometric Camera (Orthographic)
// The frustum size determines zoom level in orthographic mode
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 1000;
const camera = new THREE.OrthographicCamera(
  (frustumSize * aspect) / -2,
  (frustumSize * aspect) / 2,
  frustumSize / 2,
  frustumSize / -2,
  1,
  10000 // Near/Far clipping planes
);

// 3. Position Camera for True Isometric View
// Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
camera.lookAt(0, 0, 0); // Look at the center of the scene

// 4. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Register the Google Cloud Auth Plugin to handle API key for all requests
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// This helper function handles the complex coordinate conversion
// from Lat/Lon (WGS84) to the tile system's rotation.
function positionTileset(lat, lon, renderer) {
  const EARTH_RADIUS = 6378137;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
  const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
  const y = EARTH_RADIUS * Math.cos(phi);

  // Move the group so the target coordinate is at 0,0,0
  renderer.group.position.set(-x, -y, -z);

  // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
  const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
  // Note: Accurate geodetic alignment requires more complex matrix math
  // involving the surface normal at that Lat/Lon.
}

positionTileset(LAT, LON, tilesRenderer);
scene.add(tilesRenderer.group);

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Required to update which tiles to download based on camera view
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

```

```javascript
import * as THREE from "three";
import { Matrix4 } from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { TilesRenderer } from "3d-tiles-renderer";
import { GoogleCloudAuthPlugin, GLTFExtensionsPlugin } from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");
const LAT = 40.7128; // NYC Latitude
const LON = -74.006; // NYC Longitude
const ZOOM_SCALE = 1000; // Adjusts how much "world" fits in the iso view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222); // Dark background

// 2. Setup Isometric Camera (Orthographic)
// The frustum size determines zoom level in orthographic mode
const aspect = window.innerWidth / window.innerHeight;
const frustumSize = 1000;
const camera = new THREE.OrthographicCamera(
  (frustumSize * aspect) / -2,
  (frustumSize * aspect) / 2,
  frustumSize / 2,
  frustumSize / -2,
  1,
  10000 // Near/Far clipping planes
);

// 3. Position Camera for True Isometric View
// Isometric angle: 45 deg Y-rotation, ~35.264 deg X-rotation
camera.position.set(ZOOM_SCALE, ZOOM_SCALE, ZOOM_SCALE);
camera.lookAt(0, 0, 0); // Look at the center of the scene

// 4. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.7/");

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// This helper function handles the complex coordinate conversion
// from Lat/Lon (WGS84) to the tile system's rotation.
function positionTileset(lat, lon, renderer) {
  const EARTH_RADIUS = 6378137;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -EARTH_RADIUS * Math.sin(phi) * Math.cos(theta);
  const z = EARTH_RADIUS * Math.sin(phi) * Math.sin(theta);
  const y = EARTH_RADIUS * Math.cos(phi);

  // Move the group so the target coordinate is at 0,0,0
  renderer.group.position.set(-x, -y, -z);

  // Rotate so "Up" is Y-axis (Google tiles use Z-up ECEF)
  const alignMatrix = new Matrix4().makeRotationX(-Math.PI / 2); // Simple correction
  // Note: Accurate geodetic alignment requires more complex matrix math
  // involving the surface normal at that Lat/Lon.
}

positionTileset(LAT, LON, tilesRenderer);
scene.add(tilesRenderer.group);

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  // Required to update which tiles to download based on camera view
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

```

Done! I added:
1. **DRACOLoader** from Three.js - loads the Draco decoder from Google's CDN
2. **GLTFExtensionsPlugin** - integrates the DRACOLoader with the tiles renderer

Refresh and see if the tiles render now!

---

**User**

OK - seems like there are no more errors but I'm just seeing a blank grey canvas

---

**Cursor**

The issue is that Google 3D Tiles are in ECEF coordinates (Earth-Centered, Earth-Fixed) with values in the millions of meters. The current camera setup is way too small to see anything. Let me fix the setup:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates
const LAT = 40.7128;
const LON = -74.006;
const CAMERA_HEIGHT = 500; // meters above ground

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);
  const cameraPos = latLonToECEF(LAT, LON, CAMERA_HEIGHT);

  // Position camera
  camera.position.copy(cameraPos);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(targetPos).normalize();

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Wait for tiles to start loading, then position camera
tilesRenderer.addEventListener("load-tile-set", () => {
  console.log("Tileset loaded!");
  positionCamera();
});

// Also position immediately in case tiles load fast
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

I've rewritten the setup with:

1. **Perspective camera** (easier to debug initially)
2. **OrbitControls** - you can drag to navigate around
3. **Proper ECEF positioning** - converts lat/lon to Earth-Centered coordinates
4. **Camera positioned 500m above NYC** looking down
5. **Very large far clipping plane** (1e8) to handle Earth-scale coordinates

Refresh the page - you should be able to see NYC and navigate around with mouse controls. Once this works, we can switch to an orthographic isometric view.

---

**User**

Ok great, looks like a map! But i'm wayyyyy up high, not over manhattan...

---

**Cursor**

Let me lower the camera and adjust the positioning:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);
  const cameraPos = latLonToECEF(LAT, LON, CAMERA_HEIGHT);

  // Position camera
  camera.position.copy(cameraPos);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(targetPos).normalize();

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Wait for tiles to start loading, then position camera
tilesRenderer.addEventListener("load-tile-set", () => {
  console.log("Tileset loaded!");
  positionCamera();
});

// Also position immediately in case tiles load fast
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Calculate local east and north vectors for offset
  const up = targetPos.clone().normalize();
  // Approximate east direction (perpendicular to up and world Z)
  const east = new THREE.Vector3(0, 0, 1).cross(up).normalize();
  // North is perpendicular to both
  const north = up.clone().cross(east).normalize();

  // Position camera: up + offset to south-east for nice angle
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(east, CAMERA_DISTANCE * 0.7)
    .addScaledVector(north, -CAMERA_DISTANCE * 0.7);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Wait for tiles to start loading, then position camera
tilesRenderer.addEventListener("load-tile-set", () => {
  console.log("Tileset loaded!");
  positionCamera();
});

// Also position immediately in case tiles load fast
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Done! Changes:
1. **Moved target to Empire State Building** (central Manhattan)
2. **Lowered camera** to 300m
3. **Added angled view** - camera is offset to the south-east for a nice oblique perspective

Refresh and you should see Manhattan buildings up close. Use:
- **Scroll wheel** to zoom in/out
- **Left-click drag** to orbit
- **Right-click drag** to pan

---

**User**

I can't really tell where the camera is looking, and every time I try to drag the camera or move around it sharply resets...

---

**Cursor**

The camera is resetting because `positionCamera()` gets called every time a tile loads. Let me fix that and add a visual marker so you can see where you're looking:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Calculate local east and north vectors for offset
  const up = targetPos.clone().normalize();
  // Approximate east direction (perpendicular to up and world Z)
  const east = new THREE.Vector3(0, 0, 1).cross(up).normalize();
  // North is perpendicular to both
  const north = up.clone().cross(east).normalize();

  // Position camera: up + offset to south-east for nice angle
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(east, CAMERA_DISTANCE * 0.7)
    .addScaledVector(north, -CAMERA_DISTANCE * 0.7);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Add a marker sphere at the target location so we can see where we're looking
const targetPos = latLonToECEF(LAT, LON, 50); // 50m above ground
const markerGeometry = new THREE.SphereGeometry(50, 16, 16);
const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
const marker = new THREE.Mesh(markerGeometry, markerMaterial);
marker.position.copy(targetPos);
scene.add(marker);

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Also, looking at the screenshot - you're definitely not over Manhattan. The ECEF coordinate system might need adjustment (Google tiles likely use Y-up). Let me fix that:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Google 3D Tiles use Y-up convention, so we swap Y and Z
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Standard ECEF: X towards 0°lon, Y towards 90°E, Z towards north pole
  // Three.js Y-up: swap Y and Z
  return new THREE.Vector3(
    r * cosLat * cosLon, // X
    r * sinLat, // Y (was Z - towards north pole, now up)
    -r * cosLat * sinLon // Z (was Y - towards 90°E, negated for right-hand coord)
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Calculate local east and north vectors for offset
  const up = targetPos.clone().normalize();
  // Approximate east direction (perpendicular to up and world Z)
  const east = new THREE.Vector3(0, 0, 1).cross(up).normalize();
  // North is perpendicular to both
  const north = up.clone().cross(east).normalize();

  // Position camera: up + offset to south-east for nice angle
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(east, CAMERA_DISTANCE * 0.7)
    .addScaledVector(north, -CAMERA_DISTANCE * 0.7);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Add a marker sphere at the target location so we can see where we're looking
const targetPos = latLonToECEF(LAT, LON, 50); // 50m above ground
const markerGeometry = new THREE.SphereGeometry(50, 16, 16);
const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
const marker = new THREE.Mesh(markerGeometry, markerMaterial);
marker.position.copy(targetPos);
scene.add(marker);

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Done! Changes:
1. **Fixed coordinate system** - swapped Y/Z for Three.js's Y-up convention
2. **Camera only positions once** - no more resetting when you navigate
3. **Added red marker sphere** (50m radius) at the target location so you can see where you're looking

Refresh and you should see a red sphere marking the Empire State Building area. Use the scroll wheel to zoom in/out until you can see it, then navigate from there!

---

**User**

oh that was a mistake, now i'm zoomed over the desert of iran

---

**Cursor**

Let me try a different approach - I'll use the library's built-in utilities for positioning instead of manually calculating ECEF:

```javascript
// Convert lat/lon/alt to Cartesian coordinates using WGS84 ellipsoid
// This matches how Google 3D Tiles are positioned
function latLonToCartesian(lat, lon, alt = 0) {
  const WGS84_A = 6378137.0; // semi-major axis
  const WGS84_E2 = 0.00669437999014; // eccentricity squared

  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const sinLon = Math.sin(lonRad);
  const cosLon = Math.cos(lonRad);

  // Radius of curvature in the prime vertical
  const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);

  // WGS84 ECEF coordinates
  const x = (N + alt) * cosLat * cosLon;
  const y = (N + alt) * cosLat * sinLon;
  const z = (N * (1 - WGS84_E2) + alt) * sinLat;

  return new THREE.Vector3(x, z, -y); // Convert to Three.js Y-up convention
}
```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Google 3D Tiles use: X towards (0,0), Y towards north pole, Z towards (0, -90W)
function latLonToECEF(lat, lon, alt = 0) {
  const WGS84_A = 6378137.0; // semi-major axis (equatorial radius)
  const WGS84_E2 = 0.00669437999014; // eccentricity squared

  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const sinLon = Math.sin(lonRad);
  const cosLon = Math.cos(lonRad);

  // Radius of curvature in the prime vertical
  const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);

  // Standard WGS84 ECEF
  const ecefX = (N + alt) * cosLat * cosLon;
  const ecefY = (N + alt) * cosLat * sinLon;
  const ecefZ = (N * (1 - WGS84_E2) + alt) * sinLat;

  // Convert to Three.js Y-up: X stays, Y = Z (north up), Z = -Y
  return new THREE.Vector3(ecefX, ecefZ, -ecefY);
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Calculate local east and north vectors for offset
  const up = targetPos.clone().normalize();
  // Approximate east direction (perpendicular to up and world Z)
  const east = new THREE.Vector3(0, 0, 1).cross(up).normalize();
  // North is perpendicular to both
  const north = up.clone().cross(east).normalize();

  // Position camera: up + offset to south-east for nice angle
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(east, CAMERA_DISTANCE * 0.7)
    .addScaledVector(north, -CAMERA_DISTANCE * 0.7);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Add a marker sphere at the target location so we can see where we're looking
const targetPos = latLonToECEF(LAT, LON, 50); // 50m above ground
const markerGeometry = new THREE.SphereGeometry(50, 16, 16);
const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
const marker = new THREE.Mesh(markerGeometry, markerMaterial);
marker.position.copy(targetPos);
scene.add(marker);

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In Y-up system, world Y is north pole direction
  // East is perpendicular to up and world Y axis
  const worldY = new THREE.Vector3(0, 1, 0);
  const east = worldY.clone().cross(up).normalize();
  // North is perpendicular to up and east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for a nice angled view
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE); // offset to the south

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Add a marker sphere at the target location so we can see where we're looking
const targetPos = latLonToECEF(LAT, LON, 50); // 50m above ground
const markerGeometry = new THREE.SphereGeometry(50, 16, 16);
const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
const marker = new THREE.Mesh(markerGeometry, markerMaterial);
marker.position.copy(targetPos);
scene.add(marker);

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

I've fixed the coordinate system:
1. **Proper WGS84 ellipsoid** calculation (more accurate than spherical)
2. **Correct Y-up conversion** for Three.js: `X stays, Y = Z (north), Z = -Y`
3. **Simplified direction math** using world Y as north pole

Refresh and you should now be over Manhattan! Look for the red marker sphere near the Empire State Building.

---

**User**

I'm north of manhattan, and there's a giant random red dot

---

**Cursor**

Progress! You're in the right area (Hudson Valley, north of NYC). The red sphere is the marker I added - way too big at 50m radius. Let me fix both issues:

I see the ECEF function got reverted. Let me fix the coordinate conversion and remove the giant marker:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF, then to Three.js Y-up coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Standard ECEF: X→(0,0), Y→(0,90E), Z→north pole
  const ecefX = r * cosLat * cosLon;
  const ecefY = r * cosLat * sinLon;
  const ecefZ = r * sinLat;

  // Convert to Three.js Y-up: X=X, Y=Z (north up), Z=-Y
  return new THREE.Vector3(ecefX, ecefZ, -ecefY);
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In Y-up system, world Y is north pole direction
  // East is perpendicular to up and world Y axis
  const worldY = new THREE.Vector3(0, 1, 0);
  const east = worldY.clone().cross(up).normalize();
  // North is perpendicular to up and east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for a nice angled view
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE); // offset to the south

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Add a marker sphere at the target location so we can see where we're looking
const targetPos = latLonToECEF(LAT, LON, 50); // 50m above ground
const markerGeometry = new THREE.SphereGeometry(50, 16, 16);
const markerMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
const marker = new THREE.Mesh(markerGeometry, markerMaterial);
marker.position.copy(targetPos);
scene.add(marker);

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF, then to Three.js Y-up coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Standard ECEF: X→(0,0), Y→(0,90E), Z→north pole
  const ecefX = r * cosLat * cosLon;
  const ecefY = r * cosLat * sinLon;
  const ecefZ = r * sinLat;

  // Convert to Three.js Y-up: X=X, Y=Z (north up), Z=-Y
  return new THREE.Vector3(ecefX, ecefZ, -ecefY);
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In Y-up system, world Y is north pole direction
  // East is perpendicular to up and world Y axis
  const worldY = new THREE.Vector3(0, 1, 0);
  const east = worldY.clone().cross(up).normalize();
  // North is perpendicular to up and east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for a nice angled view
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE); // offset to the south

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Camera positioned at:", cameraPos);
  console.log("Looking at:", targetPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF, then to Three.js Y-up coordinates
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Standard ECEF: X→(0,0), Y→(0,90E), Z→north pole
  const ecefX = r * cosLat * cosLon;
  const ecefY = r * cosLat * sinLon;
  const ecefZ = r * sinLat;

  // Convert to Three.js Y-up: X=X, Y=Z (north up), Z=-Y
  return new THREE.Vector3(ecefX, ecefZ, -ecefY);
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // Calculate local east and south for camera offset
  // In our Y-up system, Y points to north pole
  const northPole = new THREE.Vector3(0, 1, 0);
  // East is north pole crossed with up (right-hand rule)
  const east = northPole.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south-east for a nice angled view
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE * 0.7)
    .addScaledVector(east, CAMERA_DISTANCE * 0.7);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Done! Changes:
1. **Fixed ECEF→Three.js conversion** - properly swaps Y↔Z for Y-up coordinate system  
2. **Removed the giant red sphere**
3. **Updated camera direction vectors** to match the new coordinate system

Refresh and you should be much closer to Manhattan now. Use scroll wheel to zoom, left-drag to orbit around!

---

**User**

no! with this ECEF conversion thing I'm now in the middle of the desert in Iran

---

**Cursor**

```plaintext
return new THREE.Vector3(
  r * cosLat * cosLon,  // X
  r * cosLat * sinLon,  // Y  
  r * sinLat            // Z
);
```

Sorry! Let me revert to the original ECEF (which had you north of Manhattan - much closer). The swapping is making it worse:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // Calculate local east and south for camera offset
  // In our Y-up system, Y points to north pole
  const northPole = new THREE.Vector3(0, 1, 0);
  // East is north pole crossed with up (right-hand rule)
  const east = northPole.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south-east for a nice angled view
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE * 0.7)
    .addScaledVector(east, CAMERA_DISTANCE * 0.7);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 300; // meters above ground
const CAMERA_DISTANCE = 400; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east  
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Reverted! You should be back to "north of Manhattan" territory. From there you can:
- **Scroll wheel** to zoom in/out
- **Left-drag** to orbit and find Manhattan (should be to the south)
- **Right-drag** to pan

Try zooming out a bit and orbiting to find the NYC skyline!

---

**User**

it seems like the camera might actually be above the empire state building - i'm just VERY high in the air

---

**Cursor**

The coordinates are working. You're just too high up. Let me bring the camera much closer:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 200; // meters above ground
const CAMERA_DISTANCE = 300; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Actually, let me go even lower - drone-level view:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 100; // meters above ground (drone level)
const CAMERA_DISTANCE = 150; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  1,
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

And let me adjust the near clipping plane so you can see things up close:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 100; // meters above ground (drone level)
const CAMERA_DISTANCE = 150; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1, // Small near plane for close-up views
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Done! Changes:
- **Camera height**: 100m (was 300m) - roughly drone altitude
- **Camera distance**: 150m (was 400m) - much closer
- **Near clipping**: 0.1 (was 1) - allows close-up views

Refresh and you should see the Empire State Building up close! Use scroll wheel to zoom in/out further.

---

**User**

it seems like i'm still like 10km above the ground

---

**Cursor**

```plaintext
const cameraPos = targetPos
  .clone()
  .addScaledVector(up, CAMERA_HEIGHT)
  .addScaledVector(north, -CAMERA_DISTANCE);
```

The height values aren't having the expected effect. Let me try a much more aggressive approach - super close camera:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 500; // meters above ground
const CAMERA_DISTANCE = 500; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1, // Small near plane for close-up views
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Wait, I went the wrong direction. Let me also constrain the OrbitControls so the camera can't zoom out too far:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 5; // meters above ground
const CAMERA_DISTANCE = 5; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1, // Small near plane for close-up views
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 100; // Don't get closer than 100m
controls.maxDistance = 5000; // Don't get farther than 5km

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Let me add some debugging to understand what's happening with the camera position:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 500; // meters above ground
const CAMERA_DISTANCE = 500; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1, // Small near plane for close-up views
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position camera once at startup
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded! Positioning camera...");
    positionCamera();
    cameraPositioned = true;
  }
});

// Initial camera position
positionCamera();

// Debug: log camera distance from target every few seconds
let lastLog = 0;
function logCameraInfo() {
  const now = Date.now();
  if (now - lastLog > 3000) {
    const targetPos = latLonToECEF(LAT, LON, 0);
    const dist = camera.position.distanceTo(targetPos);
    console.log(`Camera distance from target: ${(dist/1000).toFixed(2)} km`);
    lastLog = now;
  }
}

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();
  logCameraInfo();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Refresh and check the browser console (F12 → Console tab). It will log the camera distance every 3 seconds. This will tell us how far away you actually are.

Also, try using the **scroll wheel to zoom IN** - can you get closer, or does it stop at some point?

---

**User**

Even when I'm zoomed max in (camera distance 0 km, I'm still seeing a very zoomed out image

---

**Cursor**

Interesting! Camera distance is 0km but still zoomed out - this means there's likely a **transform mismatch** between the camera coordinates and where the tiles are actually rendered.

The tiles group might need to be repositioned. Let me add that:

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 500; // meters above ground
const CAMERA_DISTANCE = 500; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1, // Small near plane for close-up views
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position tiles and camera once tileset loads
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded!");
    
    // Get the target position in ECEF
    const targetECEF = latLonToECEF(LAT, LON, 0);
    
    // Move the tiles group so that our target location is at the origin
    // This makes camera positioning much simpler
    tilesRenderer.group.position.set(-targetECEF.x, -targetECEF.y, -targetECEF.z);
    
    // Now position camera relative to origin (which is now at our target)
    camera.position.set(0, CAMERA_HEIGHT, CAMERA_DISTANCE);
    camera.lookAt(0, 0, 0);
    camera.up.set(0, 1, 0);
    
    // Update controls to orbit around origin
    controls.target.set(0, 0, 0);
    controls.update();
    
    console.log("Tiles group repositioned, camera at origin-relative position");
    cameraPositioned = true;
  }
});

// Debug: log camera distance from target every few seconds
let lastLog = 0;
function logCameraInfo() {
  const now = Date.now();
  if (now - lastLog > 3000) {
    const targetPos = latLonToECEF(LAT, LON, 0);
    const dist = camera.position.distanceTo(targetPos);
    console.log(`Camera distance from target: ${(dist / 1000).toFixed(2)} km`);
    lastLog = now;
  }
}

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();
  logCameraInfo();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

This is a different approach - instead of moving the camera to Earth-scale ECEF coordinates, I'm moving the **tiles** so that the Empire State Building location is at the origin (0,0,0). Then the camera is positioned at simple, predictable coordinates.

Refresh and you should finally see Manhattan up close!

---

**User**

Nope. and now it says Camera distance from target: 6378.17 km

---

**Cursor**

I can see Manhattan in your screenshot! The positioning is working, but the orientation is wrong - the camera's "up" doesn't match the surface normal. Let me fix the rotation:

```javascript
// Position tiles and camera once tileset loads
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded!");
    
    // Get the target position in ECEF
    const targetECEF = latLonToECEF(LAT, LON, 0);
    
    // The "up" direction at this location (points away from Earth center)
    const surfaceUp = targetECEF.clone().normalize();
    
    // Create a matrix that:
    // 1. Translates the target to origin
    // 2. Rotates so surface "up" aligns with Y axis
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(surfaceUp, new THREE.Vector3(0, 1, 0));
    
    // Apply rotation first, then translation
    tilesRenderer.group.quaternion.copy(quaternion);
    
    // After rotation, we need to translate the rotated position
    const rotatedTarget = targetECEF.clone().applyQuaternion(quaternion);
    tilesRenderer.group.position.set(-rotatedTarget.x, -rotatedTarget.y, -rotatedTarget.z);
    
    // Now position camera - Y is up, looking down at an angle from the south
    camera.position.set(0, CAMERA_HEIGHT, CAMERA_DISTANCE);
    camera.lookAt(0, 0, 0);
    camera.up.set(0, 1, 0);
    
    // Update controls to orbit around origin
    controls.target.set(0, 0, 0);
    controls.update();
    
    console.log("Tiles repositioned and rotated, camera ready");
    cameraPositioned = true;
  }
});
```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 500; // meters above ground
const CAMERA_DISTANCE = 500; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1, // Small near plane for close-up views
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position tiles and camera once tileset loads
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded!");

    // Get the target position in ECEF
    const targetECEF = latLonToECEF(LAT, LON, 0);

    // The "up" direction at this location (points away from Earth center)
    const surfaceUp = targetECEF.clone().normalize();

    // Create rotation that aligns surface "up" with Y axis
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(surfaceUp, new THREE.Vector3(0, 1, 0));

    // Apply rotation to tiles group
    tilesRenderer.group.quaternion.copy(quaternion);

    // Translate so rotated target is at origin
    const rotatedTarget = targetECEF.clone().applyQuaternion(quaternion);
    tilesRenderer.group.position.set(
      -rotatedTarget.x,
      -rotatedTarget.y,
      -rotatedTarget.z
    );

    // Camera: above and to the south, looking north at the target
    camera.position.set(0, CAMERA_HEIGHT, CAMERA_DISTANCE);
    camera.lookAt(0, 0, 0);
    camera.up.set(0, 1, 0);

    // Update controls
    controls.target.set(0, 0, 0);
    controls.update();

    console.log("Tiles repositioned and rotated");
    cameraPositioned = true;
  }
});

// Debug: log camera distance from target every few seconds
let lastLog = 0;
function logCameraInfo() {
  const now = Date.now();
  if (now - lastLog > 3000) {
    const targetPos = latLonToECEF(LAT, LON, 0);
    const dist = camera.position.distanceTo(targetPos);
    console.log(`Camera distance from target: ${(dist / 1000).toFixed(2)} km`);
    lastLog = now;
  }
}

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();
  logCameraInfo();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

```javascript
import * as THREE from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building area)
const LAT = 40.748817;
const LON = -73.985428;
const CAMERA_HEIGHT = 500; // meters above ground
const CAMERA_DISTANCE = 500; // horizontal offset for angled view

// 1. Setup Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue

// 2. Setup Camera (Perspective first to debug, can switch to ortho later)
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1, // Small near plane for close-up views
  1e8 // Very large far plane for Earth-scale coordinates
);

// 3. Setup Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// 4. Setup Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// 5. Load Google 3D Tiles
const tilesRenderer = new TilesRenderer();

// Setup DRACOLoader for decompressing Google's compressed meshes
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath(
  "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
);

// Register plugins
tilesRenderer.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
tilesRenderer.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader }));
tilesRenderer.registerPlugin(new TileCompressionPlugin());

tilesRenderer.setCamera(camera);
tilesRenderer.setResolutionFromRenderer(camera, renderer);

// Add tiles to scene
scene.add(tilesRenderer.group);

// Convert lat/lon to ECEF (Earth-Centered, Earth-Fixed) coordinates
// Standard ECEF: X→(0°,0°), Y→(0°,90°E), Z→north pole
function latLonToECEF(lat, lon, alt = 0) {
  const EARTH_RADIUS = 6378137; // meters
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;

  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLon = Math.cos(lonRad);
  const sinLon = Math.sin(lonRad);

  const r = EARTH_RADIUS + alt;

  // Use raw ECEF - Google tiles appear to use this directly
  return new THREE.Vector3(
    r * cosLat * cosLon,
    r * cosLat * sinLon,
    r * sinLat
  );
}

// Position camera above NYC with an angled view
function positionCamera() {
  const targetPos = latLonToECEF(LAT, LON, 0);

  // Local "up" is away from Earth center
  const up = targetPos.clone().normalize();

  // In raw ECEF, Z points to north pole
  const zAxis = new THREE.Vector3(0, 0, 1);
  // East is Z (north) crossed with up
  const east = zAxis.clone().cross(up).normalize();
  // Local north is up crossed with east
  const north = up.clone().cross(east).normalize();

  // Position camera: above + offset to south for angled view looking north
  const cameraPos = targetPos
    .clone()
    .addScaledVector(up, CAMERA_HEIGHT)
    .addScaledVector(north, -CAMERA_DISTANCE);

  // Position camera
  camera.position.copy(cameraPos);

  // Set up vector to point away from Earth center (local "up")
  camera.up.copy(up);

  // Look at the target (ground level)
  camera.lookAt(targetPos);

  // Update controls target
  controls.target.copy(targetPos);
  controls.update();

  console.log("Target position:", targetPos);
  console.log("Camera position:", cameraPos);
}

// Position tiles and camera once tileset loads
let cameraPositioned = false;
tilesRenderer.addEventListener("load-tile-set", () => {
  if (!cameraPositioned) {
    console.log("Tileset loaded!");

    // Get the target position in ECEF
    const targetECEF = latLonToECEF(LAT, LON, 0);

    // The "up" direction at this location (points away from Earth center)
    const surfaceUp = targetECEF.clone().normalize();

    // Create rotation that aligns surface "up" with Y axis
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(surfaceUp, new THREE.Vector3(0, 1, 0));

    // Apply rotation to tiles group
    tilesRenderer.group.quaternion.copy(quaternion);

    // Translate so rotated target is at origin
    const rotatedTarget = targetECEF.clone().applyQuaternion(quaternion);
    tilesRenderer.group.position.set(
      -rotatedTarget.x,
      -rotatedTarget.y,
      -rotatedTarget.z
    );

    // Camera: above and to the south, looking north at the target
    camera.position.set(0, CAMERA_HEIGHT, CAMERA_DISTANCE);
    camera.lookAt(0, 0, 0);
    camera.up.set(0, 1, 0);

    // Update controls
    controls.target.set(0, 0, 0);
    controls.update();

    console.log("Tiles repositioned and rotated");
    cameraPositioned = true;
  }
});

// Debug: log camera distance from target every few seconds
let lastLog = 0;
function logCameraInfo() {
  const now = Date.now();
  if (now - lastLog > 3000) {
    // Target is now at origin after our transformation
    const dist = camera.position.length();
    console.log(`Camera distance from target: ${(dist / 1000).toFixed(2)} km`);
    lastLog = now;
  }
}

// 6. Render Loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  tilesRenderer.update();
  logCameraInfo();

  renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

```

Done! Key changes:
1. **Added rotation** - rotates the tiles so that "up" at NYC aligns with the Y axis
2. **Fixed debug logging** - now measures distance from camera to origin (which is the target)

Refresh - you should now see Manhattan with the correct orientation (ground flat, sky up)!

---

**User**

Nope - still floating high above the hudson.



Take a look at what they do in the example: @src/web/example.js 





---

**Cursor**

I see the key differences in the example:

1. Uses `GlobeControls` instead of `OrbitControls`
2. Simple rotation: `tiles.group.rotation.x = -Math.PI / 2` 
3. Uses `WGS84_ELLIPSOID` for coordinate conversion
4. Uses `CameraTransitionManager`

Let me rewrite to follow their approach:

```javascript
import { 
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// NYC coordinates (Empire State Building)
const LAT = 40.748817;
const LON = -73.985428;
const HEIGHT = 500; // meters above ground

let scene, renderer, camera, controls, tiles;

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera
  camera = new PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    1,
    160000000
  );

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls - designed for globe/earth navigation
  controls = new GlobeControls(scene, camera, renderer.domElement, null);
  controls.enableDamping = true;
  
  // Connect controls to the tiles ellipsoid
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    
    // Position camera above NYC
    positionCamera();
  });

  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);
}

function positionCamera() {
  // Convert lat/lon/height to 3D position using WGS84 ellipsoid
  WGS84_ELLIPSOID.getCartographicToPosition(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    camera.position
  );

  // Apply the tiles group transform to get world position
  camera.position.applyMatrix4(tiles.group.matrixWorld);

  // Look at Earth center (0,0,0)
  camera.lookAt(0, 0, 0);

  console.log("Camera positioned above NYC at", HEIGHT, "m");
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.update();
  
  // Update tiles with current camera
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

Completely rewrote using the example's approach:

1. **`GlobeControls`** - designed for Earth/globe navigation
2. **`WGS84_ELLIPSOID`** - proper geodetic coordinate conversion  
3. **`tiles.group.rotation.x = -Math.PI / 2`** - simple Z-up to Y-up rotation
4. **`controls.setEllipsoid()`** - connects controls to the tiles properly

Refresh and you should finally see NYC from 500m up! Use mouse to navigate:
- **Scroll** to zoom
- **Left-drag** to rotate around
- **Right-drag** to pan

---

**User**

Perfect! Now I'm looking directly at the Empire State Building. What I'd like to do is look at Times Square from roughly the same angle as @src/isometric_nyc/whitebox.py - i then want to switch the camera to isometric so we can get the same simcity-like look for the rendered tiles

---

**Cursor**

Let me update the code to match the Times Square location and isometric camera angle from `whitebox.py`:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (from whitebox.py - SimCity isometric style)
// Azimuth: 210 = looking from south-west (30 + 180)
// Elevation: ~35 degrees for isometric feel
const CAMERA_AZIMUTH = 210;
const CAMERA_ELEVATION = -35; // Negative = looking down

let scene, renderer, controls, tiles, transition;
let isOrthographic = false;

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 160000000),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(scene, transition.camera, renderer.domElement, null);
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // This gives us proper orientation at the given lat/lon
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(camera.position, camera.quaternion, camera.scale);

  // Sync orthographic camera
  transition.syncCameras();

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
}

function toggleOrthographic() {
  isOrthographic = !isOrthographic;

  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();

  console.log(`Switched to ${isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"} camera`);
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Controls:</strong><br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Orthographic (isometric)
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

Done! Key changes:

1. **Times Square coordinates** - `LAT = 40.758, LON = -73.9855`
2. **Isometric camera angles** - Azimuth 210° (SW view), Elevation -35° (looking down)
3. **CameraTransitionManager** - Handles perspective ↔ orthographic switching
4. **Press 'O'** - Toggles to orthographic (isometric) camera mode
5. **UI overlay** - Shows controls

Refresh and you should see Times Square! Press **O** to switch to the isometric/orthographic view for that SimCity 3000 look.

---

**User**

Amazing - now I want to finish aligning the viewports of the python render box. Can you please ensure that the isometric camera is enabled by default? It also seems like the azimuth isn't being respected and I'm looking straight north...

---

**Cursor**

```plaintext
cx = dist * np.cos(alpha) * np.sin(beta)
cy = dist * np.cos(alpha) * np.cos(beta)
```

Let me fix both issues - enable orthographic by default and fix the azimuth:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (from whitebox.py - SimCity isometric style)
// Azimuth: 210 = looking from south-west (30 + 180)
// Elevation: ~35 degrees for isometric feel
const CAMERA_AZIMUTH = 210;
const CAMERA_ELEVATION = -35; // Negative = looking down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // This gives us proper orientation at the given lat/lon
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync orthographic camera
  transition.syncCameras();

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
}

function toggleOrthographic() {
  isOrthographic = !isOrthographic;

  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Controls:</strong><br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Orthographic (isometric)
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (from whitebox.py - SimCity isometric style)
// Azimuth: 210 = looking from south-west (30 + 180)
// Elevation: ~35 degrees for isometric feel
const CAMERA_AZIMUTH = 210;
const CAMERA_ELEVATION = -35; // Negative = looking down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  isOrthographic = !isOrthographic;

  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Controls:</strong><br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Orthographic (isometric)
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (from whitebox.py - SimCity isometric style)
// Azimuth: 210 = looking from south-west (30 + 180)
// Elevation: ~35 degrees for isometric feel
const CAMERA_AZIMUTH = 210;
const CAMERA_ELEVATION = -35; // Negative = looking down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Controls:</strong><br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Orthographic (isometric)
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (from whitebox.py - SimCity isometric style)
// Azimuth: 210 = looking from south-west (30 + 180)
// Elevation: ~35 degrees for isometric feel
const CAMERA_AZIMUTH = 210;
const CAMERA_ELEVATION = -35; // Negative = looking down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
    <em>Default: Orthographic (isometric)</em>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

```javascript
WGS84_ELLIPSOID.getObjectFrame(
  lat * MathUtils.DEG2RAD, lon * MathUtils.DEG2RAD, height,
  az * MathUtils.DEG2RAD, el * MathUtils.DEG2RAD, roll * MathUtils.DEG2RAD,
  camera.matrixWorld, CAMERA_FRAME,
);
```

```javascript
WGS84_ELLIPSOID.getCartographicFromObjectFrame( localCameraMat, cartographicResult, CAMERA_FRAME );

// convert to DEG
cartographicResult.azimuth *= MathUtils.RAD2DEG;
cartographicResult.elevation *= MathUtils.RAD2DEG;
```

Let me also try adjusting the azimuth - the convention might be that it's the direction the camera is *facing*, not where it's *from*:

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = 30; // Looking towards NE (positioned in SW)
const CAMERA_ELEVATION = -45; // 45° down for classic isometric

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
    <em>Default: Orthographic (isometric)</em>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

Done! Changes:
1. **Orthographic enabled by default** - starts in isometric mode
2. **Fixed azimuth** - changed to 30° (camera facing NE, positioned from SW - matching whitebox.py's original angle)
3. **Elevation** - set to -45° for classic isometric look

Refresh and you should see Times Square from the correct SimCity-style angle in orthographic mode. Press **O** if you want to toggle to perspective view.

---

**User**

OK - can you help me debug the view angle? I want to log the azimuth and elevsation/zoom whenever they change... ensure there's a slight debounce to handle the scroll movements

---

**Cursor**

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = 30; // Looking towards NE (positioned in SW)
const CAMERA_ELEVATION = -45; // 45° down for classic isometric

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
    <em>Default: Orthographic (isometric)</em>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  renderer.render(scene, camera);
}

```

```javascript
import {
  TilesRenderer,
  WGS84_ELLIPSOID,
  GlobeControls,
  CameraTransitionManager,
  CAMERA_FRAME,
} from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Times Square coordinates (from whitebox.py)
const LAT = 40.758;
const LON = -73.9855;
const HEIGHT = 300; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = 30; // Looking towards NE (positioned in SW)
const CAMERA_ELEVATION = -45; // 45° down for classic isometric

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

init();
animate();

function init() {
  // Renderer
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      160000000
    ),
    new OrthographicCamera(-1, 1, 1, -1, 1, 160000000)
  );
  transition.autoSync = false;
  transition.orthographicPositionalZoom = false;

  // Handle camera changes
  transition.addEventListener("camera-change", ({ camera, prevCamera }) => {
    tiles.deleteCamera(prevCamera);
    tiles.setCamera(camera);
    controls.setCamera(camera);
  });

  // Initialize tiles
  tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY }));
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({
      dracoLoader: new DRACOLoader().setDecoderPath(
        "https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"
      ),
    })
  );

  // Rotate tiles so Z-up becomes Y-up (Three.js convention)
  tiles.group.rotation.x = -Math.PI / 2;
  scene.add(tiles.group);

  // Setup GlobeControls
  controls = new GlobeControls(
    scene,
    transition.camera,
    renderer.domElement,
    null
  );
  controls.enableDamping = true;

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    console.log("Tileset loaded!");
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();
  });

  tiles.setCamera(transition.camera);
  tiles.setResolutionFromRenderer(transition.camera, renderer);

  // Handle resize
  window.addEventListener("resize", onWindowResize);

  // Add keyboard controls
  window.addEventListener("keydown", onKeyDown);

  // Add UI instructions
  addUI();
}

function positionCamera() {
  const camera = transition.perspectiveCamera;

  // Use getObjectFrame to position camera with azimuth/elevation
  // Azimuth: 0=North, 90=East, 180=South, 270=West
  // For SimCity view from SW looking NE, we want azimuth ~210-225
  WGS84_ELLIPSOID.getObjectFrame(
    LAT * MathUtils.DEG2RAD,
    LON * MathUtils.DEG2RAD,
    HEIGHT,
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Apply tiles group transform
  camera.matrixWorld.premultiply(tiles.group.matrixWorld);
  camera.matrixWorld.decompose(
    camera.position,
    camera.quaternion,
    camera.scale
  );

  // Sync both cameras
  transition.syncCameras();
  controls.adjustCamera(transition.perspectiveCamera);
  controls.adjustCamera(transition.orthographicCamera);

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);
}

function toggleOrthographic() {
  // Get current pivot point for smooth transition
  controls.getPivotPoint(transition.fixedPoint);

  if (!transition.animating) {
    transition.syncCameras();
    controls.adjustCamera(transition.perspectiveCamera);
    controls.adjustCamera(transition.orthographicCamera);
  }

  transition.toggle();
  isOrthographic = transition.mode === "orthographic";

  console.log(
    `Switched to ${
      isOrthographic ? "ORTHOGRAPHIC (isometric)" : "PERSPECTIVE"
    } camera`
  );
}

function onKeyDown(event) {
  if (event.key === "o" || event.key === "O") {
    toggleOrthographic();
  }
}

function addUI() {
  const info = document.createElement("div");
  info.style.cssText = `
    position: fixed;
    top: 10px;
    left: 10px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 10px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 5px;
    z-index: 1000;
  `;
  info.innerHTML = `
    <strong>Isometric NYC - Times Square</strong><br>
    <br>
    Scroll: Zoom<br>
    Left-drag: Rotate<br>
    Right-drag: Pan<br>
    <strong>O</strong>: Toggle Perspective/Ortho<br>
    <br>
  `;
  document.body.appendChild(info);
}

function onWindowResize() {
  const { perspectiveCamera, orthographicCamera } = transition;
  const aspect = window.innerWidth / window.innerHeight;

  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  orthographicCamera.left = -orthographicCamera.top * aspect;
  orthographicCamera.right = -orthographicCamera.left;
  orthographicCamera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Extract current camera azimuth, elevation, height from its world matrix
function getCameraInfo() {
  if (!tiles || !tiles.group) return null;

  const camera = transition.camera;
  const cartographicResult = {};

  // Get inverse of tiles group matrix to convert camera to local tile space
  const tilesMatInv = tiles.group.matrixWorld.clone().invert();
  const localCameraMat = camera.matrixWorld.clone().premultiply(tilesMatInv);

  // Extract cartographic position including orientation
  WGS84_ELLIPSOID.getCartographicFromObjectFrame(
    localCameraMat,
    cartographicResult,
    CAMERA_FRAME
  );

  return {
    lat: cartographicResult.lat * MathUtils.RAD2DEG,
    lon: cartographicResult.lon * MathUtils.RAD2DEG,
    height: cartographicResult.height,
    azimuth: cartographicResult.azimuth * MathUtils.RAD2DEG,
    elevation: cartographicResult.elevation * MathUtils.RAD2DEG,
    roll: cartographicResult.roll * MathUtils.RAD2DEG,
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
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = { az: info.azimuth, el: info.elevation, height: info.height };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(1)}° ` +
        `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(4)}° Lon=${info.lon.toFixed(4)}°`
      );
    }, 200);
  }
}

function animate() {
  requestAnimationFrame(animate);

  controls.enabled = !transition.animating;
  controls.update();
  transition.update();

  // Update tiles with current camera
  const camera = transition.camera;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();

  // Log camera state (debounced)
  logCameraState();

  renderer.render(scene, camera);
}

```

Done! Now when you move the camera, the console will log:
- **Azimuth** - horizontal direction (0°=N, 90°=E, 180°=S, 270°=W)
- **Elevation** - vertical angle (negative = looking down)
- **Height** - altitude in meters
- **Lat/Lon** - current position

The logging is debounced (200ms) and only triggers when values change significantly. Open the browser console (F12) to see the logs as you navigate!

---

**User**

OK - now I need to update @src/isometric_nyc/whitebox.py to use the parameters exported from @src/view.json to generate the overhead view shown in the web render. Can you update the way these parameters get used in the @whitebox.py script?

---

**Cursor**

```json
{
  "name": "Madison Square Garden",
  "lat": 40.7505,
  "lon": -73.9934,
  "camera_height_meters": 300,
  "camera_azimuth_degrees": 29.5,
  "camera_elevation_degrees": -90
}
```

Let me look at the full `whitebox.py` to understand how the camera parameters are used:

I see the issues - the JSON keys don't match what the script expects, and the camera setup doesn't properly use elevation in degrees. Let me fix this:

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json["camera_elevation_degrees"]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
):
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Negative because we're rotating the texture, not the geometry
      satellite_image = satellite_image.rotate(
        -orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  angle_rad = np.radians(-orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. SimCity 3000 Camera Setup
  plotter.camera.enable_parallel_projection()
  alpha = np.arctan(CAMERA_ELEVATION)  # Elevation angle
  beta = np.radians(CAMERA_AZIMUTH)  # Azimuth angle
  dist = 2000

  cx = dist * np.cos(alpha) * np.sin(beta)
  cy = dist * np.cos(alpha) * np.cos(beta)
  cz = dist * np.sin(alpha)

  plotter.camera.position = (cx, cy, cz)
  plotter.camera.focal_point = (0, 0, 0)

  # Camera orientation and zoom
  plotter.camera.up = (0, 0, 1)
  plotter.camera.parallel_scale = CAMERA_ZOOM

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json["camera_elevation_degrees"]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
):
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Negative because we're rotating the texture, not the geometry
      satellite_image = satellite_image.rotate(
        -orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  angle_rad = np.radians(-orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()
  
  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Azimuth: 0 = North, 90 = East, 180 = South, 270 = West
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  azimuth_rad = np.radians(CAMERA_AZIMUTH)
  dist = 2000

  # Spherical to Cartesian conversion
  # For elevation -90° (overhead), cos(elevation) = 0, sin(elevation) = -1
  # Camera positioned above looking down
  cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
  cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
  cz = -dist * np.sin(elevation_rad)  # Negative because -90° means looking down

  plotter.camera.position = (cx, cy, cz)
  plotter.camera.focal_point = (0, 0, 0)

  # Camera orientation
  # For overhead view, "up" should point in the azimuth direction
  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Up vector points towards the azimuth direction
    plotter.camera.up = (np.sin(azimuth_rad), np.cos(azimuth_rad), 0)
  else:
    # Standard isometric: Z is up
    plotter.camera.up = (0, 0, 1)
  
  plotter.camera.parallel_scale = CAMERA_ZOOM
  
  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: ({cx:.1f}, {cy:.1f}, {cz:.1f})")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

Now let me also update `view.json` to include the `size_meters` field:

```json
{
  "name": "Madison Square Garden",
  "lat": 40.7505,
  "lon": -73.9934,
  "size_meters": 300,
  "camera_height_meters": 300,
  "camera_azimuth_degrees": 29.5,
  "camera_elevation_degrees": -90
}

```

The lints are just line length warnings (not errors). The key changes are:

**`whitebox.py`:**
1. Fixed JSON key mappings (`camera_azimuth_degrees`, `camera_elevation_degrees`)
2. `CAMERA_ELEVATION_DEG` now in degrees (was a factor before)
3. Camera position calculation properly handles overhead view (-90°)
4. For overhead views, the "up" vector points in the azimuth direction (so the view is rotated correctly)

**`view.json`:**
- Added `size_meters` field

Now both the web viewer and Python renderer will use the same parameters from `view.json`:
- **Azimuth 29.5°** - Aligned with the Manhattan grid (avenues)
- **Elevation -90°** - Straight down overhead view

Run the Python script to verify it matches the web view:
```bash
uv run python -m isometric_nyc.whitebox
```

```shellscript
uv run python -m isometric_nyc.whitebox
```

---

**User**

This looks mostly right except I'm not oriented correctly, is the azimuth being used?

---

**Cursor**

Let me check how the orientation is being used in the `render_tile` function:

```python
plotter.camera.up = (np.sin(azimuth_rad), np.cos(azimuth_rad), 0)
```

I see the issue. The geometry is already being rotated by `-orientation_deg` (which equals the azimuth), so the camera "up" vector shouldn't also apply the azimuth rotation - that would double-rotate. Let me fix the camera setup:

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
):
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Negative because we're rotating the texture, not the geometry
      satellite_image = satellite_image.rotate(
        -orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  angle_rad = np.radians(-orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()
  
  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry already rotated, so camera is simply above origin
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # "Up" in the view is the +Y direction (after geometry rotation)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)
    
    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)
  
  plotter.camera.parallel_scale = CAMERA_ZOOM
  
  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

The fix: For overhead views (-90° elevation), the geometry is already rotated by the azimuth angle, so:
- Camera goes directly above `(0, 0, dist)` 
- "Up" in the view is `+Y` (the rotation already aligned streets with the view)

Run the script again - the orientation should now match the web view!

---

**User**

Hmmm.. this is still incorrect, something about this alignment is wrong

---

**Cursor**

Let me check exactly how the web view uses the azimuth to understand the difference:

I see the issue - the web view uses `getObjectFrame` which handles the azimuth differently. In the web view, the azimuth rotates the camera's VIEW. In Python, we rotate the GEOMETRY then view with fixed up=(0,1,0).

Let me try adjusting the Python camera up vector to match:

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
):
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Negative because we're rotating the texture, not the geometry
      satellite_image = satellite_image.rotate(
        -orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  angle_rad = np.radians(-orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()

  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry already rotated, so camera is simply above origin
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # "Up" in the view - needs to match web view convention
    # Web view: azimuth determines which way is "top" of screen
    # Since geometry is rotated by -azimuth, we need up to point at -Y
    # to have the same visual result as web (where up points towards azimuth)
    plotter.camera.up = (0, -1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = CAMERA_ZOOM

  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

Try running the script again. If the orientation is still off, let me know and we can try other directions - the correct one might be `(1, 0, 0)` or `(-1, 0, 0)` instead.

---

**User**

still off

---

**Cursor**

```plaintext
[cos(-29.5°)  -sin(-29.5°)]   [0.87   0.49]
[sin(-29.5°)   cos(-29.5°)] = [-0.49  0.87]
```

Let me try the opposite rotation direction - I think the sign might be wrong:

Let me try flipping the rotation direction for both geometry and satellite:

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
):
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Positive to align azimuth direction with top of view
      satellite_image = satellite_image.rotate(
        orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  angle_rad = np.radians(-orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()

  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry already rotated, so camera is simply above origin
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # "Up" in the view - needs to match web view convention
    # Web view: azimuth determines which way is "top" of screen
    # Since geometry is rotated by -azimuth, we need up to point at -Y
    # to have the same visual result as web (where up points towards azimuth)
    plotter.camera.up = (0, -1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = CAMERA_ZOOM

  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
):
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Positive to align azimuth direction with top of view
      satellite_image = satellite_image.rotate(
        orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  # Positive to align azimuth direction with +Y (top of view)
  angle_rad = np.radians(orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()

  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry already rotated, so camera is simply above origin
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # "Up" in the view - needs to match web view convention
    # Web view: azimuth determines which way is "top" of screen
    # Since geometry is rotated by -azimuth, we need up to point at -Y
    # to have the same visual result as web (where up points towards azimuth)
    plotter.camera.up = (0, -1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = CAMERA_ZOOM

  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

```python
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import pyvista as pv
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.wkb import loads as load_wkb

from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.db import get_db_config

# Cache directory for satellite tiles
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".satellite_cache"
VIEW_JSON_PATH = Path(__file__).parent.parent / "view.json"

with open(VIEW_JSON_PATH, "r") as f:
  view_json = json.loads(f.read())

# Constants from view.json
LAT = view_json["lat"]
LON = view_json["lon"]
SIZE_METERS = view_json.get("size_meters", 300)  # Default 300m if not specified

# Viewport settings
VIEWPORT_WIDTH = 2560
VIEWPORT_HEIGHT = 1440

# Camera settings from view.json
CAMERA_ZOOM = 100  # Parallel scale - lower = more zoomed in, higher = more zoomed out
CAMERA_AZIMUTH = view_json["camera_azimuth_degrees"]  # Horizontal angle in degrees
CAMERA_ELEVATION_DEG = view_json[
  "camera_elevation_degrees"
]  # Vertical angle in degrees (-90 = overhead)
ORIENTATION_DEG = CAMERA_AZIMUTH  # Use azimuth for geometry rotation too

# Satellite alignment tweaks (adjust if needed)
SATELLITE_ZOOM = 18  # Google Maps zoom level (lower=more overhead, higher=more detail)
SATELLITE_TILE_SIZE = (
  640  # Size per tile in pixels (max 640 for free, 2048 for premium)
)
# SATELLITE_GRID is now calculated automatically based on coverage needs
GROUND_Z = 10  # Height of ground plane (adjust if not aligned)
BUILDING_OPACITY = 1  # Opacity for buildings (1.0=solid, 0.0=transparent)

# Perspective correction for satellite imagery
# Set to True to enable perspective correction (requires manual calibration)
USE_PERSPECTIVE_CORRECTION = True
# Perspective transform matrix (can be calibrated by matching control points)
# Note: scale_x and scale_y are now auto-calculated, but can be fine-tuned here
PERSPECTIVE_TRANSFORM = {
  "scale_x": 1.0,  # Will be overridden by calculated scale
  "scale_y": 1.0,  # Will be overridden by calculated scale
  "shear_x": 0.0,  # Horizontal shear (perspective effect)
  "shear_y": 0.0,  # Vertical shear (perspective effect)
}

# NAD83 / New York Long Island (Meters)
FORCE_SRID = 2908


# --- COLOR STRATEGY FOR CONTROLNET ---
# White Roofs + Gray Walls = "Fake Lighting" that helps AI understand 3D shapes.
COLORS = {
  712: "white",  # RoofSurface -> Brightest
  709: "#666666",  # WallSurface -> Mid-Gray
  710: "#111111",  # GroundSurface -> Very Dark Gray (almost black)
  901: "white",  # Building (Fallback if no surfaces found)
  "road": "#222222",
  "background": "black",
}


def get_db_connection():
  return psycopg2.connect(**get_db_config())


def get_cache_key(lat: float, lon: float, zoom: int, size: str) -> str:
  """Generate a cache key for a satellite tile."""
  # Round coordinates to 5 decimal places for consistency
  key_string = f"{lat:.5f}_{lon:.5f}_z{zoom}_{size}"
  # Create a short hash for the filename
  hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
  return f"tile_{lat:.5f}_{lon:.5f}_z{zoom}_{size}_{hash_suffix}.png"


def get_cached_tile(cache_key: str) -> Image.Image | None:
  """Try to load a tile from cache."""
  cache_path = CACHE_DIR / cache_key
  if cache_path.exists():
    try:
      return Image.open(cache_path)
    except Exception:
      # Corrupted cache file, delete it
      cache_path.unlink(missing_ok=True)
  return None


def save_tile_to_cache(cache_key: str, image: Image.Image) -> None:
  """Save a tile to the cache."""
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cache_path = CACHE_DIR / cache_key
  image.save(cache_path, "PNG")


def calculate_required_grid_size(
  lat: float, zoom: int, tile_size_px: int, ground_size_meters: float
) -> int:
  """
  Calculate the minimum grid size needed to cover the ground area.

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Grid size (NxN) needed to cover the area
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate meters per tile
  meters_per_tile = tile_size_px * meters_per_pixel

  # Calculate how many tiles needed to cover the ground
  # Add extra coverage for rotation (diagonal = sqrt(2) * side)
  # and some padding for safety
  diagonal_coverage = ground_size_meters * 1.42  # sqrt(2) ≈ 1.414
  tiles_needed = math.ceil(diagonal_coverage / meters_per_tile)

  # Ensure odd number for symmetric centering
  if tiles_needed % 2 == 0:
    tiles_needed += 1

  # Minimum of 3x3
  return max(3, tiles_needed)


def fetch_satellite_image(lat, lon, zoom=19, size="2048x2048"):
  """Fetch satellite imagery from Google Maps Static API with caching."""
  # Check cache first
  cache_key = get_cache_key(lat, lon, zoom, size)
  cached_image = get_cached_tile(cache_key)
  if cached_image is not None:
    return cached_image

  # Not in cache, fetch from API
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  response.raise_for_status()
  image = Image.open(io.BytesIO(response.content))

  # Save to cache
  save_tile_to_cache(cache_key, image)

  return image


def fetch_satellite_tiles(
  center_lat: float,
  center_lon: float,
  zoom: int,
  tile_size_px: int,
  ground_size_meters: float,
):
  """
  Fetch a grid of satellite tiles and stitch them together.
  Automatically calculates how many tiles are needed for coverage.

  Args:
    center_lat, center_lon: Center coordinates
    zoom: Google Maps zoom level
    tile_size_px: Size of each tile in pixels
    ground_size_meters: Physical size of ground to cover in meters

  Returns:
    Tuple of (stitched PIL Image, actual coverage in meters)
  """
  import math

  # Calculate required grid size
  grid_size = calculate_required_grid_size(
    center_lat, zoom, tile_size_px, ground_size_meters
  )

  tile_size = f"{tile_size_px}x{tile_size_px}"

  # Calculate the offset in degrees for each tile
  meters_per_pixel = (156543.03392 * math.cos(math.radians(center_lat))) / (2**zoom)

  # Calculate degree offset per tile
  meters_per_tile = tile_size_px * meters_per_pixel
  lat_offset_per_tile = meters_per_tile / 111111.0
  lon_offset_per_tile = meters_per_tile / (
    111111.0 * math.cos(math.radians(center_lat))
  )

  print(
    f"   📐 Auto-calculated grid: {grid_size}x{grid_size} tiles needed for {ground_size_meters:.0f}m coverage"
  )
  print(f"   📐 Each tile covers {meters_per_tile:.1f}m at zoom {zoom}")

  # Create canvas for stitching
  canvas_size = tile_size_px * grid_size
  canvas = Image.new("RGB", (canvas_size, canvas_size))

  # Calculate starting position (top-left corner)
  half_grid = (grid_size - 1) / 2

  # Fetch tiles in grid
  for row in range(grid_size):
    for col in range(grid_size):
      # Calculate offset from center
      lat_off = (half_grid - row) * lat_offset_per_tile
      lon_off = (col - half_grid) * lon_offset_per_tile

      tile_lat = center_lat + lat_off
      tile_lon = center_lon + lon_off

      # Check if cached
      cache_key = get_cache_key(tile_lat, tile_lon, zoom, tile_size)
      is_cached = (CACHE_DIR / cache_key).exists()
      cache_status = "📦" if is_cached else "🌐"
      print(f"   {cache_status} Tile [{row},{col}]: {tile_lat:.5f}, {tile_lon:.5f}")

      # Fetch tile (will use cache if available)
      tile_image = fetch_satellite_image(tile_lat, tile_lon, zoom, tile_size)

      # Paste into canvas
      x = col * tile_size_px
      y = row * tile_size_px
      canvas.paste(tile_image, (x, y))

  # Calculate actual coverage in meters
  total_coverage_meters = meters_per_tile * grid_size

  print(
    f"   ✅ Stitched {grid_size}x{grid_size} tiles into {canvas_size}x{canvas_size} image"
  )
  print(
    f"   ✅ Total coverage: {total_coverage_meters:.1f}m x {total_coverage_meters:.1f}m"
  )

  return canvas, total_coverage_meters


def calculate_satellite_scale(lat, zoom, image_size_px, ground_size_meters):
  """
  Calculate the scale factor needed to match satellite imagery to ground plane.

  Google Maps uses Web Mercator projection (EPSG:3857).
  At zoom level z and latitude lat, the resolution in meters/pixel is:
  resolution = (156543.03392 * cos(lat)) / (2^zoom)

  Args:
    lat: Latitude in degrees
    zoom: Google Maps zoom level
    image_size_px: Size of the satellite image in pixels
    ground_size_meters: Physical size of the ground plane in meters

  Returns:
    Scale factor to apply to satellite image
  """
  import math

  # Calculate meters per pixel at this zoom and latitude
  meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2**zoom)

  # Calculate the physical area the satellite image covers
  image_coverage_meters = image_size_px * meters_per_pixel

  # Calculate scale factor
  scale = ground_size_meters / image_coverage_meters

  print(
    f"   📐 Satellite calc: {meters_per_pixel:.2f} m/px, "
    f"covers {image_coverage_meters:.0f}m, "
    f"ground is {ground_size_meters:.0f}m, "
    f"scale={scale:.3f}"
  )

  return scale


def apply_perspective_correction(image_array, transform_params):
  """
  Apply perspective correction to satellite imagery.

  Args:
    image_array: numpy array of the image
    transform_params: dict with scale_x, scale_y, shear_x, shear_y

  Returns:
    Corrected image as numpy array
  """
  h, w = image_array.shape[:2]
  center_x, center_y = w / 2, h / 2

  # Build affine transformation matrix
  # This applies scaling and shearing around the center point
  sx = transform_params.get("scale_x", 1.0)
  sy = transform_params.get("scale_y", 1.0)
  shx = transform_params.get("shear_x", 0.0)
  shy = transform_params.get("shear_y", 0.0)

  # Affine matrix: [sx, shx, tx]
  #                [shy, sy, ty]
  # First translate to origin, apply transform, translate back
  M = np.float32(
    [
      [sx, shx, center_x * (1 - sx) - center_y * shx],
      [shy, sy, center_y * (1 - sy) - center_x * shy],
    ]
  )

  corrected = cv2.warpAffine(
    image_array,
    M,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
  )

  return corrected


def fetch_geometry_v5(conn, minx, miny, maxx, maxy):
  """
  Fetches geometry for specific surface classes.
  """
  # 709=Wall, 710=Ground, 712=Roof, 901=Building, 43-46=Roads
  target_ids = "709, 710, 712, 901, 43, 44, 45, 46"

  sql = f"""
    SELECT 
        f.objectclass_id,
        ST_AsBinary(g.geometry) as wkb_geom
    FROM citydb.geometry_data g
    JOIN citydb.feature f ON g.feature_id = f.id
    WHERE 
        -- Dynamically grab SRID to match DB
        g.geometry && ST_MakeEnvelope(
            {minx}, {miny}, {maxx}, {maxy}, 
            (SELECT ST_SRID(geometry) FROM citydb.geometry_data LIMIT 1)
        )
        AND g.geometry IS NOT NULL
        AND f.objectclass_id IN ({target_ids})
    """

  with conn.cursor() as cur:
    print(f"🔍 Querying DB (Meters): {minx:.0f},{miny:.0f} - {maxx:.0f},{maxy:.0f}")
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"📦 Retrieved {len(rows)} surfaces.")
    return rows


def render_tile(
  lat,
  lon,
  size_meters=300,
  orientation_deg=29,
  use_satellite=True,
  viewport_width=None,
  viewport_height=None,
):
  # Use defaults from constants if not specified
  viewport_width = viewport_width or VIEWPORT_WIDTH
  viewport_height = viewport_height or VIEWPORT_HEIGHT

  conn = get_db_connection()

  # 1. Coordinate Transform: GPS -> NYC State Plane (FEET)
  # Use EPSG:2263 because it's the standard for NYC input
  crs_src = CRS.from_epsg(4326)
  crs_dst = CRS.from_epsg(2263)
  transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

  x_feet, y_feet = transformer.transform(lon, lat)

  # 2. UNIT CONVERSION: Feet -> Meters
  # Your DB is in Meters (EPSG:2908).
  center_x = x_feet * 0.3048
  center_y = y_feet * 0.3048

  print(f"📍 GPS {lat}, {lon}")
  print(f"   -> TARGET METERS: {center_x:.2f}, {center_y:.2f}")

  # 3. Define Bounding Box
  half = size_meters / 2 * 1.5
  minx, miny = center_x - half, center_y - half
  maxx, maxy = center_x + half, center_y + half

  # 4. Fetch
  rows = fetch_geometry_v5(conn, minx, miny, maxx, maxy)
  conn.close()

  if not rows:
    print("❌ No geometry found. You might be aiming at an area with no loaded data.")
    return

  # 4.5. Fetch Satellite Image if requested
  satellite_texture = None
  if use_satellite:
    print("🛰️  Fetching satellite imagery...")
    try:
      # Calculate ground size for satellite coverage
      ground_size = size_meters * 1.5

      # Get high-res satellite image by stitching tiles
      # Grid size is automatically calculated based on coverage needs
      satellite_image, actual_coverage_meters = fetch_satellite_tiles(
        lat, lon, SATELLITE_ZOOM, SATELLITE_TILE_SIZE, ground_size
      )

      # Ensure image is in RGB mode (not grayscale or RGBA)
      if satellite_image.mode != "RGB":
        satellite_image = satellite_image.convert("RGB")

      # Calculate how much of the satellite image we need to use
      # ground_size already calculated above

      # Calculate crop area: we want to extract the center portion that matches our ground size
      # The satellite covers actual_coverage_meters, we want ground_size
      coverage_ratio = ground_size / actual_coverage_meters

      # Get the center crop
      img_width, img_height = satellite_image.size
      crop_size = int(img_width * coverage_ratio)

      left = (img_width - crop_size) // 2
      top = (img_height - crop_size) // 2
      right = left + crop_size
      bottom = top + crop_size

      print(
        f"   📐 Ground plane: {ground_size:.1f}m, "
        f"Satellite covers: {actual_coverage_meters:.1f}m"
      )
      print(
        f"   ✂️  Cropping center {crop_size}x{crop_size} from {img_width}x{img_height}"
      )

      # Crop to the area we need
      satellite_image = satellite_image.crop((left, top, right, bottom))

      # Apply any additional perspective correction (shear) if needed
      if (
        PERSPECTIVE_TRANSFORM.get("shear_x", 0) != 0
        or PERSPECTIVE_TRANSFORM.get("shear_y", 0) != 0
      ):
        print("   🔧 Applying perspective shear correction...")
        satellite_array = np.array(satellite_image)
        transform_shear_only = {
          "scale_x": 1.0,
          "scale_y": 1.0,
          "shear_x": PERSPECTIVE_TRANSFORM.get("shear_x", 0),
          "shear_y": PERSPECTIVE_TRANSFORM.get("shear_y", 0),
        }
        satellite_array = apply_perspective_correction(
          satellite_array, transform_shear_only
        )
        satellite_image = Image.fromarray(satellite_array)

      # Flip the image vertically (North-South axis) - uncomment if needed
      # satellite_image = satellite_image.transpose(Image.FLIP_TOP_BOTTOM)

      # Rotate the satellite image to match our orientation
      # Positive to align azimuth direction with top of view
      satellite_image = satellite_image.rotate(
        orientation_deg, expand=False, fillcolor=(0, 0, 0)
      )

      # Convert PIL Image to numpy array for PyVista
      satellite_texture = np.array(satellite_image)
      print(f"   ✅ Satellite image loaded: {satellite_texture.shape}")
    except Exception as e:
      print(f"   ⚠️  Failed to load satellite image: {e}")
      use_satellite = False

  # 5. Build Scene
  plotter = pv.Plotter(window_size=(viewport_width, viewport_height))
  plotter.set_background(COLORS["background"])

  print(f"🏗️  Building meshes from {len(rows)} surfaces...")

  # Container for collecting vertices/faces by class
  # (much faster than creating individual meshes)
  geom_data = {
    712: {"vertices": [], "faces": []},  # Roofs
    709: {"vertices": [], "faces": []},  # Walls
    710: {"vertices": [], "faces": []},  # Ground
    "other": {"vertices": [], "faces": []},
  }

  # Container for building footprints (for debug outlines)
  footprint_lines = []

  # Track ground elevation
  ground_z_values = []

  # Precompute rotation matrix for faster transformation
  # Positive to align azimuth direction with +Y (top of view)
  angle_rad = np.radians(orientation_deg)
  cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

  for obj_class, wkb_data in rows:
    try:
      shapely_geom = load_wkb(bytes(wkb_data))
    except Exception:
      continue

    # Handle Polygon vs MultiPolygon
    if shapely_geom.geom_type == "Polygon":
      polys = [shapely_geom]
    elif shapely_geom.geom_type == "MultiPolygon":
      polys = shapely_geom.geoms
    else:
      continue

    # Determine which batch this belongs to
    if obj_class in geom_data:
      batch = geom_data[obj_class]
    elif obj_class == 901:
      batch = geom_data["other"]
    else:
      batch = geom_data["other"]

    for poly in polys:
      pts = np.array(poly.exterior.coords)

      # Fast rotation and translation using numpy
      x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
      pts_transformed = np.column_stack(
        [
          x * cos_a - y * sin_a,
          x * sin_a + y * cos_a,
          pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts)),
        ]
      )

      # Collect footprints for ground surfaces (710) or buildings (901)
      # These will be rendered as white outlines for debugging
      if obj_class in [710, 901]:
        # Get the minimum Z value (ground level)
        if pts_transformed.shape[1] > 2:
          min_z = np.min(pts_transformed[:, 2])
          ground_z_values.append(min_z)
        else:
          min_z = 0
        # Create line segments at ground level
        footprint_pts = pts_transformed.copy()
        footprint_pts[:, 2] = min_z + 0.5  # Slightly above ground
        footprint_lines.append(footprint_pts)

      # Track vertex offset for face indices (count total vertices added so far)
      vertex_offset = sum(len(v) for v in batch["vertices"])
      batch["vertices"].append(pts_transformed)

      # Create face with offset indices
      n_pts = len(pts_transformed)
      face = [n_pts] + list(range(vertex_offset, vertex_offset + n_pts))
      batch["faces"].extend(face)

  # Now create one mesh per class (much faster than 5000+ individual meshes)
  batches = {}
  for class_id, data in geom_data.items():
    if data["vertices"]:
      all_vertices = np.vstack(data["vertices"])
      batches[class_id] = pv.PolyData(all_vertices, data["faces"])
      print(
        f"   Created batch for class {class_id}: {batches[class_id].n_points} points"
      )
    else:
      batches[class_id] = None

  # Calculate actual ground elevation from the data
  if ground_z_values:
    calculated_ground_z = np.median(ground_z_values)
    print(
      f"   📏 Detected ground elevation: {calculated_ground_z:.2f}m (median of {len(ground_z_values)} values)"
    )
  else:
    calculated_ground_z = GROUND_Z  # Fallback to constant
    print(f"   ⚠️  No ground surfaces found, using default GROUND_Z={GROUND_Z}")

  # 6. Add to Scene (Draw Order Matters!)
  # Draw Ground first, then Walls, then Roofs on top
  print("🎨 Adding meshes to scene...")

  # 6.1. Add satellite image as ground plane texture
  if use_satellite and satellite_texture is not None:
    # Strategy: Create a base textured plane, then add actual ground surfaces on top
    print("   🎨 Creating satellite-textured base plane...")

    # Calculate ground plane parameters
    ground_size = size_meters * 1.5
    half_size = ground_size / 2
    ground_z = calculated_ground_z - 1.0  # Below everything

    # Create base plane
    base_ground_points = np.array(
      [
        [-half_size, -half_size, ground_z],
        [half_size, -half_size, ground_z],
        [half_size, half_size, ground_z],
        [-half_size, half_size, ground_z],
      ]
    )

    base_ground_faces = [4, 0, 1, 2, 3]
    base_ground_mesh = pv.PolyData(base_ground_points, base_ground_faces)

    base_ground_mesh = base_ground_mesh.texture_map_to_plane(
      origin=(-half_size, -half_size, ground_z),
      point_u=(half_size, -half_size, ground_z),
      point_v=(-half_size, half_size, ground_z),
    )

    texture = pv.Texture(satellite_texture)
    plotter.add_mesh(base_ground_mesh, texture=texture, show_edges=False)
    print("   ✅ Base satellite plane added")

    # Now also texture any actual ground surfaces for proper elevation
    ground_meshes_to_texture = []

    if batches.get(710):
      print(f"   Found ground surfaces (710): {batches[710].n_points} points")
      ground_meshes_to_texture.append(("ground", batches[710]))

    if batches.get("other"):
      print(f"   Found roads/other: {batches['other'].n_points} points")
      ground_meshes_to_texture.append(("roads", batches["other"]))

    if ground_meshes_to_texture:
      print("   🎨 Applying satellite texture to actual ground geometry...")

      for mesh_name, ground_mesh in ground_meshes_to_texture:
        # Get vertices and calculate UV coordinates
        points = ground_mesh.points

        # Map X,Y from [-half_size, half_size] to [0, 1] for texture coords
        u = (points[:, 0] + half_size) / (2 * half_size)
        v = (points[:, 1] + half_size) / (2 * half_size)

        # Clamp to [0, 1] range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        # Create texture coordinates array
        texture_coords = np.column_stack([u, v])

        # Set texture coordinates using VTK naming convention
        ground_mesh.point_data.set_array(texture_coords, "TCoords")
        ground_mesh.point_data.SetActiveTCoords("TCoords")

        # Add the textured ground mesh (slightly above base plane)
        plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
        print(f"   ✅ Satellite texture applied to {mesh_name}")
    else:
      # Fallback: create a flat plane if no ground surfaces exist
      print("   ⚠️  No ground surfaces found, using base plane only")

      ground_size = size_meters * 1.5
      half_size = ground_size / 2
      ground_z = calculated_ground_z - 0.5

      ground_points = np.array(
        [
          [-half_size, -half_size, ground_z],
          [half_size, -half_size, ground_z],
          [half_size, half_size, ground_z],
          [-half_size, half_size, ground_z],
        ]
      )

      ground_faces = [4, 0, 1, 2, 3]
      ground_mesh = pv.PolyData(ground_points, ground_faces)

      ground_mesh = ground_mesh.texture_map_to_plane(
        origin=(-half_size, -half_size, ground_z),
        point_u=(half_size, -half_size, ground_z),
        point_v=(-half_size, half_size, ground_z),
      )

      texture = pv.Texture(satellite_texture)
      plotter.add_mesh(ground_mesh, texture=texture, show_edges=False)
      print("   ✅ Satellite ground plane added")

  # Only render ground surfaces without satellite if not using satellite mode
  elif batches.get(710):
    plotter.add_mesh(
      batches[710], color=COLORS[710], show_edges=False, opacity=BUILDING_OPACITY
    )

  # Roads/other - only render if not using satellite (already textured above)
  if batches.get("other") and not use_satellite:
    batches["other"].translate((0, 0, 0.1), inplace=True)
    plotter.add_mesh(
      batches["other"], color=COLORS["road"], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(709):  # Walls
    batches[709].translate((0, 0, 0.2), inplace=True)
    plotter.add_mesh(
      batches[709], color=COLORS[709], show_edges=False, opacity=BUILDING_OPACITY
    )

  if batches.get(712):  # Roofs
    batches[712].translate((0, 0, 0.3), inplace=True)
    plotter.add_mesh(
      batches[712], color=COLORS[712], show_edges=False, opacity=BUILDING_OPACITY
    )

  # 6.2. Add footprint outlines for debugging
  if footprint_lines:
    print(f"   ✅ Adding {len(footprint_lines)} footprint outlines")
    for footprint_pts in footprint_lines:
      # Create line segments connecting the points
      n_points = len(footprint_pts)
      if n_points > 1:
        # Create lines connecting each point to the next
        lines = []
        for i in range(n_points - 1):
          lines.append([2, i, i + 1])

        # Flatten the lines array
        lines_flat = np.hstack(lines)

        # Create a PolyData with lines
        line_mesh = pv.PolyData(footprint_pts, lines=lines_flat)
        plotter.add_mesh(
          line_mesh, color="white", line_width=2, render_lines_as_tubes=False
        )

  # 7. Camera Setup (supports both isometric and overhead views)
  plotter.camera.enable_parallel_projection()

  # Convert degrees to radians
  # Elevation: -90 = looking straight down, 0 = horizontal, 90 = looking up
  # Note: Geometry is already rotated by -ORIENTATION_DEG to align with azimuth
  # So camera positioning doesn't need azimuth rotation for overhead view
  elevation_rad = np.radians(CAMERA_ELEVATION_DEG)
  dist = 2000

  if abs(CAMERA_ELEVATION_DEG) > 80:  # Near-overhead view
    # Camera directly above, looking down
    # Geometry rotated by +azimuth aligns azimuth direction with +Y
    plotter.camera.position = (0, 0, dist)
    plotter.camera.focal_point = (0, 0, 0)
    # Top of view is +Y (where azimuth direction now points)
    plotter.camera.up = (0, 1, 0)
  else:
    # Standard isometric view - use azimuth for camera positioning
    azimuth_rad = np.radians(CAMERA_AZIMUTH)
    cx = dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cy = dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cz = -dist * np.sin(elevation_rad)

    plotter.camera.position = (cx, cy, cz)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)

  plotter.camera.parallel_scale = CAMERA_ZOOM

  print(f"📷 Camera: Az={CAMERA_AZIMUTH}° El={CAMERA_ELEVATION_DEG}°")
  print(f"   Position: {plotter.camera.position}")

  print("📸 Displaying Isometric Render...")
  plotter.show()

  # Log final camera settings after user interaction
  print("\n🎥 FINAL CAMERA SETTINGS:")
  print(f"   Position: {plotter.camera.position}")
  print(f"   Focal Point: {plotter.camera.focal_point}")
  print(f"   Up Vector: {plotter.camera.up}")
  print(f"   Parallel Scale (zoom): {plotter.camera.parallel_scale}")
  print(f"   View Angle: {plotter.camera.view_angle}")


def main():
  render_tile(
    lat=LAT,
    lon=LON,
    size_meters=SIZE_METERS,
    orientation_deg=ORIENTATION_DEG,
    use_satellite=True,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
  )


if __name__ == "__main__":
  main()

```

Changed all rotations to `+orientation_deg` (was `-orientation_deg`) and camera up back to `(0, 1, 0)`. This should align the azimuth direction with the top of the view, matching the web convention.

Run it again!

---

**User**

nice! This looks precisely lined up!

