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
  Matrix4,
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  OrthographicCamera,
  MathUtils,
} from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

// Configuration with URL param overrides
const CANVAS_WIDTH = parseInt(urlParams.get("width")) || view.width_px;
const CANVAS_HEIGHT = parseInt(urlParams.get("height")) || view.height_px;
const LAT = parseFloat(urlParams.get("lat")) || view.lat;
const LON = parseFloat(urlParams.get("lon")) || view.lon;
const CAMERA_AZIMUTH =
  parseFloat(urlParams.get("azimuth")) || view.camera_azimuth_degrees;
const CAMERA_ELEVATION =
  parseFloat(urlParams.get("elevation")) || view.camera_elevation_degrees;
const VIEW_HEIGHT_METERS =
  parseFloat(urlParams.get("view_height")) || view.view_height_meters || 200;

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0, zoom: 1 };

// Tile loading tracking
let tilesStableStartTime = 0;
window.TILES_LOADED = false;
let cameraInitialized = false;

// Also, whitebox.py calculates positions relative to Z=0.
// However, buildings have height, and we probably want to look at the ground (Z=0).
// But wait! whitebox.py detected ground elevation:
// "Detected ground elevation: 10.00m" (typically around 10-15m for NYC)
// And then it textures the ground at calculated_ground_z.
// But the camera focal point is (0,0,0).
// The geometry in whitebox.py is shifted:
// x, y = pts[:, 0] - center_x, pts[:, 1] - center_y
// But Z values are preserved from the DB.
// So if the DB has ground at Z=10, the camera looking at Z=0 is looking 10m BELOW ground.

// In 3D Tiles (Google), the tiles are positioned on the WGS84 ellipsoid.
// When we ask for a frame at height=0, we get the ellipsoid surface.
// NYC ground level is indeed around 10-30m above the ellipsoid in some places,
// but Google 3D tiles usually match the ellipsoid roughly or have their own geoid.

// If whitebox.py is rendering geometry where Z=0 is "arbitrary zero", but actual geometry is at Z=10,
// and camera looks at Z=0, then the view is centered 10m below the buildings.

// In the web view, we are centering on the ellipsoid surface (height=0).
// If the visible Google 3D tiles are at height=10, then we are also looking 10m below the buildings.

// It seems they match in intent (looking at Z=0/Ellipsoid), but we might need to tweak the
// center point height to match exactly if there is a shift.

// For now, let's keep it at 0. If there is a vertical offset, we can adjust here.
// Example: Look at 15m elevation to center on "street level" if streets are elevated.

// Whitebox.py finds median ground Z around 10-15m for MSG.
// It then constructs the scene around that.
// Since its camera looks at (0,0,0), it is looking at Z=0 relative to the *PostGIS* coordinates.
// If PostGIS coordinates have ground at Z=10, then the camera is looking 10m below ground.
// We are doing the same here (looking at Z=0 on ellipsoid).

// HOWEVER, if we want them to align PIXEL-PERFECTLY:
// We need to account for any difference in how the "center" is defined.
// Whitebox centers on the *projected* coordinates of (LAT, LON).
// 3D Tiles Renderer centers on the *cartesian* coordinates of (LAT, LON, 0).

// There might be a small shift due to projection.
// But more likely, it's the Z-height of the "center of rotation".
// Let's try bumping the target height to match the ground elevation ~10m.
// Or, if the whitebox image is "higher" (building lower in frame), we need to look *lower* (smaller Z).

// Observation: The web view shows the building slightly "higher" in the frame than whitebox.
// This means the camera is looking *below* the point that whitebox is looking at.
// Or whitebox is looking *above* the point we are looking at.

// Whitebox look-at: (0,0,0). Ground is at Z~10. So it looks 10m below ground.
// Web look-at: Ellipsoid surface (Z=0).

// Let's try adjusting the target height to see if it aligns better.
// If we look at 15m, the camera moves up, and the scene moves DOWN in the frame.
// If the web view is too high, we need to look HIGHER (larger Z).
const TARGET_HEIGHT = -31.3; // Geoid height for NYC (approx -31.3m)

init();
animate();

function init() {
  // Use fixed canvas dimensions for consistent rendering
  const aspect = CANVAS_WIDTH / CANVAS_HEIGHT;
  console.log(
    `🖥️ Fixed canvas: ${CANVAS_WIDTH}x${CANVAS_HEIGHT}, aspect: ${aspect.toFixed(
      3
    )}`
  );

  // Renderer - fixed size, no devicePixelRatio scaling for consistency
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x87ceeb); // Sky blue
  renderer.setPixelRatio(1); // Fixed 1:1 pixel ratio for consistent rendering
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new Scene();

  // Camera transition manager (handles both perspective and orthographic)
  transition = new CameraTransitionManager(
    new PerspectiveCamera(60, aspect, 1, 160000000),
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
  controls.minZoom = 0.1; // Allow zooming out
  controls.maxZoom = 20.0; // Allow zooming in

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);

    // Delay camera positioning to ensure controls/transition are fully initialized
    // This fixes the "zoomed out on first load" issue
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (!cameraInitialized) {
          positionCamera();
          cameraInitialized = true;
        }
      });
    });
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
    TARGET_HEIGHT, // CENTER AT TARGET HEIGHT
    CAMERA_AZIMUTH * MathUtils.DEG2RAD,
    CAMERA_ELEVATION * MathUtils.DEG2RAD,
    0, // roll
    camera.matrixWorld,
    CAMERA_FRAME
  );

  // Move camera back 2000m along its own Z axis (viewing direction)
  // This matches whitebox.py's "dist = 2000" logic

  camera.matrixWorld.multiply(new Matrix4().makeTranslation(0, 0, 2000));

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
  // IMPORTANT: Do this BEFORE setting frustum, as toggle() may reset camera values
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

  // Calculate orthographic frustum to match whitebox.py's CAMERA_ZOOM
  // In VTK, parallel_scale = half the view height in world units
  // whitebox.py uses CAMERA_ZOOM = 100, so view height = 200m
  const ortho = transition.orthographicCamera;
  const aspect = CANVAS_WIDTH / CANVAS_HEIGHT;

  // Match whitebox.py visually
  // view.view_height_meters determines the vertical extent of the view in world units (meters)
  // NOTE: whitebox.py uses parallel_scale = VIEW_HEIGHT_METERS / 2
  // This means the total height of the view is VIEW_HEIGHT_METERS.
  // However, the camera is positioned at a significant height.
  // In whitebox.py, the camera is positioned so the focal point (0,0,0) is in the CENTER of the screen.
  // (0,0,0) corresponds to the lat/lon in view.json.

  const frustumHeight = VIEW_HEIGHT_METERS;
  const halfHeight = frustumHeight / 2;
  const halfWidth = halfHeight * aspect;

  console.log(`📐 Frustum: height=${frustumHeight}m`);

  // Set frustum with calculated values
  ortho.top = halfHeight;
  ortho.bottom = -halfHeight;
  ortho.left = -halfWidth;
  ortho.right = halfWidth;

  // Reset zoom to 1.0 to ensure strict 1:1 scale with world units
  ortho.zoom = 1.0;
  ortho.updateProjectionMatrix();

  // Shift the camera to center the target point
  // In 3d-tiles-renderer, the camera looks at the target point.
  // But if we are using getObjectFrame, the camera is positioned relative to the target point.
  // We want the target point to be in the center of the screen.
  // That is already what getObjectFrame does (looks at the origin of the frame).

  console.log(`Camera positioned above Times Square at ${HEIGHT}m`);
  console.log(`Azimuth: ${CAMERA_AZIMUTH}°, Elevation: ${CAMERA_ELEVATION}°`);
  console.log(`Mode: ${transition.mode}`);

  // Log camera info
  console.log(
    `📷 Ortho frustum: L=${ortho.left.toFixed(0)} R=${ortho.right.toFixed(
      0
    )} ` +
      `T=${ortho.top.toFixed(0)} B=${ortho.bottom.toFixed(
        0
      )} (aspect=${aspect.toFixed(2)})`
  );
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
  // Hide UI in export mode for clean screenshots
  if (EXPORT_MODE) return;

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
  // Canvas is fixed size - don't resize on window changes
  // This ensures consistent rendering regardless of window size
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
          `Height=${info.height.toFixed(0)}m Zoom=${info.zoom.toFixed(
            2
          )} | Lat=${info.lat.toFixed(4)}° Lon=${info.lon.toFixed(4)}°`
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
  logCameraState();

  renderer.render(scene, camera);
}
