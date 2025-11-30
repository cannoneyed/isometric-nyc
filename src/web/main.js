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

import view from "../view.json";

// --- CONFIGURATION ---
const API_KEY = import.meta.env.VITE_GOOGLE_TILES_API_KEY;
console.log("API Key loaded:", API_KEY ? "Yes" : "No");

// Fixed canvas size (consistent rendering regardless of window size)
const CANVAS_WIDTH = 1280;
const CANVAS_HEIGHT = 720;

// Check if we're in export mode (for screenshot capture)
const urlParams = new URLSearchParams(window.location.search);
const EXPORT_MODE = urlParams.get("export") === "true";

if (EXPORT_MODE) {
  console.log("🎬 Export mode enabled - will signal when ready for screenshot");
  window.__EXPORT_READY = false;
  window.__TILES_LOADED = false;
  window.__CAMERA_POSITIONED = false;
}

const EXPORT_DELAY_MS = 2000; // 2 seconds should be enough for tiles to render

// Madison Square Garden coordinates
const LAT = view.lat;
const LON = view.lon;
const HEIGHT = view.camera_height_meters; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = view.camera_azimuth_degrees; // Square on the avenues
const CAMERA_ELEVATION = view.camera_elevation_degrees; // Straight down

let scene, renderer, controls, tiles, transition;
let isOrthographic = true; // Start in orthographic (isometric) mode

// Debounced camera info logging
let logTimeout = null;
let lastCameraState = { az: 0, el: 0, height: 0 };

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

  // Connect controls to the tiles ellipsoid and position camera
  tiles.addEventListener("load-tile-set", () => {
    controls.setEllipsoid(tiles.ellipsoid, tiles.group);
    positionCamera();

    // Signal tileset is loaded for export mode
    if (EXPORT_MODE) {
      window.__TILES_LOADED = true;
      console.log("🗺️ Tileset loaded");
      checkExportReady();
    }
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

  // Fix orthographic camera aspect ratio to avoid distortion
  // GlobeControls sets the frustum, but we need to adjust for our canvas aspect
  const ortho = transition.orthographicCamera;
  const aspect = CANVAS_WIDTH / CANVAS_HEIGHT;
  const frustumHeight = ortho.top - ortho.bottom; // Get current vertical size
  const halfHeight = frustumHeight / 2;
  const halfWidth = halfHeight * aspect;

  // Recalculate frustum with correct aspect ratio
  ortho.left = -halfWidth;
  ortho.right = halfWidth;
  // Keep top/bottom as GlobeControls set them
  ortho.updateProjectionMatrix();

  // Switch to orthographic mode by default
  if (isOrthographic && transition.mode === "perspective") {
    controls.getPivotPoint(transition.fixedPoint);
    transition.toggle();
  }

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

  // Signal camera is positioned for export mode
  if (EXPORT_MODE) {
    window.__CAMERA_POSITIONED = true;
    console.log("📷 Camera positioned - waiting for tiles to load...");
    checkExportReady();
  }
}

// Check if ready for export screenshot (called after tiles loaded and camera positioned)
let exportReadyTimeout = null;
function checkExportReady() {
  if (!EXPORT_MODE) return;

  // Need both conditions to be true
  if (!window.__TILES_LOADED || !window.__CAMERA_POSITIONED) {
    return;
  }

  // Clear any existing timeout
  if (exportReadyTimeout) {
    clearTimeout(exportReadyTimeout);
  }

  // Wait for tiles to finish loading and rendering
  // The tileset loads progressively, so we need to give it time
  console.log(
    `⏳ Waiting ${EXPORT_DELAY_MS / 1000}s for tiles to finish rendering...`
  );

  exportReadyTimeout = setTimeout(() => {
    window.__EXPORT_READY = true;
    console.log("✅ Export ready - screenshot can be taken now");
  }, EXPORT_DELAY_MS);
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
  // Don't add UI in export mode (we want a clean screenshot)
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
  };
}

// Debounced logging of camera state
function logCameraState() {
  // Skip logging in export mode to reduce noise
  if (EXPORT_MODE) return;

  const info = getCameraInfo();
  if (!info) return;

  // Check if state has changed significantly
  const changed =
    Math.abs(info.azimuth - lastCameraState.az) > 0.5 ||
    Math.abs(info.elevation - lastCameraState.el) > 0.5 ||
    Math.abs(info.height - lastCameraState.height) > 1;

  if (changed) {
    lastCameraState = {
      az: info.azimuth,
      el: info.elevation,
      height: info.height,
    };

    // Clear existing timeout
    if (logTimeout) clearTimeout(logTimeout);

    // Debounce: wait 200ms before logging
    logTimeout = setTimeout(() => {
      console.log(
        `📷 Camera: Az=${info.azimuth.toFixed(1)}° El=${info.elevation.toFixed(
          1
        )}° ` +
          `Height=${info.height.toFixed(0)}m | Lat=${info.lat.toFixed(
            4
          )}° Lon=${info.lon.toFixed(4)}°`
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
