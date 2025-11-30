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

// Madison Square Garden coordinates
const LAT = 40.7505;
const LON = -73.9934;
const HEIGHT = 300; // meters above ground

// Camera angles (SimCity isometric style)
// Azimuth: direction camera is FACING (30° = NE-ish, like whitebox.py)
// Elevation: negative = looking down at ground
const CAMERA_AZIMUTH = 30; // North/south
const CAMERA_ELEVATION = -90; // Straight down

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
