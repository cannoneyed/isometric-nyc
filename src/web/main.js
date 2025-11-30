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
import { Scene, WebGLRenderer, PerspectiveCamera, MathUtils } from "three";
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
