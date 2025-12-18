import { useState, useCallback, useEffect } from "react";
import { PMTiles } from "pmtiles";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";
import { defaultShaderParams } from "./shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
}

// Legacy manifest format (for backward compatibility)
interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  originalWidth?: number;
  originalHeight?: number;
  tileSize: number;
  totalTiles: number;
  maxZoomLevel: number;
  generated: string;
  urlPattern: string;
}

// PMTiles metadata format
interface PMTilesMetadata {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoom: number;
  pmtilesZoomMap?: Record<string, number>; // Maps our level -> PMTiles z (keys are strings in JSON)
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Check for reset query parameter
function checkForReset(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("reset") === "1") {
    localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
    // Clean URL without reload
    window.history.replaceState({}, "", window.location.pathname);
    return true;
  }
  return false;
}

// Load saved view state from localStorage
function loadSavedViewState(tileConfig?: TileConfig): ViewState | null {
  // Check for reset first
  if (checkForReset()) {
    return null;
  }

  try {
    const saved = localStorage.getItem(VIEW_STATE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Validate the structure - target can be 2 or 3 elements
      if (
        Array.isArray(parsed.target) &&
        parsed.target.length >= 2 &&
        typeof parsed.zoom === "number"
      ) {
        // Normalize to 3-element target
        const target: [number, number, number] = [
          parsed.target[0],
          parsed.target[1],
          parsed.target[2] ?? 0,
        ];

        // Validate position is within reasonable bounds if we have config
        if (tileConfig) {
          const maxX = tileConfig.gridWidth * tileConfig.tileSize;
          const maxY = tileConfig.gridHeight * tileConfig.tileSize;
          if (
            target[0] < 0 ||
            target[0] > maxX ||
            target[1] < 0 ||
            target[1] > maxY
          ) {
            console.warn(
              "Saved view position out of bounds, resetting:",
              target
            );
            localStorage.removeItem(VIEW_STATE_STORAGE_KEY);
            return null;
          }
        }

        return { target, zoom: parsed.zoom };
      }
    }
  } catch (e) {
    console.warn("Failed to load saved view state:", e);
  }
  return null;
}

// Save view state to localStorage (debounced to avoid excessive writes)
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
function saveViewState(viewState: ViewState): void {
  // Debounce saves to avoid excessive localStorage writes during panning
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }
  saveTimeout = setTimeout(() => {
    try {
      localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(viewState));
    } catch (e) {
      console.warn("Failed to save view state:", e);
    }
  }, 500);
}

function App() {
  const [tileConfig, setTileConfig] = useState<TileConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tile configuration on mount
  // Tries PMTiles first, falls back to legacy manifest.json
  useEffect(() => {
    const pmtilesUrl = "/tiles.pmtiles";

    // Try PMTiles first
    const pmtiles = new PMTiles(pmtilesUrl);
    pmtiles
      .getHeader()
      .then(() => {
        // PMTiles file exists, get metadata
        return pmtiles.getMetadata();
      })
      .then((metadata) => {
        const meta = metadata as PMTilesMetadata;
        console.log("Loaded PMTiles metadata:", meta);

        // Convert zoom map keys from strings to numbers
        const zoomMap: Record<number, number> | undefined = meta.pmtilesZoomMap
          ? Object.fromEntries(
              Object.entries(meta.pmtilesZoomMap).map(([k, v]) => [
                parseInt(k, 10),
                v,
              ])
            )
          : undefined;

        setTileConfig({
          gridWidth: meta.gridWidth,
          gridHeight: meta.gridHeight,
          originalWidth: meta.originalWidth ?? meta.gridWidth,
          originalHeight: meta.originalHeight ?? meta.gridHeight,
          tileSize: meta.tileSize ?? 512,
          maxZoomLevel: meta.maxZoom ?? 4,
          pmtilesUrl: pmtilesUrl,
          pmtilesZoomMap: zoomMap,
        });
        setLoading(false);
      })
      .catch((pmtilesErr) => {
        console.log(
          "PMTiles not available, falling back to legacy manifest:",
          pmtilesErr
        );

        // Fall back to legacy manifest.json
        fetch("/tiles/manifest.json")
          .then((res) => {
            if (!res.ok)
              throw new Error(`Failed to load manifest: ${res.status}`);
            return res.json() as Promise<TileManifest>;
          })
          .then((manifest) => {
            setTileConfig({
              gridWidth: manifest.gridWidth,
              gridHeight: manifest.gridHeight,
              originalWidth: manifest.originalWidth ?? manifest.gridWidth,
              originalHeight: manifest.originalHeight ?? manifest.gridHeight,
              tileSize: manifest.tileSize,
              tileUrlPattern: `/tiles/{z}/{x}_{y}.png`,
              maxZoomLevel: manifest.maxZoomLevel ?? 0,
            });
            setLoading(false);
          })
          .catch((err) => {
            console.error("Failed to load tile manifest:", err);
            setError(err.message);
            setLoading(false);
          });
      });
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first (pass tileConfig for bounds validation)
      const savedViewState = loadSavedViewState(tileConfig);
      if (savedViewState) {
        console.log(
          `View init: restoring saved position (${savedViewState.target[0]}, ${savedViewState.target[1]}), zoom=${savedViewState.zoom}`
        );
        setViewState(savedViewState);
        return;
      }

      // Fall back to centering on content
      const { originalWidth, originalHeight, gridHeight, tileSize } =
        tileConfig;

      // Content is at deck.gl x = 0 to originalWidth-1
      // Content is at deck.gl y = gridHeight-originalHeight to gridHeight-1 (due to Y-flip)
      // Center of content:
      const centerX = (originalWidth / 2) * tileSize;
      const centerY = (gridHeight - originalHeight / 2) * tileSize;

      console.log(
        `View init: centering at (${centerX}, ${centerY}), original=${originalWidth}x${originalHeight}, padded=${tileConfig.gridWidth}x${gridHeight}`
      );

      setViewState({
        target: [centerX, centerY, 0],
        zoom: -2,
      });
    }
  }, [tileConfig, viewState]);

  // Light direction for future use (currently unused)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_lightDirection, _setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.05,
  });

  const [waterShader, setWaterShader] = useState({
    enabled: true,
    showMask: false,
    params: defaultShaderParams,
  });

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
      saveViewState(params.viewState);
    },
    []
  );

  const handleTileHover = useCallback(
    (tile: { x: number; y: number } | null) => {
      setHoveredTile(tile);
    },
    []
  );

  // Loading state
  if (loading) {
    return (
      <div className="app loading">
        <div className="loading-message">Loading tile manifest...</div>
      </div>
    );
  }

  // Error state
  if (error || !tileConfig) {
    return (
      <div className="app error">
        <div className="error-message">
          Failed to load tiles: {error || "Unknown error"}
        </div>
      </div>
    );
  }

  // Wait for view state to be initialized
  if (!viewState) {
    return (
      <div className="app loading">
        <div className="loading-message">Initializing view...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <IsometricMap
        tileConfig={tileConfig}
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        lightDirection={_lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            title="About / Making Of"
            onClick={() => {
              // TODO: Open about modal
              console.log("About clicked");
            }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
        </div>
      </header>

      <ControlPanel
        scanlines={scanlines}
        onScanlinesChange={setScanlines}
        waterShader={waterShader}
        onWaterShaderChange={setWaterShader}
      />

      <TileInfo hoveredTile={hoveredTile} viewState={viewState} />
    </div>
  );
}

export default App;
