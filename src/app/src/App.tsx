import { useState, useCallback, useMemo, useEffect } from "react";
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
  tileUrlPattern: string;
  maxZoomLevel: number;
}

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

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

const VIEW_STATE_STORAGE_KEY = "isometric-nyc-view-state";

// Load saved view state from localStorage
function loadSavedViewState(): ViewState | null {
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

  // Load tile manifest on mount
  useEffect(() => {
    fetch("/tiles/manifest.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load manifest: ${res.status}`);
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
  }, []);

  const [viewState, setViewState] = useState<ViewState | null>(null);

  // Initialize view state once tile config is loaded
  // Try to restore from localStorage, otherwise center on the content area
  useEffect(() => {
    if (tileConfig && !viewState) {
      // Try to load saved view state first
      const savedViewState = loadSavedViewState();
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

  const [lightDirection, setLightDirection] = useState<
    [number, number, number]
  >([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [scanlines, setScanlines] = useState({
    enabled: true,
    count: 600,
    opacity: 0.1,
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

  // Compute visible tile count for stats
  const visibleTiles = useMemo(() => {
    if (!viewState || !tileConfig) return 0;
    const scale = Math.pow(2, viewState.zoom);
    const viewportWidth = window.innerWidth / scale;
    const viewportHeight = window.innerHeight / scale;
    const tilesX = Math.ceil(viewportWidth / tileConfig.tileSize) + 1;
    const tilesY = Math.ceil(viewportHeight / tileConfig.tileSize) + 1;
    return tilesX * tilesY;
  }, [viewState, tileConfig]);

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
        lightDirection={lightDirection}
        onTileHover={handleTileHover}
        scanlines={scanlines}
        waterShader={waterShader}
      />

      <header className="header">
        <h1>Isometric NYC</h1>
        <span className="subtitle">Pixel Art City Explorer</span>
      </header>

      <ControlPanel
        zoom={viewState.zoom}
        lightDirection={lightDirection}
        onLightDirectionChange={setLightDirection}
        visibleTiles={visibleTiles}
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
