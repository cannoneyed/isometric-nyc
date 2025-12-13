import { useState, useCallback, useMemo, useEffect } from "react";
import { IsometricMap } from "./components/IsometricMap";
import { ControlPanel } from "./components/ControlPanel";
import { TileInfo } from "./components/TileInfo";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  tileSize: number;
  tileUrlPattern: string;
}

interface TileManifest {
  gridWidth: number;
  gridHeight: number;
  tileSize: number;
  totalTiles: number;
  generated: string;
  urlPattern: string;
}

export interface ViewState {
  target: [number, number, number];
  zoom: number;
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
          tileSize: manifest.tileSize,
          tileUrlPattern: `/tiles/{z}/{x}_{y}.png`,
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
  useEffect(() => {
    if (tileConfig && !viewState) {
      setViewState({
        target: [
          (tileConfig.gridWidth * tileConfig.tileSize) / 2,
          (tileConfig.gridHeight * tileConfig.tileSize) / 2,
          0,
        ],
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

  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      setViewState(params.viewState);
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
      />

      <TileInfo hoveredTile={hoveredTile} viewState={viewState} />
    </div>
  );
}

export default App;
