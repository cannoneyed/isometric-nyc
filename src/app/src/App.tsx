import { useState, useCallback, useMemo } from 'react';
import { IsometricMap } from './components/IsometricMap';
import { ControlPanel } from './components/ControlPanel';
import { TileInfo } from './components/TileInfo';

// Configuration for the tile grid
const TILE_CONFIG = {
  // Grid dimensions (20x20 for testing)
  gridWidth: 20,
  gridHeight: 20,
  // Tile size in pixels (at zoom level 0, 1 pixel = 1 world unit)
  tileSize: 512,
  // URL pattern for tiles
  tileUrlPattern: '/tiles/{z}/{x}/{y}.png',
};

export interface ViewState {
  target: [number, number, number];
  zoom: number;
}

function App() {
  const [viewState, setViewState] = useState<ViewState>({
    // Center on middle of 20x20 grid
    target: [
      (TILE_CONFIG.gridWidth * TILE_CONFIG.tileSize) / 2,
      (TILE_CONFIG.gridHeight * TILE_CONFIG.tileSize) / 2,
      0,
    ],
    zoom: -2, // Start zoomed out to see multiple tiles
  });

  const [lightDirection, setLightDirection] = useState<[number, number, number]>([0.5, 0.5, 1.0]);
  const [hoveredTile, setHoveredTile] = useState<{ x: number; y: number } | null>(null);

  const handleViewStateChange = useCallback((params: { viewState: ViewState }) => {
    setViewState(params.viewState);
  }, []);

  const handleTileHover = useCallback((tile: { x: number; y: number } | null) => {
    setHoveredTile(tile);
  }, []);

  // Compute visible tile count for stats
  const visibleTiles = useMemo(() => {
    const scale = Math.pow(2, viewState.zoom);
    const viewportWidth = window.innerWidth / scale;
    const viewportHeight = window.innerHeight / scale;
    const tilesX = Math.ceil(viewportWidth / TILE_CONFIG.tileSize) + 1;
    const tilesY = Math.ceil(viewportHeight / TILE_CONFIG.tileSize) + 1;
    return tilesX * tilesY;
  }, [viewState.zoom]);

  return (
    <div className="app">
      <IsometricMap
        tileConfig={TILE_CONFIG}
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

      <TileInfo
        hoveredTile={hoveredTile}
        viewState={viewState}
      />
    </div>
  );
}

export default App;

