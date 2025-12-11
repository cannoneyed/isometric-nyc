import { useMemo, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import { OrthographicView } from "@deck.gl/core";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";
import type { ViewState } from "../App";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  tileSize: number;
  tileUrlPattern: string;
}

interface IsometricMapProps {
  tileConfig: TileConfig;
  viewState: ViewState;
  onViewStateChange: (params: { viewState: ViewState }) => void;
  lightDirection: [number, number, number];
  onTileHover: (tile: { x: number; y: number } | null) => void;
}

export function IsometricMap({
  tileConfig,
  viewState,
  onViewStateChange,
  onTileHover,
}: IsometricMapProps) {
  // Calculate the total extent of the tile grid
  const extent = useMemo(() => {
    const { gridWidth, gridHeight, tileSize } = tileConfig;
    return {
      minX: 0,
      minY: 0,
      maxX: gridWidth * tileSize,
      maxY: gridHeight * tileSize,
    };
  }, [tileConfig]);

  // Determine max zoom based on tile size
  // At zoom 0, 1 pixel = 1 world unit
  // At zoom 1, 1 pixel = 0.5 world units (2x magnification)
  const maxZoom = 4; // Up to 16x magnification
  const minZoom = -4; // Down to 1/16x (see full map)

  // Create the tile layer
  const layers = useMemo(() => {
    const { tileSize, tileUrlPattern } = tileConfig;

    return [
      new TileLayer({
        id: "isometric-tiles",
        // Data fetching - return the image URL directly
        // TileLayer will handle loading it
        getTileData: ({
          index,
        }: {
          index: { x: number; y: number; z: number };
        }) => {
          const { x, y, z } = index;

          // Calculate actual tile coordinates
          // At zoom 0, we show the base tiles (0-19, 0-19)
          // At higher zooms, we subdivide
          const scale = Math.pow(2, z);
          const baseTileX = Math.floor(x / scale);
          const baseTileY = Math.floor(y / scale);

          // Bounds check - only load tiles that exist
          if (
            baseTileX < 0 ||
            baseTileX >= tileConfig.gridWidth ||
            baseTileY < 0 ||
            baseTileY >= tileConfig.gridHeight
          ) {
            return Promise.resolve(null);
          }

          // Flip Y coordinate: deck.gl has (0,0) at bottom-left,
          // but our tiles have (0,0) at top-left (image convention)
          const flippedY = tileConfig.gridHeight - 1 - baseTileY;

          // For now, always use z=0 (native resolution tiles)
          // z=0 = max zoom in, higher z = more zoomed out
          const url = tileUrlPattern
            .replace("{z}", "0")
            .replace("{x}_{y}", `${baseTileX}_${flippedY}`);

          // Return the URL string - BitmapLayer will load the image
          return Promise.resolve(url);
        },

        // Tile bounds calculation
        tileSize,

        // Extent of the tileset
        extent: [extent.minX, extent.minY, extent.maxX, extent.maxY],

        // Min/max zoom levels
        minZoom: 0,
        maxZoom: 0, // We only have base tiles for now

        // Refinement strategy - keep parent tiles visible while loading
        refinementStrategy: "best-available",

        // Cache settings
        maxCacheSize: 200,
        maxCacheByteSize: 512 * 1024 * 1024, // 512MB

        // Render each tile as a BitmapLayer
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        renderSubLayers: (props: any) => {
          // Destructure to exclude 'data' from being spread to BitmapLayer
          // BitmapLayer expects 'data' to be an array, not a string URL
          const { data, tile, ...layerProps } = props;
          const { left, bottom, right, top } = tile.bbox;

          // Flip the image vertically by swapping top and bottom in bounds
          // This corrects for image Y-axis (top-down) vs deck.gl Y-axis (bottom-up)
          const bounds: [number, number, number, number] = [
            left,
            top,
            right,
            bottom,
          ];

          if (!data) {
            // No image data - render a placeholder
            return new BitmapLayer({
              ...layerProps,
              image: createPlaceholderImage(tile.index.x, tile.index.y),
              bounds,
            });
          }

          // data is the image URL
          return new BitmapLayer({
            ...layerProps,
            image: data,
            bounds,
            // Pixel-perfect rendering (no interpolation)
            textureParameters: {
              minFilter: "nearest",
              magFilter: "nearest",
            },
          });
        },

        // Handle tile hover
        onHover: (info: { tile?: { index: { x: number; y: number } } }) => {
          if (info.tile) {
            onTileHover({ x: info.tile.index.x, y: info.tile.index.y });
          } else {
            onTileHover(null);
          }
        },
      }),
    ];
  }, [tileConfig, extent, onTileHover]);

  // Handle view state changes with constraints
  const handleViewStateChange = useCallback(
    (params: { viewState: ViewState }) => {
      const newViewState = { ...params.viewState };

      // Constrain zoom
      newViewState.zoom = Math.max(
        minZoom,
        Math.min(maxZoom, newViewState.zoom)
      );

      // Optionally constrain pan to keep map in view
      // (disabled for now to allow free exploration)

      onViewStateChange({ viewState: newViewState });
    },
    [maxZoom, minZoom, onViewStateChange]
  );

  return (
    <div className="map-container">
      <DeckGL
        views={
          new OrthographicView({
            id: "ortho",
            flipY: false, // Y increases upward
          })
        }
        viewState={viewState}
        onViewStateChange={handleViewStateChange}
        layers={layers}
        controller={{
          scrollZoom: { speed: 0.01, smooth: true },
          dragPan: true,
          dragRotate: false,
          doubleClickZoom: true,
          touchZoom: true,
          touchRotate: false,
          keyboard: true,
          inertia: 300,
        }}
        getCursor={() => "grab"}
      />
    </div>
  );
}

// Create a placeholder image for missing tiles
function createPlaceholderImage(x: number, y: number): HTMLCanvasElement {
  const size = 512;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;

  // Background - subtle grid pattern
  ctx.fillStyle = "#1a1e2e";
  ctx.fillRect(0, 0, size, size);

  // Grid lines
  ctx.strokeStyle = "#232838";
  ctx.lineWidth = 1;

  // Isometric grid pattern
  const gridSize = 32;
  for (let i = 0; i <= size; i += gridSize) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(size, i);
    ctx.stroke();
  }

  // Diamond pattern overlay (isometric feel)
  ctx.strokeStyle = "rgba(123, 104, 238, 0.1)";
  ctx.lineWidth = 1;
  for (let i = -size; i <= size * 2; i += 64) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i + size, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(i + size, 0);
    ctx.lineTo(i, size);
    ctx.stroke();
  }

  // Tile coordinates label
  ctx.fillStyle = "#5a5958";
  ctx.font = '500 24px "Azeret Mono", monospace';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`${x}, ${y}`, size / 2, size / 2);

  // Border
  ctx.strokeStyle = "rgba(255, 107, 53, 0.2)";
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, size - 2, size - 2);

  return canvas;
}
