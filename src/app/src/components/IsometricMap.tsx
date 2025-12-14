import { useMemo, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import { OrthographicView } from "@deck.gl/core";
import { TileLayer } from "@deck.gl/geo-layers";
import { BitmapLayer } from "@deck.gl/layers";
import type { ViewState } from "../App";
import { ScanlineOverlay } from "./ScanlineOverlay";
import { WaterShaderOverlay } from "./WaterShaderOverlay";
import type { ShaderParams } from "../shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  tileSize: number;
  tileUrlPattern: string;
  maxZoomLevel: number;
}

interface ScanlineSettings {
  enabled: boolean;
  count: number;
  opacity: number;
}

interface WaterShaderSettings {
  enabled: boolean;
  showMask: boolean;
  params: ShaderParams;
}

interface IsometricMapProps {
  tileConfig: TileConfig;
  viewState: ViewState;
  onViewStateChange: (params: { viewState: ViewState }) => void;
  lightDirection: [number, number, number];
  onTileHover: (tile: { x: number; y: number } | null) => void;
  scanlines?: ScanlineSettings;
  waterShader?: WaterShaderSettings;
}

export function IsometricMap({
  tileConfig,
  viewState,
  onViewStateChange,
  onTileHover,
  scanlines = { enabled: true, count: 480, opacity: 0.15 },
  waterShader,
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

  // Create the tile layer - use base tiles and let deck.gl handle zoom
  const layers = useMemo(() => {
    const { tileSize, tileUrlPattern, gridWidth, gridHeight } = tileConfig;

    console.log(
      `Creating TileLayer: grid=${gridWidth}x${gridHeight}, tileSize=${tileSize}, extent=[0,0,${extent.maxX},${extent.maxY}]`
    );

    return [
      new TileLayer({
        id: "isometric-tiles",

        // Use TileLayer's built-in data prop pattern for simpler setup
        // Data URL pattern with {x}, {y}, {z} placeholders
        data: tileUrlPattern.replace("{x}_{y}", "{x}_{y}"),

        // Custom getTileData to handle our file structure
        getTileData: ({
          index,
          signal,
        }: {
          index: { x: number; y: number; z: number };
          signal?: AbortSignal;
        }) => {
          const { x, y, z } = index;

          console.log(
            `getTileData: x=${x}, y=${y}, z=${z}, gridWidth=${gridWidth}, gridHeight=${gridHeight}`
          );

          // Always use z=0 tiles, but scale the coordinates for higher z values
          // At z=0: full resolution, 1 tile = 1 file tile
          // At z=1: 2x zoom out, but we still use z=0 files
          // The TileLayer handles the visual scaling

          // For now, always use z=0 tiles
          // Bounds check against the grid at z=0
          if (x < 0 || x >= gridWidth || y < 0 || y >= gridHeight) {
            console.log(`Out of bounds: (${x},${y})`);
            return Promise.resolve(null);
          }

          // Flip Y coordinate: deck.gl has (0,0) at bottom-left,
          // but our tiles have (0,0) at top-left (image convention)
          const flippedY = gridHeight - 1 - y;

          const url = `/tiles/0/${x}_${flippedY}.png`;

          console.log(`Fetching: ${url}`);

          // Fetch the image
          return fetch(url, { signal })
            .then((response) => {
              if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
              }
              return response.blob();
            })
            .then((blob) => createImageBitmap(blob))
            .catch((err) => {
              if (err.name !== "AbortError") {
                console.warn(`Failed to load tile ${url}: ${err}`);
              }
              return null;
            });
        },

        // Tile size in pixels
        tileSize,

        // Extent of the tileset (in world coordinates)
        extent: [extent.minX, extent.minY, extent.maxX, extent.maxY],

        // Allow any zoom level - we'll always serve z=0 tiles
        // Setting these helps deck.gl know what tile indices to request
        minZoom: 0,
        maxZoom: 8, // High value to allow zooming in

        // Refinement strategy
        refinementStrategy: "best-available",

        // Cache settings
        maxCacheSize: 200,
        maxCacheByteSize: 512 * 1024 * 1024, // 512MB

        // Render each tile as a BitmapLayer
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        renderSubLayers: (props: any) => {
          const { data, tile, ...layerProps } = props;
          const { left, bottom, right, top } = tile.bbox;

          // Flip the image vertically by swapping top and bottom in bounds
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

        // Enable update triggers
        updateTriggers: {
          getTileData: [tileUrlPattern, gridWidth, gridHeight],
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
      {waterShader && (
        <WaterShaderOverlay
          enabled={waterShader.enabled}
          viewState={viewState}
          tileConfig={tileConfig}
          shaderParams={waterShader.params}
          showMask={waterShader.showMask}
        />
      )}
      <ScanlineOverlay
        enabled={scanlines.enabled}
        scanlineCount={scanlines.count}
        scanlineOpacity={scanlines.opacity}
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
