import { useEffect, useRef, useCallback } from "react";
import OpenSeadragon from "openseadragon";
import { PMTiles } from "pmtiles";
import type { ViewState } from "../App";
import { ScanlineOverlay } from "./ScanlineOverlay";
import { WaterShaderOverlay } from "./WaterShaderOverlay";
import type { ShaderParams } from "../shaders/water";

interface TileConfig {
  gridWidth: number;
  gridHeight: number;
  originalWidth: number;
  originalHeight: number;
  tileSize: number;
  maxZoomLevel: number;
  pmtilesUrl?: string; // URL to PMTiles file (optional, falls back to tile directory)
  pmtilesZoomMap?: Record<number, number>; // Maps our level -> PMTiles z
  tileUrlPattern?: string; // Legacy: URL pattern for individual tiles
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
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const pmtilesRef = useRef<PMTiles | null>(null);
  const isUpdatingFromOSD = useRef(false);
  const isUpdatingFromProps = useRef(false);

  const {
    gridWidth,
    gridHeight,
    tileSize,
    maxZoomLevel,
    pmtilesUrl,
    pmtilesZoomMap,
  } = tileConfig;

  // Total image dimensions in pixels
  const totalWidth = gridWidth * tileSize;
  const totalHeight = gridHeight * tileSize;

  // Convert our view state to OSD viewport coordinates
  // Our viewState: { target: [worldX, worldY, 0], zoom: log2Scale }
  // OSD viewport: center is in image coordinates (0-1 for x, 0-aspectRatio for y)
  const worldToOsd = useCallback(
    (vs: ViewState) => {
      // Our target is in world pixels, convert to normalized coordinates
      // Note: Our Y=0 is at bottom, OSD Y=0 is at top
      const centerX = vs.target[0] / totalWidth;
      const centerY = 1 - vs.target[1] / totalHeight;

      // Our zoom: 0 = 1:1 pixels, positive = zoom in, negative = zoom out
      // OSD zoom: 1 = fit width in viewport
      // At zoom=0, we want 1 pixel = 1 pixel on screen
      // So OSD zoom = viewport_width / total_width * 2^ourZoom
      const scale = Math.pow(2, vs.zoom);
      const osdZoom = (window.innerWidth / totalWidth) * scale;

      return { centerX, centerY, zoom: osdZoom };
    },
    [totalWidth, totalHeight]
  );

  // Convert OSD viewport to our view state
  const osdToWorld = useCallback(
    (viewer: OpenSeadragon.Viewer): ViewState => {
      const viewport = viewer.viewport;
      const center = viewport.getCenter();
      const osdZoom = viewport.getZoom();

      // Convert normalized coordinates back to world pixels
      // OSD Y is top-down, ours is bottom-up
      const worldX = center.x * totalWidth;
      const worldY = (1 - center.y) * totalHeight;

      // Convert OSD zoom to our zoom
      // osdZoom = (windowWidth / totalWidth) * 2^ourZoom
      // ourZoom = log2(osdZoom * totalWidth / windowWidth)
      const ourZoom = Math.log2((osdZoom * totalWidth) / window.innerWidth);

      return {
        target: [worldX, worldY, 0],
        zoom: ourZoom,
      };
    },
    [totalWidth, totalHeight]
  );

  // Initialize PMTiles source if URL is provided
  useEffect(() => {
    if (pmtilesUrl && !pmtilesRef.current) {
      pmtilesRef.current = new PMTiles(pmtilesUrl);
    }
    return () => {
      pmtilesRef.current = null;
    };
  }, [pmtilesUrl]);

  // Initialize OpenSeadragon
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    // Calculate initial OSD viewport from our view state
    const { centerX, centerY, zoom: initialZoom } = worldToOsd(viewState);

    // Create tile source configuration
    // OSD pyramid: level 0 = least detail (few tiles), maxLevel = most detail (many tiles)
    // Our export: level 0 = most detail (128×128), level 4 = least detail (8×8)
    // PMTiles z: matches our level (0 = most detail when stored, but we flip it)
    // So we invert: ourLevel = maxZoomLevel - osdLevel
    const tileSourceConfig: OpenSeadragon.TileSourceOptions = {
      width: totalWidth,
      height: totalHeight,
      tileSize: tileSize,
      tileOverlap: 0,
      minLevel: 0,
      maxLevel: maxZoomLevel,
    };

    // If using PMTiles, we need a custom tile loading approach
    if (pmtilesUrl) {
      // For PMTiles, we use getTileUrl to generate a virtual URL
      // and downloadTileStart/downloadTileAbort to handle actual loading
      Object.assign(tileSourceConfig, {
        getTileUrl: (level: number, x: number, y: number) => {
          // Return a virtual URL that encodes the tile coordinates
          // This will be parsed by our custom tile loading
          const ourLevel = maxZoomLevel - level;

          // Get the PMTiles zoom level from the map, or calculate it
          let pmtilesZ: number;
          if (pmtilesZoomMap && pmtilesZoomMap[ourLevel] !== undefined) {
            pmtilesZ = pmtilesZoomMap[ourLevel];
          } else {
            // Fallback: calculate based on grid size
            const scale = Math.pow(2, ourLevel);
            const maxDim = Math.max(
              Math.ceil(gridWidth / scale),
              Math.ceil(gridHeight / scale)
            );
            pmtilesZ = maxDim <= 1 ? 0 : Math.ceil(Math.log2(maxDim));
          }

          return `pmtiles://${pmtilesZ}/${x}/${y}`;
        },
      });
    } else {
      // Legacy file-based tiles
      Object.assign(tileSourceConfig, {
        getTileUrl: (level: number, x: number, y: number) => {
          // Invert level mapping: OSD level 0 -> our level maxZoomLevel
          const ourLevel = maxZoomLevel - level;

          // Calculate grid dimensions at this level
          const scale = Math.pow(2, ourLevel);
          const levelGridWidth = Math.ceil(gridWidth / scale);
          const levelGridHeight = Math.ceil(gridHeight / scale);

          // Bounds check for this level
          if (x < 0 || x >= levelGridWidth || y < 0 || y >= levelGridHeight) {
            return "";
          }

          return `${__TILES_BASE_URL__}/tiles/${ourLevel}/${x}_${y}.png`;
        },
      });
    }

    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "",
      showNavigationControl: false,
      showNavigator: false,
      animationTime: 0.15,
      blendTime: 0.1,
      minZoomImageRatio: 0.1,
      maxZoomPixelRatio: 16,
      visibilityRatio: 0.2,
      constrainDuringPan: false,
      gestureSettingsMouse: {
        scrollToZoom: true,
        clickToZoom: false,
        dblClickToZoom: true,
        flickEnabled: true,
      },
      gestureSettingsTouch: {
        scrollToZoom: false,
        clickToZoom: false,
        dblClickToZoom: true,
        flickEnabled: true,
        pinchToZoom: true,
      },
      // Disable image smoothing for pixel art
      imageSmoothingEnabled: false,
      tileSources: tileSourceConfig,
    });

    // If using PMTiles, set up custom tile downloading
    if (pmtilesUrl && pmtilesRef.current) {
      const pmtiles = pmtilesRef.current;

      // Override the tile downloading for PMTiles
      viewer.addHandler("open", () => {
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          tiledImage.setCompositeOperation("source-over");

          // Get the tile source and override its download method
          const source = tiledImage.source as OpenSeadragon.TileSource;

          // Store original methods
          const originalDownloadTileStart = (
            source as unknown as Record<string, unknown>
          ).downloadTileStart;

          // Override downloadTileStart to use PMTiles
          (source as unknown as Record<string, unknown>).downloadTileStart = (
            context: {
              src: string;
              finish: (
                data: HTMLImageElement | null,
                request: null,
                errorMsg?: string
              ) => void;
            }
          ) => {
            const url = context.src;

            // Check if this is a PMTiles URL
            if (url.startsWith("pmtiles://")) {
              const parts = url.replace("pmtiles://", "").split("/");
              const z = parseInt(parts[0], 10);
              const x = parseInt(parts[1], 10);
              const y = parseInt(parts[2], 10);

              // Fetch tile from PMTiles
              pmtiles
                .getZxy(z, x, y)
                .then((response) => {
                  if (response && response.data) {
                    // Convert ArrayBuffer to Blob URL
                    const blob = new Blob([response.data], {
                      type: "image/png",
                    });
                    const blobUrl = URL.createObjectURL(blob);

                    // Create an image element
                    const img = new Image();
                    img.onload = () => {
                      URL.revokeObjectURL(blobUrl);
                      context.finish(img, null);
                    };
                    img.onerror = () => {
                      URL.revokeObjectURL(blobUrl);
                      context.finish(null, null, "Failed to load tile image");
                    };
                    img.src = blobUrl;
                  } else {
                    // Tile not found, return null (transparent)
                    context.finish(null, null);
                  }
                })
                .catch((err) => {
                  console.error("PMTiles fetch error:", err);
                  context.finish(null, null, String(err));
                });
            } else if (typeof originalDownloadTileStart === "function") {
              // Fall back to original method for non-PMTiles URLs
              originalDownloadTileStart.call(source, context);
            }
          };
        }
      });
    }

    // Set initial viewport position (only for non-PMTiles mode,
    // PMTiles mode sets this in its own "open" handler above)
    if (!pmtilesUrl) {
      viewer.addHandler("open", () => {
        // Disable interpolation for crisp pixels
        const tiledImage = viewer.world.getItemAt(0);
        if (tiledImage) {
          tiledImage.setCompositeOperation("source-over");
        }

        // Set initial position
        viewer.viewport.zoomTo(initialZoom, undefined, true);
        viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), true);
      });
    } else {
      // For PMTiles, add another handler for initial position
      // (the tile downloading setup handler already runs first)
      viewer.addHandler("open", () => {
        viewer.viewport.zoomTo(initialZoom, undefined, true);
        viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), true);
      });
    }

    // Track viewport changes
    viewer.addHandler("viewport-change", () => {
      if (isUpdatingFromProps.current) return;

      isUpdatingFromOSD.current = true;
      const newViewState = osdToWorld(viewer);
      onViewStateChange({ viewState: newViewState });
      isUpdatingFromOSD.current = false;
    });

    // Track mouse position for tile hover
    viewer.addHandler("canvas-exit", () => {
      onTileHover(null);
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseMove = (event: any) => {
      if (!event.position) return;
      const pos = event.position as OpenSeadragon.Point;

      const viewportPoint = viewer.viewport.pointFromPixel(pos);
      const imagePoint =
        viewer.viewport.viewportToImageCoordinates(viewportPoint);

      const tileX = Math.floor(imagePoint.x / tileSize);
      const tileY = Math.floor(imagePoint.y / tileSize);

      if (tileX >= 0 && tileX < gridWidth && tileY >= 0 && tileY < gridHeight) {
        onTileHover({ x: tileX, y: tileY });
      } else {
        onTileHover(null);
      }
    };

    viewer.addHandler("canvas-drag", handleMouseMove);
    viewer.addHandler("canvas-scroll", handleMouseMove);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (viewer as any).innerTracker.moveHandler = (event: any) => {
      handleMouseMove(event);
    };

    viewerRef.current = viewer;

    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [
    gridWidth,
    gridHeight,
    tileSize,
    maxZoomLevel,
    totalWidth,
    totalHeight,
    worldToOsd,
    osdToWorld,
    onViewStateChange,
    onTileHover,
    pmtilesUrl,
    pmtilesZoomMap,
  ]);

  // Sync external view state changes to OSD
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !viewer.viewport || isUpdatingFromOSD.current) return;

    isUpdatingFromProps.current = true;

    const { centerX, centerY, zoom } = worldToOsd(viewState);

    viewer.viewport.zoomTo(zoom, undefined, false);
    viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), false);

    isUpdatingFromProps.current = false;
  }, [viewState, worldToOsd]);

  return (
    <div className="map-container">
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          background: "#0a0c14",
        }}
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
