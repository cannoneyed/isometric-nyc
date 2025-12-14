import { useEffect, useRef, useCallback } from "react";
import OpenSeadragon from "openseadragon";
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
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const isUpdatingFromOSD = useRef(false);
  const isUpdatingFromProps = useRef(false);

  const { gridWidth, gridHeight, tileSize, maxZoomLevel } = tileConfig;

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

  // Initialize OpenSeadragon
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    // Calculate initial OSD viewport from our view state
    const { centerX, centerY, zoom: initialZoom } = worldToOsd(viewState);

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
      // Custom tile source with multi-level pyramid support
      // OSD pyramid: level 0 = least detail (few tiles), maxLevel = most detail (many tiles)
      // Our export: level 0 = most detail (128×128), level 4 = least detail (8×8)
      // So we invert: ourLevel = maxZoomLevel - osdLevel
      tileSources: {
        width: totalWidth,
        height: totalHeight,
        tileSize: tileSize,
        tileOverlap: 0,
        minLevel: 0,
        maxLevel: maxZoomLevel,
        getTileUrl: (level: number, x: number, y: number) => {
          // Invert level mapping: OSD level 0 -> our level maxZoomLevel
          const ourLevel = maxZoomLevel - level;

          // Calculate grid dimensions at this level
          // At our level 0: 128×128, level 1: 64×64, level 2: 32×32, etc.
          const scale = Math.pow(2, ourLevel);
          const levelGridWidth = Math.ceil(gridWidth / scale);
          const levelGridHeight = Math.ceil(gridHeight / scale);

          // Bounds check for this level
          if (x < 0 || x >= levelGridWidth || y < 0 || y >= levelGridHeight) {
            return "";
          }

          return `/tiles/${ourLevel}/${x}_${y}.png`;
        },
      },
    });

    // Set initial viewport position
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

    const handleMouseMove = (event: OpenSeadragon.ViewerEvent) => {
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
    viewer.innerTracker.moveHandler = (event) => {
      handleMouseMove(event as unknown as OpenSeadragon.ViewerEvent);
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
