import { useEffect, useRef, useCallback } from "react";
import OpenSeadragon from "openseadragon";
import type { ViewState } from "../App";
import { WaterShaderOverlay } from "./WaterShaderOverlay";
import type { ShaderParams } from "../shaders/water";
import { tilesBaseUrl } from "../config";

// Debounced debug logger for viewport changes (only when ?debug=true)
let debugLogTimeout: ReturnType<typeof setTimeout> | null = null;
const isDebugMode =
	new URLSearchParams(window.location.search).get("debug") === "true";
function logViewportDebug(viewState: ViewState): void {
	if (!isDebugMode) return;
	if (debugLogTimeout) {
		clearTimeout(debugLogTimeout);
	}
	debugLogTimeout = setTimeout(() => {
		console.log(
			`📍 Viewport: x=${Math.round(viewState.target[0])}, y=${Math.round(viewState.target[1])}, zoom=${viewState.zoom.toFixed(2)}`,
		);
	}, 200);
}

interface TileConfig {
	gridWidth: number;
	gridHeight: number;
	originalWidth: number;
	originalHeight: number;
	tileSize: number;
	maxZoomLevel: number;
	// Origin offset: (0,0) corresponds to database (originX, originY)
	originX: number;
	originY: number;
	// DZI source (preferred - native OpenSeadragon support)
	dziUrl?: string;
	// Legacy file-based tiles
	tileUrlPattern?: string;
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
	waterShader?: WaterShaderSettings;
	showMinimap?: boolean; // Toggle minimap visibility
}

export function IsometricMap({
	tileConfig,
	viewState,
	onViewStateChange,
	onTileHover,
	waterShader,
	showMinimap = true,
}: IsometricMapProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
	const isUpdatingFromProps = useRef(false);
	const lastOSDViewState = useRef<ViewState | null>(null);

	const { gridWidth, gridHeight, tileSize, maxZoomLevel, dziUrl } = tileConfig;

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
		[totalWidth, totalHeight],
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
		[totalWidth, totalHeight],
	);

	// Initialize OpenSeadragon
	useEffect(() => {
		if (!containerRef.current || viewerRef.current) return;

		// Calculate initial OSD viewport from our view state
		const { centerX, centerY, zoom: initialZoom } = worldToOsd(viewState);

		// Determine tile source based on configuration priority:
		// 1. DZI (native OpenSeadragon support - simplest and most performant)
		// 2. Legacy file-based tiles
		let tileSourceConfig: OpenSeadragon.TileSourceOptions | string;

		if (dziUrl) {
			// DZI: Create tile source with level mapping
			// OpenSeadragon calculates levels based on image dimensions:
			// - maxLevel = ceil(log2(imageWidth)) = 17 for 123904px
			// - minLevel = ceil(log2(tileSize)) = 9 for 512px tiles
			// Our export creates levels 0-8, so we offset by -9
			const osdMaxLevel = Math.ceil(
				Math.log2(Math.max(totalWidth, totalHeight)),
			);
			const osdMinLevel = Math.ceil(Math.log2(tileSize));
			const levelOffset = osdMinLevel; // Our level 0 = OSD level 9

			tileSourceConfig = {
				width: totalWidth,
				height: totalHeight,
				tileSize: tileSize,
				tileOverlap: 0,
				minLevel: osdMinLevel,
				maxLevel: osdMaxLevel,
				getTileUrl: (level: number, x: number, y: number) => {
					// Map OSD level to our file level (offset by -9)
					const fileLevel = level - levelOffset;
					const dziBase = dziUrl.replace(".dzi", "_files");
					return `${dziBase}/${fileLevel}/${x}_${y}.webp`;
				},
			};
		} else {
			// Legacy file-based tiles
			tileSourceConfig = {
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
					const scale = Math.pow(2, ourLevel);
					const levelGridWidth = Math.ceil(gridWidth / scale);
					const levelGridHeight = Math.ceil(gridHeight / scale);

					// Bounds check for this level
					if (x < 0 || x >= levelGridWidth || y < 0 || y >= levelGridHeight) {
						return "";
					}

					return `${tilesBaseUrl}/tiles/${ourLevel}/${x}_${y}.png`;
				},
			};
		}

		// 1:1 pixel ratio for max zoom (no upscaling), 1:1 of top level image for min zoom
		const osdMaxZoomPixelRatio = 1; // 1 screen pixel per image pixel at max
		const osdMinZoomPixelRatio = 1;

		// Extended options not in @types/openseadragon but supported by the library
		const extendedOptions = {
			// Round tile positions to prevent subpixel aliasing during pan
			// SUBPIXEL_ROUNDING_OCCURRENCES.ALWAYS = 1
			subPixelRoundingForTransparency: 1,
			// Smooth tile edges only when zoomed out past 1:1 pixel ratio
			smoothTileEdgesMinZoom: 1.0,
		};

		const viewer = OpenSeadragon({
			element: containerRef.current,
			prefixUrl: "",
			showNavigationControl: false,
			// Allow loading tiles from R2 CDN (different origin)
			crossOriginPolicy: "Anonymous",
			// Minimap in top-right corner
			showNavigator: true,
			navigatorPosition: "TOP_RIGHT",
			navigatorSizeRatio: 0.15,
			navigatorMaintainSizeRatio: true,
			navigatorAutoFade: false,
			navigatorBackground: "#0a1525",
			animationTime: 0.15,
			blendTime: 0.1,
			minZoomImageRatio: osdMinZoomPixelRatio,
			maxZoomPixelRatio: osdMaxZoomPixelRatio,
			visibilityRatio: 0.2,
			constrainDuringPan: false,
			// Visible blue placeholder for loading tiles
			placeholderFillStyle: "rgba(35, 65, 105, 0.7)",
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
			// Disable image smoothing for pixel art (crisp pixels)
			imageSmoothingEnabled: false,
			tileSources: tileSourceConfig,
			...extendedOptions,
		} as OpenSeadragon.Options);

		// Set initial viewport position and configure tile display
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

			const newViewState = osdToWorld(viewer);
			// Track this viewState so sync useEffect knows to skip it
			lastOSDViewState.current = newViewState;
			onViewStateChange({ viewState: newViewState });

			// Debug logging (debounced)
			logViewportDebug(newViewState);
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
		dziUrl,
	]);

	// Sync external view state changes to OSD
	// Only sync if viewState came from an external source (not from OSD itself)
	useEffect(() => {
		const viewer = viewerRef.current;
		if (!viewer || !viewer.viewport) return;

		// Skip if this viewState originated from OSD (prevents feedback loop)
		const lastOSD = lastOSDViewState.current;
		if (
			lastOSD &&
			Math.abs(lastOSD.target[0] - viewState.target[0]) < 0.01 &&
			Math.abs(lastOSD.target[1] - viewState.target[1]) < 0.01 &&
			Math.abs(lastOSD.zoom - viewState.zoom) < 0.01
		) {
			return;
		}

		const { centerX, centerY, zoom } = worldToOsd(viewState);

		isUpdatingFromProps.current = true;
		viewer.viewport.zoomTo(zoom, undefined, false);
		viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), false);
		isUpdatingFromProps.current = false;
	}, [viewState, worldToOsd]);

	// Toggle minimap visibility
	useEffect(() => {
		const viewer = viewerRef.current;
		if (!viewer || !viewer.navigator) return;

		const navigatorElement = viewer.navigator.element;
		if (navigatorElement) {
			if (showMinimap) {
				navigatorElement.classList.remove("hidden");
			} else {
				navigatorElement.classList.add("hidden");
			}
		}
	}, [showMinimap]);

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
		</div>
	);
}
