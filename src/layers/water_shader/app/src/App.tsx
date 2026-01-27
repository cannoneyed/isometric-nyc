import { useState, useEffect, useCallback } from "react";
import { WaterTile } from "./components/WaterTile";
import { ControlPanel } from "./components/ControlPanel";
import "./App.css";

export interface ShaderParams {
	waveSpeed: number;
	waveFrequency: number;
	foamThreshold: number;
	pixelSize: number;
	rippleDarkness: number;
	waterDarkness: number;
}

export interface TileCoords {
	x: number;
	y: number;
}

const defaultParams: ShaderParams = {
	waveSpeed: 2.0,
	waveFrequency: 10.0,
	foamThreshold: 0.8,
	pixelSize: 256.0,
	rippleDarkness: 0.12,
	waterDarkness: 0.0,
};

// Parse coordinates from URL query params
function getCoordsFromUrl(): TileCoords {
	const params = new URLSearchParams(window.location.search);
	const x = parseInt(params.get("x") || "0", 10);
	const y = parseInt(params.get("y") || "0", 10);
	return { x: Number.isNaN(x) ? 0 : x, y: Number.isNaN(y) ? 0 : y };
}

// Update URL with new coordinates
function updateUrlCoords(coords: TileCoords): void {
	const params = new URLSearchParams(window.location.search);
	params.set("x", coords.x.toString());
	params.set("y", coords.y.toString());
	const newUrl = `${window.location.pathname}?${params.toString()}`;
	window.history.replaceState({}, "", newUrl);
}

// API base - with Vite proxy configured, we can always use relative paths
function getApiBase(): string {
	return "";
}

function App() {
	const [shaderParams, setShaderParams] = useState<ShaderParams>(defaultParams);
	const [maskOpacity, setMaskOpacity] = useState(0);
	const [coords, setCoords] = useState<TileCoords>(getCoordsFromUrl);
	const [tileUrl, setTileUrl] = useState<string | null>(null);
	const [maskUrl, setMaskUrl] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [availableTiles, setAvailableTiles] = useState<[number, number][]>([]);
	const [availableMasks, setAvailableMasks] = useState<[number, number][]>([]);

	const apiBase = getApiBase();

	// Fetch available tiles and masks on mount
	useEffect(() => {
		async function fetchAvailable() {
			try {
				const [tilesRes, masksRes] = await Promise.all([
					fetch(`${apiBase}/api/available-tiles`),
					fetch(`${apiBase}/api/available-masks`),
				]);

				if (tilesRes.ok) {
					const tilesData = await tilesRes.json();
					setAvailableTiles(tilesData.tiles || []);
				}

				if (masksRes.ok) {
					const masksData = await masksRes.json();
					setAvailableMasks(masksData.masks || []);
				}
			} catch (e) {
				console.warn("Could not fetch available tiles/masks:", e);
			}
		}

		fetchAvailable();
	}, [apiBase]);

	// Load tile and mask when coordinates change
	const loadTileAndMask = useCallback(
		async (newCoords: TileCoords) => {
			setLoading(true);
			setError(null);

			const tileApiUrl = `${apiBase}/api/tile/${newCoords.x}/${newCoords.y}`;
			const maskApiUrl = `${apiBase}/api/mask/${newCoords.x}/${newCoords.y}`;

			try {
				// Check if tile exists
				const tileRes = await fetch(tileApiUrl);
				if (!tileRes.ok) {
					// Check if response is JSON before parsing
					const contentType = tileRes.headers.get("content-type");
					if (contentType?.includes("application/json")) {
						const err = await tileRes.json();
						throw new Error(
							err.error || `Tile not found at (${newCoords.x}, ${newCoords.y})`,
						);
					} else {
						// Backend probably isn't running - got HTML instead of JSON
						throw new Error(
							"Backend server not running. Start it with: uv run python src/water_shader_demo/server.py --mask_dir <mask_dir> --generations_dir <gen_dir>",
						);
					}
				}
				setTileUrl(tileApiUrl);

				// Check if mask exists (optional, don't fail if missing)
				const maskRes = await fetch(maskApiUrl, { method: "HEAD" });
				if (maskRes.ok) {
					setMaskUrl(maskApiUrl);
				} else {
					setMaskUrl(null);
					console.warn(
						`No mask available for (${newCoords.x}, ${newCoords.y})`,
					);
				}

				setError(null);
			} catch (e) {
				if (e instanceof TypeError && e.message.includes("fetch")) {
					setError(
						"Cannot connect to backend. Make sure the server is running on port 5001.",
					);
				} else {
					setError(e instanceof Error ? e.message : "Failed to load tile");
				}
				setTileUrl(null);
				setMaskUrl(null);
			} finally {
				setLoading(false);
			}
		},
		[apiBase],
	);

	// Load tile when coords change
	useEffect(() => {
		loadTileAndMask(coords);
	}, [coords, loadTileAndMask]);

	// Handle coordinate change from control panel
	const handleCoordsChange = useCallback((newCoords: TileCoords) => {
		setCoords(newCoords);
		updateUrlCoords(newCoords);
	}, []);

	// Handle browser back/forward
	useEffect(() => {
		const handlePopState = () => {
			setCoords(getCoordsFromUrl());
		};
		window.addEventListener("popstate", handlePopState);
		return () => window.removeEventListener("popstate", handlePopState);
	}, []);

	return (
		<div className="app">
			<header className="app-header">
				<h1>Water Shader Demo</h1>
				<p className="subtitle">
					Isometric NYC • Tile ({coords.x}, {coords.y})
				</p>
			</header>

			<main className="app-main">
				<div className="tile-container">
					{loading && (
						<div className="loading-overlay">
							<div className="loading-spinner" />
							<p>Loading tile...</p>
						</div>
					)}
					{error && (
						<div className="error-overlay">
							<p className="error-message">⚠️ {error}</p>
							<p className="error-hint">
								Try different coordinates or check that the backend is running.
							</p>
						</div>
					)}
					{!loading && !error && tileUrl && (
						<WaterTile
							size={1024}
							imageSrc={tileUrl}
							maskSrc={maskUrl || undefined}
							shaderParams={shaderParams}
							maskOpacity={maskOpacity}
						/>
					)}
				</div>
			</main>

			<aside className="app-sidebar">
				<ControlPanel
					params={shaderParams}
					onParamsChange={setShaderParams}
					maskOpacity={maskOpacity}
					onMaskOpacityChange={setMaskOpacity}
					coords={coords}
					onCoordsChange={handleCoordsChange}
					availableTiles={availableTiles}
					availableMasks={availableMasks}
					hasMask={!!maskUrl}
				/>
			</aside>
		</div>
	);
}

export default App;
