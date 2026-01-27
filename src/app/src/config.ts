// Worker URL for DZI tiles (adds caching headers to R2)
const R2_TILES_URL = "https://isometric-nyc-tiles.cannoneyed.com";

// Default export directory name (subdirectory under public/dzi/)
const DEFAULT_EXPORT_DIR = "tiny-nyc";

// Check URL parameters
function getUrlParams(): URLSearchParams {
	return new URLSearchParams(window.location.search);
}

// Get the export directory name
// Priority: USE_R2_NYC env var > MAP_ID env var > ?export= URL param > default
// - USE_R2_NYC: fetches from R2 dzi/ directly (no map id subdir)
// - MAP_ID: set via `MAP_ID=tiny-nyc bun run dev`
// - ?export: use URL param like ?export=tiny-nyc
function getExportDir(): string {
	// Check USE_R2_NYC first - uses dzi/ directly without map id subdir
	if (__USE_R2_NYC__) {
		console.log("📂 Using map: dzi (via USE_R2_NYC env var)");
		return "dzi";
	}

	// Check MAP_ID env var first (set at build/dev time)
	if (__MAP_ID__) {
		console.log(`📂 Using map: dzi/${__MAP_ID__} (via MAP_ID env var)`);
		return `dzi/${__MAP_ID__}`;
	}

	// Check URL param override
	const params = getUrlParams();
	const exportParam = params.get("export");
	if (exportParam) {
		console.log(`📂 Using map: dzi/${exportParam} (via ?export=${exportParam})`);
		return `dzi/${exportParam}`;
	}

	// Fall back to default
	console.log(`📂 Using map: dzi/${DEFAULT_EXPORT_DIR} (default)`);
	return `dzi/${DEFAULT_EXPORT_DIR}`;
}

// Get the base URL for tiles
// - In production: always use R2
// - In dev: use local by default, or R2 if ?r2=true or LOCAL_R2=true or USE_R2_NYC=true
function getTilesBaseUrl(): string {
	const params = getUrlParams();

	// Check for USE_R2_NYC env var (fetches from R2 dzi/ directly)
	if (__USE_R2_NYC__) {
		console.log("📡 Using R2 tiles (via USE_R2_NYC env var)");
		return R2_TILES_URL;
	}

	// Check for R2 override via env var (LOCAL_R2=true bun run dev)
	if (__LOCAL_R2__) {
		console.log("📡 Using R2 tiles (via LOCAL_R2 env var)");
		return R2_TILES_URL;
	}

	// Check for R2 override in dev mode via query param
	if (params.get("r2") === "true") {
		console.log("📡 Using R2 tiles (via ?r2=true)");
		return R2_TILES_URL;
	}

	// Use the build-time configured URL
	return __TILES_BASE_URL__;
}

// Determine if we should show debug UI
// Hidden by default, shown only with ?debug=true
function shouldShowDebugUI(): boolean {
	const params = getUrlParams();
	return params.get("debug") === "true";
}

// Export computed values
export const tilesBaseUrl = getTilesBaseUrl();
export const exportDir = getExportDir();
export const showDebugUI = shouldShowDebugUI();

