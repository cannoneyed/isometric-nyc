/**
 * Generate placeholder tiles for testing the viewer.
 * Creates a 20x20 grid of 512x512 tiles with isometric-style placeholders.
 *
 * Run with: bun run generate-tiles
 *
 * For browser-based placeholder generation, the viewer creates placeholders
 * on-the-fly when tiles are missing.
 */

import { mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const GRID_SIZE = 20;
const TILE_SIZE = 512;
const OUTPUT_DIR = join(import.meta.dir, "..", "public", "tiles", "0");

// This is a placeholder script - actual tile generation would use canvas/sharp
// For now, we just create the directory structure
console.log("Creating tile directory structure...");

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Create a simple info file
const info = {
  gridSize: GRID_SIZE,
  tileSize: TILE_SIZE,
  totalTiles: GRID_SIZE * GRID_SIZE,
  generated: new Date().toISOString(),
  urlPattern: "{z}/{x}_{y}.png",
  zoomConvention:
    "z=0 is native resolution (max zoom in), higher z = more zoomed out",
  note: "Placeholder tiles are generated client-side when images are missing",
};

const infoPath = join(import.meta.dir, "..", "public", "tiles", "info.json");
await Bun.write(infoPath, JSON.stringify(info, null, 2));

console.log(`Created tile info at public/tiles/info.json`);
console.log(`Tile directory ready at ${OUTPUT_DIR}`);
console.log("\nNote: The viewer generates placeholder tiles client-side.");
console.log("To add real tiles, place them at: public/tiles/0/{x}_{y}.png");
console.log("(z=0 is native resolution / max zoom in)");
