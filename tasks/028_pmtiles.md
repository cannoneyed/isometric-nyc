# pmtiles file format

Can we refactor our tile export process in
src/isometric_nyc/generation/export_tiles_for_app.py and the web app tile
viewer in src/app to use pmtiles file format? We need to account for our zoom
levels and postprocessing

## Status: ✅ Completed

## Implementation Summary

### Python Export (export_pmtiles.py)

New script `src/isometric_nyc/generation/export_pmtiles.py` that:

- Reads tiles from the quadrants.db SQLite database
- Applies the same postprocessing (pixelation, color quantization) as the
  original export
- Creates a single `.pmtiles` archive containing all zoom levels (0-4)
- Stores metadata in the PMTiles JSON section for the web app to read

Usage:

```bash
# Export all tiles to PMTiles
uv run python src/isometric_nyc/generation/export_pmtiles.py generations/v01

# Export without postprocessing
uv run python src/isometric_nyc/generation/export_pmtiles.py generations/v01 --no-postprocess

# Custom output path
uv run python src/isometric_nyc/generation/export_pmtiles.py generations/v01 -o my-tiles.pmtiles
```

### Web App (React/OpenSeadragon)

Updated `src/app/src/App.tsx` and `src/app/src/components/IsometricMap.tsx` to:

- First attempt to load from `/tiles.pmtiles`
- Fall back to legacy `/tiles/manifest.json` + directory structure if PMTiles
  not available
- Read tile metadata (gridWidth, gridHeight, etc.) from PMTiles metadata section
- Use custom tile loading in OpenSeadragon to fetch tiles from PMTiles via the
  `pmtiles` npm package

### Dependencies Added

- Python: `pmtiles` package (3.5.0)
- JavaScript: `pmtiles` package (4.3.0)

### Backward Compatibility

The implementation maintains full backward compatibility:

- The existing `export_tiles_for_app.py` script still works
- The web app automatically detects whether to use PMTiles or legacy tiles
- No changes required to existing deployments
