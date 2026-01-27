# DZI or other image type

It seems like the app in @src/app is struggling to load pmtiles images manually using OpenSeaDragon... I think we might get better performance by using a different image type, such as DZI or another format that is more optimized for web use.

First, look at the docs on OpenSeaDragon for more information on how to use DZI images and determine what library would be best suited for our use case. Write a detailed report with a conclusion below:

---

## Research Report: PMTiles vs DZI for OpenSeadragon Performance

**Date:** 2026-01-20
**Status:** Research Complete

### Executive Summary

**Recommendation: Migrate from PMTiles to DZI format**

The current PMTiles implementation requires extensive custom code (500+ lines) to work with OpenSeadragon, while DZI is OpenSeadragon's native format requiring zero custom tile loading logic. Research shows DZI with libvips can provide 67-117ms faster tile display compared to custom tile sources.

### Current Architecture Analysis

The existing implementation in `src/app/src/components/IsometricMap.tsx` includes:

| Component | Lines | Purpose |
|-----------|-------|---------|
| LRU Tile Cache | 41-73 | 500-tile cache with manual eviction |
| Priority Queue | 136-257 | Viewport-aware tile prioritization |
| Request Throttling | - | Max 10 concurrent requests |
| Custom Tile Source | 341-548 | Virtual `pmtiles://` URL handling |
| Off-thread Decoding | - | `createImageBitmap()` for PNG decoding |

**Problem:** PMTiles was designed for map tiles (MapLibre GL, Leaflet), not general imagery. OpenSeadragon has no native PMTiles support, requiring all this custom code.

### Format Comparison

| Format | Native OSD Support | Single File | HTTP Range | Best For |
|--------|-------------------|-------------|------------|----------|
| **DZI** | ✅ Yes | ❌ No | ❌ No | Pre-generated image pyramids |
| **IIIF** | ✅ Yes | Either | Depends | Cultural heritage, interop |
| **PMTiles** | ❌ No | ✅ Yes | ✅ Yes | Map tiles (not general images) |
| **SZI** | Plugin | ✅ Yes | ✅ Yes | DZI in single ZIP file |
| **COG** | Plugin | ✅ Yes | ✅ Yes | Geospatial imagery |

### Why DZI is Recommended

1. **Native OpenSeadragon Support** - Zero custom tile loading code
2. **Proven Performance** - Research shows 375ms → 240ms field-of-view improvement
3. **Simple Static Hosting** - Works directly with Cloudflare R2 public access
4. **libvips Generation** - Fast, multi-threaded pyramid generation
5. **Industry Standard** - Widely used, well-documented

### Alternative Considered: SZI (Single-file Zipped DZI)

SZI bundles DZI as a single ZIP file with Range request access. This would preserve single-file deployment but requires a plugin and less mature ecosystem. **Verdict:** If single-file is critical, consider SZI, but standard DZI is simpler.

---

## Implementation Plan

### Phase 1: Validation & Prototyping

#### 1.1 Test libvips DZI Generation

**Goal:** Verify we can generate DZI from existing tile pipeline

**Challenge:** Current 128×144 tile grid at 512px = 65,536 × 73,728 pixels (~4.8 billion pixels). Direct assembly may exceed memory.

**Approach Options:**
- **Option A:** Use libvips streaming mode to avoid loading full image
- **Option B:** Generate DZI directly from tile database using custom script
- **Option C:** Process in quadrants and stitch pyramids

**Test Script:**
```python
# test_dzi_export.py
import pyvips

# Test with small subset first
image = pyvips.Image.new_from_file("test_source.png", access="sequential")
image.dzsave("output", tile_size=512, suffix=".webp[Q=85]")
```

**Files to create:**
- `src/isometric_nyc/generation/export_dzi.py`

#### 1.2 Benchmark Performance

**Test:**
1. Generate small DZI manually (~10×10 tile area)
2. Create test page with native OSD DZI source
3. Compare load times, memory usage, network requests

**Metrics to capture:**
- Time to first tile visible
- Time to full viewport rendered
- Memory usage (DevTools Performance)
- Network request count

### Phase 2: Export Pipeline

#### 2.1 Create DZI Export Script

**File:** `src/isometric_nyc/generation/export_dzi.py`

```python
# Pseudo-code structure
def export_dzi(db_path: str, output_dir: str, tile_size: int = 512):
    """
    Generate DZI pyramid from tile database.

    1. Load tiles from SQLite database
    2. Apply postprocessing (palette, quantization, bounds)
    3. Assemble into source image using pyvips
    4. Generate DZI pyramid with dzsave()
    5. Create metadata sidecar JSON
    """
    pass
```

**Key Requirements:**
- Preserve unified palette processing
- Apply bounds clipping (alpha mask for NYC outline)
- Support WebP output tiles (25-35% smaller than PNG)
- Generate metadata JSON with `originX`, `originY`, zoom mapping

#### 2.2 Metadata Sidecar File

DZI doesn't support custom metadata. Create `tiles_metadata.json`:

```json
{
  "originX": -79,
  "originY": -16,
  "gridWidth": 128,
  "gridHeight": 144,
  "tileSize": 512,
  "maxZoom": 4,
  "dziLevelMap": {
    "0": 8,
    "1": 7,
    "2": 6,
    "3": 5,
    "4": 4
  }
}
```

### Phase 3: Frontend Migration

#### 3.1 Update App.tsx

**Current (PMTiles):**
```typescript
const pmtiles = new PMTiles(pmtilesUrl);
pmtiles.getHeader().then(() => pmtiles.getMetadata()).then(metadata => {
  // Process metadata
});
```

**New (DZI):**
```typescript
// Fetch metadata sidecar
const metadata = await fetch(`${tilesBaseUrl}/tiles_metadata.json`).then(r => r.json());

// DZI descriptor loaded by OpenSeadragon automatically
const dziUrl = `${tilesBaseUrl}/tiles.dzi`;
```

**Files to modify:**
- `src/app/src/App.tsx` (lines 178-221)

#### 3.2 Simplify IsometricMap.tsx

**Remove (~400 lines):**
- LRU cache (lines 41-73) - browser handles caching
- Priority queue (lines 136-257) - native OSD handles
- Custom `getTileUrl` (lines 359-378)
- Custom `downloadTileStart` (lines 454-548)

**Update tile source config:**
```typescript
// Before: Custom tile source with virtual URLs
const tileSourceConfig = {
  getTileUrl: (level, x, y) => `pmtiles://${z}/${x}/${y}`,
  // ... complex configuration
};

// After: Native DZI
const viewer = OpenSeadragon({
  tileSources: `${tilesBaseUrl}/tiles.dzi`,
  // Standard OSD options only
});
```

**Files to modify:**
- `src/app/src/components/IsometricMap.tsx`

#### 3.3 Update Water Shader (if needed)

**Current:** Fetches tiles at `/tiles/0/{x}_{y}.png`
**DZI Structure:** `tiles_files/{level}/{col}_{row}.{format}`

May need to update `WaterShaderOverlay.tsx` to match DZI URL pattern, or ensure DZI structure matches expected pattern.

**Files to check:**
- `src/app/src/components/WaterShaderOverlay.tsx`

### Phase 4: Infrastructure

#### 4.1 Configure R2 for DZI Serving

**Current:** Worker intercepts requests, extracts tiles from PMTiles archive
**New:** Serve DZI files directly from R2 with public access

**R2 Configuration:**
```toml
# wrangler.toml for R2 public bucket
[[r2_buckets]]
binding = "TILES"
bucket_name = "isometric-nyc-tiles"
```

**CORS Headers (via R2 CORS rules or minimal Worker):**
```
Access-Control-Allow-Origin: *
Cache-Control: public, max-age=31536000, immutable
```

#### 4.2 Simplify or Remove Worker

**Option A: Remove entirely**
- R2 public access serves DZI directly
- Configure CORS at R2 level
- Simplest solution

**Option B: Minimal header Worker**
- Keep Worker for CORS/cache headers
- Remove all PMTiles extraction logic
- Trivial proxy to R2

**Files to modify:**
- `src/app/worker/src/index.ts`
- `src/app/worker/wrangler.toml`

### Phase 5: Deployment & Validation

#### 5.1 Deploy DZI to R2

```bash
# Upload DZI structure to R2
rclone sync ./output_files r2:isometric-nyc-tiles/tiles_files/
rclone copy ./tiles.dzi r2:isometric-nyc-tiles/
rclone copy ./tiles_metadata.json r2:isometric-nyc-tiles/
```

**Estimated file count:** ~20,000 files (128×144 base + 4 zoom levels)

#### 5.2 Validation Checklist

- [ ] DZI loads correctly in OpenSeadragon
- [ ] Zoom levels match expected behavior
- [ ] View state persistence works
- [ ] Water shader still functions (debug mode)
- [ ] Mobile pinch-to-zoom works
- [ ] Navigator (minimap) displays correctly
- [ ] Error handling for missing tiles

#### 5.3 Performance Comparison

| Metric | PMTiles (Current) | DZI (Expected) |
|--------|-------------------|----------------|
| Initial load | Measure | Should improve |
| Tile load time | Measure | ~67-117ms faster |
| JS bundle size | ~50KB (pmtiles) | ~0KB |
| Custom code | ~500 lines | ~50 lines |

---

## Risk Analysis & Mitigation

### High Risk: Source Image Assembly

**Risk:** 4.8 billion pixels may exceed memory limits
**Mitigation:**
1. Test with libvips sequential access mode first
2. If fails, create custom tile-to-DZI converter that bypasses full assembly
3. Fallback: Process in quadrants and merge pyramids

### Medium Risk: File Count / Upload Time

**Risk:** 20,000+ files = slow upload, R2 operation costs
**Mitigation:**
1. Use `rclone` parallel uploads
2. Consider SZI format if file count is prohibitive
3. Budget for R2 Class A operations (~$4.50 per million)

### Medium Risk: CORS Configuration

**Risk:** DZI tiles may fail CORS without Worker
**Mitigation:**
1. Configure R2 CORS rules
2. Keep minimal Worker as fallback
3. Test cross-origin access early

### Low Risk: View State Compatibility

**Risk:** Saved localStorage coordinates may be invalid after migration
**Mitigation:**
1. Grid dimensions unchanged (128×144), so coordinates should work
2. Add version check to clear invalid saved state if needed

---

## Estimated Effort

| Phase | Tasks | Estimate |
|-------|-------|----------|
| Phase 1 | Validation & Prototyping | 1-2 days |
| Phase 2 | Export Pipeline | 2-3 days |
| Phase 3 | Frontend Migration | 2-3 days |
| Phase 4 | Infrastructure | 1-2 days |
| Phase 5 | Deployment & Validation | 1 day |
| **Total** | | **7-11 days** |

---

## Files Summary

### Files to Create

| File | Purpose |
|------|---------|
| `src/isometric_nyc/generation/export_dzi.py` | DZI export script |
| `tiles_metadata.json` | Custom metadata sidecar |

### Files to Modify

| File | Changes |
|------|---------|
| `src/app/src/App.tsx` | Replace PMTiles init with DZI/metadata fetch |
| `src/app/src/components/IsometricMap.tsx` | Remove custom tile source (~400 lines), use native DZI |
| `src/app/src/components/WaterShaderOverlay.tsx` | Update tile URL pattern if needed |
| `src/app/src/config.ts` | Update URL handling |
| `src/app/worker/src/index.ts` | Simplify or remove |
| `src/app/package.json` | Remove `pmtiles` dependency |
| `pyproject.toml` | Add `pyvips` dependency |

### Files to Delete (after migration)

| File | Reason |
|------|--------|
| PMTiles-related Worker code | No longer needed |
| Legacy manifest.json support | Obsolete |

---

## Conclusion

**Migrate to DZI format.** The benefits are clear:

1. **~400 fewer lines of custom code** - Native OSD support
2. **Faster tile loading** - Research shows 67-117ms improvement
3. **Simpler infrastructure** - Static files, standard CDN caching
4. **Smaller JS bundle** - Remove pmtiles dependency (~50KB)

The main risk is source image assembly for a 4.8 billion pixel image. This should be validated in Phase 1 before committing to the full migration. If libvips streaming mode works, proceed with confidence. If not, create a custom tile-to-DZI converter.

---

## References

### OpenSeadragon Documentation
- DZI Tile Source: https://openseadragon.github.io/examples/tilesource-dzi/
- Creating Zooming Images: https://openseadragon.github.io/examples/creating-zooming-images/
- Custom Tile Source: https://openseadragon.github.io/examples/tilesource-custom/

### Performance Research
- FlexTileSource Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8529343/
- Simon Willison's VIPS Guide: https://til.simonwillison.net/javascript/openseadragon

### libvips Documentation
- Image Pyramids: https://www.libvips.org/API/8.17/making-image-pyramids.html
- Python bindings: https://libvips.github.io/pyvips/

### Cloudflare
- R2 Documentation: https://developers.cloudflare.com/r2/
- Cache Configuration: https://developers.cloudflare.com/cache/

### PMTiles (current format)
- Documentation: https://docs.protomaps.com/pmtiles/
- GitHub: https://github.com/protomaps/PMTiles
