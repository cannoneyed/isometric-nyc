# Speed up pmtiles

The process for exporting the pmtiles (src/isometric_nyc/generation/export_pmtiles.py) is very slow - likely because it's in python. Please come up with a plan for speeding this up - if we can make it faster while still using python, great, otherwise think about doing the work in a faster language e.g. Go

Please think very hard about this, then write the plan below:

## Analysis

### Current Performance Bottlenecks

The current implementation (`export_pmtiles.py`) has several performance issues:

1. **Sequential Database Reads**: Each tile is loaded from SQLite one at a time via individual `get_quadrant_data()` calls, each opening/closing a connection.

2. **Sequential Image Processing**: PIL image operations (decode, postprocess, re-encode) happen sequentially in the main thread.

3. **Python GIL Limitation**: Even with threading, PIL's CPU-intensive operations are GIL-bound.

4. **Memory Inefficiency**: All base tiles (~28K+) are loaded into memory before writing begins.

5. **Sequential Zoom Level Generation**: Each zoom level is computed sequentially after all base tiles are loaded.

### Scale Context

- ~28,129 tiles with generation data
- Grid spans from (-87, -84) to (154, 110) = 242 × 195 = ~47,190 potential tiles
- Output: ~9.5 GB PMTiles file
- Each tile: 512×512 PNG/WebP

## Solution: Python with Multiprocessing

**Recommendation**: Stay with Python but use `multiprocessing.Pool` for parallel processing.

### Why Python + Multiprocessing (Not Go)

1. **Existing PIL/Pillow ecosystem**: The postprocessing (palette quantization, resizing) is complex and well-tested in PIL. Rewriting in Go would require reimplementing this logic with a different image library (likely `image` stdlib or `bimg`/`libvips`).

2. **PMTiles writer is simple**: The Python `pmtiles` library is straightforward - it writes tiles to a temp file, then constructs the directory structure. The writer isn't the bottleneck.

3. **The bottleneck is CPU-bound**: Image decode/encode/quantize is CPU-bound. Python's `multiprocessing` bypasses the GIL by using separate processes.

4. **Minimal code changes**: We can parallelize without rewriting the entire script.

5. **Similar patterns exist in codebase**: `detect_water_tiles.py` and `postprocess_tiles.py` already use `ProcessPoolExecutor` successfully.

## Implementation Plan

### Phase 1: Batch Database Reads

Replace individual `get_quadrant_data()` calls with a single bulk query that loads all tiles into memory:

```python
def get_all_quadrant_data_in_range(
    db_path: Path,
    tl: tuple[int, int],
    br: tuple[int, int],
    use_render: bool = False
) -> dict[tuple[int, int], bytes]:
    """Load all tile data in range with a single query."""
    conn = sqlite3.connect(db_path)
    column = "render" if use_render else "generation"
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT quadrant_x, quadrant_y, {column}
        FROM quadrants
        WHERE quadrant_x >= ? AND quadrant_x <= ?
          AND quadrant_y >= ? AND quadrant_y <= ?
          AND {column} IS NOT NULL
        """,
        (tl[0], br[0], tl[1], br[1]),
    )
    return {(row[0], row[1]): row[2] for row in cursor.fetchall()}
```

### Phase 2: Parallel Tile Processing

Use `multiprocessing.Pool` to process tiles in parallel:

```python
def process_tile_worker(args: tuple) -> tuple[int, int, bytes]:
    """Worker function for parallel tile processing."""
    x, y, raw_data, palette_bytes, pixel_scale, dither, image_format, webp_quality = args
    
    # Reconstruct palette from bytes (can't pickle PIL Image)
    palette_img = Image.open(io.BytesIO(palette_bytes)) if palette_bytes else None
    
    if raw_data is None:
        return x, y, create_black_tile_bytes(...)
    
    img = Image.open(io.BytesIO(raw_data))
    if palette_img:
        img = postprocess_image(img, palette_img, pixel_scale, dither)
    return x, y, image_to_bytes(img, image_format, webp_quality)
```

Main processing loop:

```python
from multiprocessing import Pool, cpu_count

def export_base_tiles_parallel(
    raw_tiles: dict[tuple[int, int], bytes],
    palette_img: Image.Image | None,
    ...
) -> dict[tuple[int, int], bytes]:
    # Serialize palette for workers (PIL Images aren't picklable)
    palette_bytes = None
    if palette_img:
        buf = io.BytesIO()
        palette_img.save(buf, format='PNG')
        palette_bytes = buf.getvalue()
    
    # Prepare work items
    work_items = [
        (x, y, raw_tiles.get((x, y)), palette_bytes, pixel_scale, ...)
        for y in range(padded_height)
        for x in range(padded_width)
    ]
    
    # Process in parallel
    num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
    with Pool(num_workers) as pool:
        results = pool.imap_unordered(process_tile_worker, work_items, chunksize=100)
        processed_tiles = {}
        for x, y, tile_bytes in tqdm(results, total=len(work_items)):
            processed_tiles[(x, y)] = tile_bytes
    
    return processed_tiles
```

### Phase 3: Streaming Writes (Optional Enhancement)

Instead of holding all tiles in memory, stream directly to the PMTiles writer:

```python
# Write tiles as they're processed instead of buffering all in memory
with pmtiles_write(str(output_path)) as writer:
    with Pool(num_workers) as pool:
        for x, y, tile_bytes in pool.imap(process_tile_worker, work_items, chunksize=100):
            tileid = zxy_to_tileid(pmtiles_z, x, y)
            writer.write_tile(tileid, tile_bytes)
```

**Note**: PMTiles writer requires tiles in sorted order by tileid. We may need to buffer per-zoom-level or sort coordinates before processing.

### Phase 4: Parallel Zoom Level Generation

Process zoom levels in parallel too:

```python
def generate_zoom_tile_worker(args):
    """Generate a single tile for a higher zoom level."""
    zx, zy, scale, base_tiles_subset, image_format, webp_quality = args
    combined = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 255))
    
    for dy in range(scale):
        for dx in range(scale):
            base_x = zx * scale + dx
            base_y = zy * scale + dy
            tile_data = base_tiles_subset.get((base_x, base_y), BLACK_TILE)
            # ... combine logic
    
    return zx, zy, image_to_bytes(combined.convert("RGB"), image_format, webp_quality)
```

## Expected Performance Improvement

| Optimization | Estimated Speedup |
|-------------|-------------------|
| Batch DB reads | 2-3x |
| Parallel processing (8 workers) | 5-7x |
| Combined | 10-20x |

For ~28K tiles, this could reduce export time from potentially hours to ~10-20 minutes.

## Alternative: Go Implementation (If Needed)

If Python multiprocessing proves insufficient, a Go implementation would:

1. Use `github.com/protomaps/go-pmtiles` for PMTiles writing
2. Use Go's `image` stdlib or `bimg` (libvips bindings) for image processing  
3. Use goroutines for easy parallelism
4. Require reimplementing the palette quantization logic

This is a larger undertaking (~500+ lines of Go) but would be faster due to no GIL and better memory efficiency.

## Recommended Implementation Order

1. **Quick win**: Batch database reads (Phase 1) - ~30 min of work
2. **Major speedup**: Parallel tile processing (Phase 2) - ~1-2 hours
3. **Memory optimization**: Streaming writes (Phase 3) - ~1 hour
4. **Final polish**: Parallel zoom generation (Phase 4) - ~1 hour

Total estimated implementation time: 3-5 hours for Python solution.

## Status: ✅ Completed

## Implementation Summary

The optimized `export_pmtiles.py` now includes:

1. **Batch Database Reads**: Single query loads all tiles at once via `get_all_quadrant_data_in_range()`
2. **Parallel Tile Processing**: Uses `ProcessPoolExecutor` to process tiles concurrently
3. **Parallel Zoom Generation**: Each zoom level is also generated using parallel workers
4. **Performance Metrics**: Script now reports detailed timing for each phase

### New CLI Options

- `-w, --workers N`: Control the number of parallel workers (default: 8, capped at CPU count)

### Verified Performance (16x16 tile test)

- Database load: 0.5s
- Tile processing: 4.9s (with 4 workers, ~53 tiles/sec)
- Total: 8.5s for 341 tiles

### Expected Full Export Performance

For the full ~28K tile export:
- Previous sequential approach: Potentially hours
- New parallel approach: ~10-20 minutes (estimated 10-20x speedup)
