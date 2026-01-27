#!/usr/bin/env python3
"""
Generate an overview image from PMTiles for the minimap.
Extracts the lowest zoom level tiles and stitches them together.
"""

import asyncio
import io
import sys
from pathlib import Path

from PIL import Image
from pmtiles.reader import Reader, MmapSource


async def generate_overview(pmtiles_path: str, output_path: str, target_zoom: int | None = None):
    """Extract tiles at the lowest zoom level and stitch into overview image."""
    
    print(f"📂 Opening PMTiles: {pmtiles_path}")
    
    with open(pmtiles_path, "rb") as f:
        source = MmapSource(f)
        reader = Reader(source)
        
        # Get metadata
        header = reader.header()
        metadata = reader.metadata()
        
        print(f"📊 PMTiles metadata:")
        print(f"   Min zoom: {header.get('min_zoom', 'N/A')}")
        print(f"   Max zoom: {header.get('max_zoom', 'N/A')}")
        print(f"   Tile type: {header.get('tile_type', 'N/A')}")
        
        # Use the minimum zoom level for the overview
        min_zoom = header.get('min_zoom', 0)
        if target_zoom is not None:
            zoom = target_zoom
        else:
            zoom = min_zoom
        
        print(f"\n🔍 Extracting tiles at zoom level {zoom}...")
        
        # Calculate tile grid size at this zoom level
        # At zoom z, there are 2^z tiles in each dimension
        num_tiles = 2 ** zoom
        
        # Collect all tiles at this zoom level
        tiles: dict[tuple[int, int], Image.Image] = {}
        tile_size = None  # Will detect from first tile
        
        for x in range(num_tiles):
            for y in range(num_tiles):
                try:
                    tile_data = reader.get(zoom, x, y)
                    if tile_data:
                        img = Image.open(io.BytesIO(tile_data))
                        tiles[(x, y)] = img
                        # Detect tile size from first tile
                        if tile_size is None:
                            tile_size = img.width
                            print(f"   📐 Detected tile size: {tile_size}x{img.height}")
                        print(f"   ✓ Tile {zoom}/{x}/{y} ({img.width}x{img.height}, {len(tile_data)} bytes)")
                except Exception as e:
                    print(f"   ✗ Tile {zoom}/{x}/{y} failed: {e}")
        
        if tile_size is None:
            tile_size = 256  # Fallback
        
        if not tiles:
            print(f"\n❌ No tiles found at zoom level {zoom}")
            # Try other zoom levels
            for z in range(min_zoom, min(min_zoom + 5, header.get('max_zoom', 10) + 1)):
                print(f"   Trying zoom {z}...")
                try:
                    tile_data = reader.get(z, 0, 0)
                    if tile_data:
                        print(f"   Found tiles at zoom {z}")
                        zoom = z
                        break
                except:
                    pass
            return
        
        # Determine actual bounds of tiles we found
        min_x = min(t[0] for t in tiles.keys())
        max_x = max(t[0] for t in tiles.keys())
        min_y = min(t[1] for t in tiles.keys())
        max_y = max(t[1] for t in tiles.keys())
        
        # Get the actual tile dimensions from a sample tile
        sample_tile = next(iter(tiles.values()))
        actual_tile_w = sample_tile.width
        actual_tile_h = sample_tile.height
        
        width = (max_x - min_x + 1) * actual_tile_w
        height = (max_y - min_y + 1) * actual_tile_h
        
        print(f"\n🖼️  Stitching {len(tiles)} tiles into {width}x{height} image...")
        print(f"   Grid: {max_x - min_x + 1}x{max_y - min_y + 1} tiles")
        print(f"   Tile size: {actual_tile_w}x{actual_tile_h}")
        
        # Create output image
        overview = Image.new('RGBA', (width, height), (10, 21, 37, 255))  # Dark blue background
        
        # Paste tiles
        for (x, y), tile_img in tiles.items():
            paste_x = (x - min_x) * actual_tile_w
            paste_y = (y - min_y) * actual_tile_h
            
            # Convert to RGBA if needed
            if tile_img.mode != 'RGBA':
                tile_img = tile_img.convert('RGBA')
            
            # Handle tiles that might be smaller (edge tiles)
            overview.paste(tile_img, (paste_x, paste_y))
        
        # Save
        overview.save(output_path, 'PNG', optimize=True)
        print(f"\n✅ Saved overview to: {output_path}")
        print(f"   Size: {width}x{height} pixels")
        
        # Also create a smaller version for faster loading
        small_path = output_path.replace('.png', '-small.png')
        max_dim = 512
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            small_size = (int(width * scale), int(height * scale))
            small_overview = overview.resize(small_size, Image.Resampling.LANCZOS)
            small_overview.save(small_path, 'PNG', optimize=True)
            print(f"   Also saved smaller version: {small_path} ({small_size[0]}x{small_size[1]})")


def main():
    # Default paths
    script_dir = Path(__file__).parent
    app_dir = script_dir.parent
    
    # Look for PMTiles in common locations
    pmtiles_candidates = [
        app_dir / "public" / "tiles.pmtiles",
        app_dir / "tiles.pmtiles",
        app_dir.parent.parent / "output" / "tiles.pmtiles",
    ]
    
    pmtiles_path = None
    for candidate in pmtiles_candidates:
        if candidate.exists():
            pmtiles_path = candidate
            break
    
    if len(sys.argv) > 1:
        pmtiles_path = Path(sys.argv[1])
    
    if not pmtiles_path or not pmtiles_path.exists():
        print("❌ Could not find PMTiles file.")
        print("   Usage: python generate_overview.py <path_to_tiles.pmtiles>")
        print(f"   Searched: {[str(c) for c in pmtiles_candidates]}")
        sys.exit(1)
    
    output_path = app_dir / "public" / "overview.png"
    
    # Parse optional zoom level argument
    target_zoom = None
    if len(sys.argv) > 2:
        target_zoom = int(sys.argv[2])
    
    asyncio.run(generate_overview(str(pmtiles_path), str(output_path), target_zoom))


if __name__ == "__main__":
    main()

