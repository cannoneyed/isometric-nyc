"""
Create Bounds - Interactive polygon editor for defining generation boundaries.

This script creates an interactive web-based polygon editor that allows you to:
- Draw and edit polygon boundaries on a map
- View existing generated and pending tiles
- Load and edit existing boundary files
- Save new boundaries to the bounds directory

Usage:
  uv run python src/isometric_nyc/generation/create_bounds.py [generation_dir]
  uv run python src/isometric_nyc/generation/create_bounds.py [generation_dir] --load bounds/custom.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/nyc)
  --load: Path to an existing bounds JSON file to edit

Output:
  Opens a web browser with the polygon editor
  Saves new bounds to generation/bounds/ directory
"""

import argparse
import json
import sqlite3
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from isometric_nyc.generation.bounds import get_bounds_dir, load_bounds, save_bounds
from isometric_nyc.generation.debug_map import (
  calculate_quadrant_corners,
  get_generated_quadrants,
  get_generation_config,
  get_pending_quadrants,
)


def generate_editor_html(
  config: dict,
  quadrant_polygons: list[tuple[int, int, list[tuple[float, float]], int]],
  pending_polygons: list[tuple[int, int, list[tuple[float, float]]]],
  seed_lat: float,
  seed_lng: float,
  center_lat: float,
  center_lng: float,
  nyc_boundary_geojson: dict,
  existing_bounds: dict | None = None,
  bounds_dir: Path | None = None,
) -> str:
  """Generate the HTML content for the bounds editor."""

  # Convert quadrant data to JSON
  quadrants_json = json.dumps(
    [
      {
        "x": qx,
        "y": qy,
        "corners": [[lat, lng] for lat, lng in corners],
        "water_status": water_status,
      }
      for qx, qy, corners, water_status in quadrant_polygons
    ]
  )

  pending_json = json.dumps(
    [
      {
        "x": qx,
        "y": qy,
        "corners": [[lat, lng] for lat, lng in corners],
      }
      for qx, qy, corners in pending_polygons
    ]
  )

  nyc_boundary_json = json.dumps(nyc_boundary_geojson)

  # Existing bounds polygon if provided
  existing_bounds_json = json.dumps(existing_bounds) if existing_bounds else "null"

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Create Bounds - Polygon Editor</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600&display=swap');

    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body {{
      font-family: 'Space Grotesk', sans-serif;
      background: #0a0a0f;
    }}

    #map {{
      width: 100vw;
      height: 100vh;
    }}

    .toolbar {{
      position: fixed;
      top: 16px;
      left: 16px;
      background: rgba(10, 10, 15, 0.95);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-width: 280px;
    }}

    .toolbar h2 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.8);
      margin-bottom: 4px;
    }}

    .toolbar-section {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .toolbar-row {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}

    .toolbar button {{
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #fff;
      padding: 8px 16px;
      border-radius: 8px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s ease;
    }}

    .toolbar button:hover {{
      background: rgba(255, 255, 255, 0.2);
      border-color: rgba(255, 255, 255, 0.3);
    }}

    .toolbar button:active {{
      transform: scale(0.98);
    }}

    .toolbar button.primary {{
      background: rgba(59, 130, 246, 0.8);
      border-color: rgba(59, 130, 246, 0.9);
    }}

    .toolbar button.primary:hover {{
      background: rgba(59, 130, 246, 1);
    }}

    .toolbar button.danger {{
      background: rgba(239, 68, 68, 0.6);
      border-color: rgba(239, 68, 68, 0.7);
    }}

    .toolbar button.danger:hover {{
      background: rgba(239, 68, 68, 0.8);
    }}

    .toolbar input[type="text"] {{
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #fff;
      padding: 8px 12px;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      flex: 1;
    }}

    .toolbar input[type="text"]::placeholder {{
      color: rgba(255, 255, 255, 0.4);
    }}

    .toolbar label {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: rgba(255, 255, 255, 0.7);
    }}

    .toolbar input[type="checkbox"] {{
      width: 16px;
      height: 16px;
    }}

    .info-text {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: rgba(255, 255, 255, 0.5);
      line-height: 1.5;
    }}

    .vertex-count {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: #4ecdc4;
    }}

    .legend {{
      position: fixed;
      bottom: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
    }}

    .legend-swatch {{
      width: 16px;
      height: 16px;
      border-radius: 4px;
    }}

    .legend-swatch.bounds {{
      background: rgba(16, 185, 129, 0.35);
      border: 2px solid rgba(16, 185, 129, 0.9);
    }}

    .legend-swatch.nyc-boundary {{
      background: rgba(59, 130, 246, 0.15);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.quadrant {{
      background: rgba(255, 107, 107, 0.35);
      border: 2px solid rgba(255, 107, 107, 0.7);
    }}

    .legend-swatch.pending {{
      background: rgba(168, 85, 247, 0.35);
      border: 2px solid rgba(168, 85, 247, 0.7);
    }}

    .vertex-marker {{
      width: 14px;
      height: 14px;
      background: #10b981;
      border: 3px solid #fff;
      border-radius: 50%;
      cursor: move;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
      z-index: 1000;
    }}

    .vertex-marker:hover {{
      transform: scale(1.2);
      background: #34d399;
    }}

    .toast {{
      position: fixed;
      bottom: 80px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(16, 185, 129, 0.95);
      color: #fff;
      padding: 12px 24px;
      border-radius: 8px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 14px;
      z-index: 2000;
      opacity: 0;
      transition: opacity 0.3s ease;
      pointer-events: none;
    }}

    .toast.error {{
      background: rgba(239, 68, 68, 0.95);
    }}

    .toast.show {{
      opacity: 1;
    }}

    .help-panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
      max-width: 280px;
    }}

    .help-panel h3 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 12px;
    }}

    .help-item {{
      display: flex;
      gap: 10px;
      font-size: 12px;
      margin-bottom: 8px;
      color: rgba(255, 255, 255, 0.7);
    }}

    .help-key {{
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255, 255, 255, 0.1);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="toolbar">
    <h2>🗺️ Bounds Editor</h2>

    <div class="toolbar-section">
      <div class="toolbar-row">
        <button onclick="resetToRectangle()" title="Reset to default rectangle">Reset</button>
        <button onclick="clearBounds()" class="danger" title="Clear all vertices">Clear</button>
      </div>
    </div>

    <div class="toolbar-section">
      <label>
        <input type="checkbox" id="showTiles" checked onchange="toggleTiles()">
        Show Generated Tiles
      </label>
      <label>
        <input type="checkbox" id="showPending" checked onchange="togglePending()">
        Show Pending Tiles
      </label>
      <label>
        <input type="checkbox" id="showNyc" checked onchange="toggleNyc()">
        Show NYC Boundary
      </label>
    </div>

    <div class="toolbar-section">
      <div class="info-text">
        Vertices: <span class="vertex-count" id="vertexCount">0</span>
      </div>
    </div>

    <div class="toolbar-section">
      <div class="toolbar-row">
        <input type="text" id="boundsName" placeholder="my-bounds" />
        <button onclick="saveBounds()" class="primary">Save</button>
      </div>
      <div class="info-text">
        Saves to: bounds/[name].json
      </div>
    </div>
  </div>

  <div class="help-panel">
    <h3>Controls</h3>
    <div class="help-item">
      <span class="help-key">Drag vertex</span>
      <span>Move vertex</span>
    </div>
    <div class="help-item">
      <span class="help-key">Dbl-click line</span>
      <span>Add vertex</span>
    </div>
    <div class="help-item">
      <span class="help-key">Dbl-click vertex</span>
      <span>Delete vertex</span>
    </div>
    <div class="help-item">
      <span class="help-key">Min vertices</span>
      <span>3</span>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch bounds"></div>
      <span>Editable Bounds</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch nyc-boundary"></div>
      <span>NYC Boundary</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch quadrant"></div>
      <span>Generated Tiles</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch pending"></div>
      <span>Pending Tiles</span>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    // Data from Python
    const quadrants = {quadrants_json};
    const pendingQuadrants = {pending_json};
    const nycBoundary = {nyc_boundary_json};
    const existingBounds = {existing_bounds_json};
    const seedLat = {seed_lat};
    const seedLng = {seed_lng};
    const centerLat = {center_lat};
    const centerLng = {center_lng};

    // Initialize map (disable double-click zoom since we use it for vertex editing)
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true,
      doubleClickZoom: false
    }}).setView([centerLat, centerLng], 12);

    // Add dark tile layer
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; OpenStreetMap &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 20
    }}).addTo(map);

    // Layers for visibility toggles
    let tilesLayer = L.layerGroup().addTo(map);
    let pendingLayer = L.layerGroup().addTo(map);
    let nycLayer = L.layerGroup().addTo(map);

    // Styles
    const quadrantStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.2,
      color: '#ff6b6b',
      weight: 1,
      opacity: 0.5
    }};

    const pendingStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.2,
      color: '#a855f7',
      weight: 1,
      opacity: 0.5
    }};

    const nycBoundaryStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.08,
      color: '#3b82f6',
      weight: 2,
      opacity: 0.5
    }};

    const boundsStyle = {{
      fillColor: '#10b981',
      fillOpacity: 0.15,
      color: '#10b981',
      weight: 2.5,
      opacity: 0.9
    }};

    // Add NYC boundary
    L.geoJSON(nycBoundary, {{ style: nycBoundaryStyle }}).addTo(nycLayer);

    // Add generated quadrants
    quadrants.forEach(q => {{
      L.polygon(q.corners, quadrantStyle).addTo(tilesLayer);
    }});

    // Add pending quadrants
    pendingQuadrants.forEach(q => {{
      L.polygon(q.corners, pendingStyle).addTo(pendingLayer);
    }});

    // Bounds polygon editing
    let boundsPolygon = null;
    let vertices = [];
    let vertexMarkers = [];
    let edgeMarkers = [];

    // Create vertex marker
    function createVertexMarker(latlng, index) {{
      const icon = L.divIcon({{
        className: 'vertex-marker',
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      }});

      const marker = L.marker(latlng, {{
        icon: icon,
        draggable: true,
        zIndexOffset: 1000
      }}).addTo(map);

      marker.vertexIndex = index;

      // Drag handlers
      marker.on('drag', (e) => {{
        vertices[marker.vertexIndex] = [e.latlng.lat, e.latlng.lng];
        updatePolygon();
        updateEdgeMarkers();
      }});

      marker.on('dragend', () => {{
        validatePolygon();
      }});

      // Double-click to delete (min 3 vertices)
      marker.on('dblclick', (e) => {{
        L.DomEvent.stopPropagation(e);
        if (vertices.length > 3) {{
          deleteVertex(marker.vertexIndex);
        }} else {{
          showToast('Cannot delete: minimum 3 vertices required', true);
        }}
      }});

      return marker;
    }}

    // Create edge marker (for adding new vertices)
    function createEdgeMarker(latlng, afterIndex) {{
      const icon = L.divIcon({{
        className: 'vertex-marker',
        html: '<div style="width:10px;height:10px;background:rgba(16,185,129,0.3);border:2px solid rgba(16,185,129,0.6);border-radius:50%;cursor:pointer;"></div>',
        iconSize: [10, 10],
        iconAnchor: [5, 5]
      }});

      const marker = L.marker(latlng, {{
        icon: icon,
        zIndexOffset: 500
      }}).addTo(map);

      marker.afterIndex = afterIndex;

      marker.on('dblclick', (e) => {{
        L.DomEvent.stopPropagation(e);
        addVertexAfter(marker.afterIndex, e.latlng);
      }});

      return marker;
    }}

    // Update polygon shape
    function updatePolygon() {{
      if (boundsPolygon) {{
        boundsPolygon.setLatLngs(vertices);
      }} else if (vertices.length >= 3) {{
        boundsPolygon = L.polygon(vertices, boundsStyle).addTo(map);
        boundsPolygon.on('dblclick', handlePolygonDoubleClick);
      }}
      document.getElementById('vertexCount').textContent = vertices.length;
    }}

    // Handle double-click on polygon edge to add vertex
    function handlePolygonDoubleClick(e) {{
      const clickLatLng = e.latlng;

      // Find closest edge
      let minDist = Infinity;
      let insertAfter = 0;

      for (let i = 0; i < vertices.length; i++) {{
        const j = (i + 1) % vertices.length;
        const p1 = L.latLng(vertices[i]);
        const p2 = L.latLng(vertices[j]);
        const dist = pointToLineDistance(clickLatLng, p1, p2);
        if (dist < minDist) {{
          minDist = dist;
          insertAfter = i;
        }}
      }}

      addVertexAfter(insertAfter, clickLatLng);
    }}

    // Calculate distance from point to line segment
    function pointToLineDistance(point, lineStart, lineEnd) {{
      const dx = lineEnd.lng - lineStart.lng;
      const dy = lineEnd.lat - lineStart.lat;
      const len2 = dx * dx + dy * dy;

      if (len2 === 0) return point.distanceTo(lineStart);

      let t = ((point.lng - lineStart.lng) * dx + (point.lat - lineStart.lat) * dy) / len2;
      t = Math.max(0, Math.min(1, t));

      const proj = L.latLng(
        lineStart.lat + t * dy,
        lineStart.lng + t * dx
      );

      return point.distanceTo(proj);
    }}

    // Add vertex after a given index
    function addVertexAfter(afterIndex, latlng) {{
      vertices.splice(afterIndex + 1, 0, [latlng.lat, latlng.lng]);
      rebuildVertexMarkers();
      updatePolygon();
      showToast('Vertex added');
    }}

    // Delete vertex at index
    function deleteVertex(index) {{
      vertices.splice(index, 1);
      rebuildVertexMarkers();
      updatePolygon();
      showToast('Vertex deleted');
    }}

    // Rebuild all vertex markers
    function rebuildVertexMarkers() {{
      // Remove old markers
      vertexMarkers.forEach(m => map.removeLayer(m));
      vertexMarkers = [];

      // Create new markers
      vertices.forEach((v, i) => {{
        const marker = createVertexMarker(L.latLng(v[0], v[1]), i);
        vertexMarkers.push(marker);
      }});

      updateEdgeMarkers();
    }}

    // Update edge markers (midpoints)
    function updateEdgeMarkers() {{
      // Remove old edge markers
      edgeMarkers.forEach(m => map.removeLayer(m));
      edgeMarkers = [];

      // Create new edge markers at midpoints
      for (let i = 0; i < vertices.length; i++) {{
        const j = (i + 1) % vertices.length;
        const midLat = (vertices[i][0] + vertices[j][0]) / 2;
        const midLng = (vertices[i][1] + vertices[j][1]) / 2;
        const marker = createEdgeMarker(L.latLng(midLat, midLng), i);
        edgeMarkers.push(marker);
      }}
    }}

    // Validate polygon (check for self-intersection)
    function validatePolygon() {{
      // Simple validation - check for self-intersecting edges
      // This is a basic check; more robust validation could be added

      for (let i = 0; i < vertices.length; i++) {{
        const p1 = vertices[i];
        const p2 = vertices[(i + 1) % vertices.length];

        for (let j = i + 2; j < vertices.length; j++) {{
          if (j === (i + vertices.length - 1) % vertices.length) continue; // Skip adjacent edge

          const p3 = vertices[j];
          const p4 = vertices[(j + 1) % vertices.length];

          if (segmentsIntersect(p1, p2, p3, p4)) {{
            showToast('Warning: Polygon has self-intersecting edges!', true);
            return false;
          }}
        }}
      }}
      return true;
    }}

    // Check if two line segments intersect
    function segmentsIntersect(p1, p2, p3, p4) {{
      function ccw(A, B, C) {{
        return (C[0] - A[0]) * (B[1] - A[1]) > (B[0] - A[0]) * (C[1] - A[1]);
      }}
      return ccw(p1, p3, p4) !== ccw(p2, p3, p4) && ccw(p1, p2, p3) !== ccw(p1, p2, p4);
    }}

    // Reset to default rectangle (50% of screen)
    function resetToRectangle() {{
      const bounds = map.getBounds();
      const center = bounds.getCenter();
      const latSpan = (bounds.getNorth() - bounds.getSouth()) * 0.25;
      const lngSpan = (bounds.getEast() - bounds.getWest()) * 0.25;

      vertices = [
        [center.lat + latSpan, center.lng - lngSpan],  // TL
        [center.lat + latSpan, center.lng + lngSpan],  // TR
        [center.lat - latSpan, center.lng + lngSpan],  // BR
        [center.lat - latSpan, center.lng - lngSpan],  // BL
      ];

      rebuildVertexMarkers();
      updatePolygon();
      showToast('Reset to rectangle');
    }}

    // Clear all bounds
    function clearBounds() {{
      if (!confirm('Clear all vertices?')) return;

      vertices = [];
      vertexMarkers.forEach(m => map.removeLayer(m));
      vertexMarkers = [];
      edgeMarkers.forEach(m => map.removeLayer(m));
      edgeMarkers = [];
      if (boundsPolygon) {{
        map.removeLayer(boundsPolygon);
        boundsPolygon = null;
      }}
      document.getElementById('vertexCount').textContent = '0';
      showToast('Bounds cleared');
    }}

    // Toggle visibility
    function toggleTiles() {{
      const show = document.getElementById('showTiles').checked;
      if (show) map.addLayer(tilesLayer);
      else map.removeLayer(tilesLayer);
    }}

    function togglePending() {{
      const show = document.getElementById('showPending').checked;
      if (show) map.addLayer(pendingLayer);
      else map.removeLayer(pendingLayer);
    }}

    function toggleNyc() {{
      const show = document.getElementById('showNyc').checked;
      if (show) map.addLayer(nycLayer);
      else map.removeLayer(nycLayer);
    }}

    // Save bounds
    async function saveBounds() {{
      if (vertices.length < 3) {{
        showToast('Need at least 3 vertices to save', true);
        return;
      }}

      const name = document.getElementById('boundsName').value.trim();
      if (!name) {{
        showToast('Please enter a name for the bounds', true);
        return;
      }}

      // Build GeoJSON
      const geojson = {{
        type: 'FeatureCollection',
        name: name,
        features: [{{
          type: 'Feature',
          properties: {{ name: name }},
          geometry: {{
            type: 'Polygon',
            coordinates: [[
              ...vertices.map(v => [v[1], v[0]]),  // GeoJSON uses [lng, lat]
              [vertices[0][1], vertices[0][0]]  // Close the ring
            ]]
          }}
        }}]
      }};

      try {{
        const response = await fetch('/save', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ name, geojson }})
        }});

        const result = await response.json();
        if (result.success) {{
          showToast('Saved to: ' + result.path);
        }} else {{
          showToast('Error: ' + result.error, true);
        }}
      }} catch (error) {{
        showToast('Error saving: ' + error.message, true);
      }}
    }}

    // Show toast notification
    function showToast(message, isError = false) {{
      const toast = document.getElementById('toast');
      toast.textContent = message;
      toast.className = isError ? 'toast error show' : 'toast show';
      setTimeout(() => {{
        toast.className = toast.className.replace(' show', '');
      }}, 3000);
    }}

    // Initialize with existing bounds or default rectangle
    function init() {{
      if (existingBounds && existingBounds.features && existingBounds.features.length > 0) {{
        // Load existing bounds
        const feature = existingBounds.features[0];
        if (feature.geometry && feature.geometry.coordinates) {{
          const coords = feature.geometry.coordinates[0];
          // GeoJSON uses [lng, lat], we use [lat, lng]
          vertices = coords.slice(0, -1).map(c => [c[1], c[0]]);  // Remove closing point
          rebuildVertexMarkers();
          updatePolygon();
          showToast('Loaded existing bounds');

          // Set name if available
          if (feature.properties && feature.properties.name) {{
            document.getElementById('boundsName').value = feature.properties.name;
          }}
        }}
      }} else {{
        // Start with default rectangle
        resetToRectangle();
      }}

      // Fit to NYC bounds
      if (quadrants.length > 0 || pendingQuadrants.length > 0) {{
        const allCorners = [
          ...quadrants.flatMap(q => q.corners),
          ...pendingQuadrants.flatMap(q => q.corners)
        ];
        if (allCorners.length > 0) {{
          const bounds = L.latLngBounds(allCorners);
          map.fitBounds(bounds.pad(0.1));
        }}
      }}
    }}

    init();
  </script>
</body>
</html>
"""
  return html


class BoundsEditorHandler(SimpleHTTPRequestHandler):
  """HTTP request handler for the bounds editor."""

  html_content: str = ""
  bounds_dir: Path = get_bounds_dir()

  def do_GET(self):
    """Handle GET requests."""
    if self.path == "/" or self.path == "/index.html":
      self.send_response(200)
      self.send_header("Content-type", "text/html")
      self.end_headers()
      self.wfile.write(self.html_content.encode("utf-8"))
    else:
      self.send_error(404)

  def do_POST(self):
    """Handle POST requests (save bounds)."""
    if self.path == "/save":
      content_length = int(self.headers["Content-Length"])
      post_data = self.rfile.read(content_length)
      data = json.loads(post_data.decode("utf-8"))

      try:
        name = data["name"]
        geojson = data["geojson"]

        # Save to bounds directory
        output_path = save_bounds(geojson, name, self.bounds_dir)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = json.dumps({"success": True, "path": str(output_path)})
        self.wfile.write(response.encode("utf-8"))
      except Exception as e:
        self.send_response(500)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = json.dumps({"success": False, "error": str(e)})
        self.wfile.write(response.encode("utf-8"))
    else:
      self.send_error(404)

  def log_message(self, format, *args):
    """Suppress logging."""
    pass


def run_editor(
  generation_dir: Path,
  existing_bounds_path: Path | None = None,
  port: int = 8765,
) -> None:
  """
  Run the bounds editor web server.

  Args:
    generation_dir: Path to the generation directory
    existing_bounds_path: Optional path to existing bounds to edit
    port: Port to run the server on
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  print(f"\n{'=' * 60}")
  print("🗺️  Bounds Editor")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"   Config: {config.get('name', 'unnamed')}")

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"   Generated quadrants: {len(generated)}")

    # Get pending quadrants
    pending = get_pending_quadrants(conn)
    generated_set = {(qx, qy) for qx, qy, _ in generated}
    pending = [(qx, qy) for qx, qy in pending if (qx, qy) not in generated_set]
    print(f"   Pending quadrants: {len(pending)}")

    # Calculate corners for quadrants
    quadrant_polygons = []
    all_lats, all_lngs = [], []

    for qx, qy, water_status in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners, water_status))
      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    pending_polygons = []
    for qx, qy in pending:
      corners = calculate_quadrant_corners(config, qx, qy)
      pending_polygons.append((qx, qy, corners))
      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate center
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]

    if all_lats and all_lngs:
      center_lat = (min(all_lats) + max(all_lats)) / 2
      center_lng = (min(all_lngs) + max(all_lngs)) / 2
    else:
      center_lat = seed_lat
      center_lng = seed_lng

    # Load NYC boundary
    nyc_boundary = load_bounds()

    # Load existing bounds if provided
    existing_bounds = None
    if existing_bounds_path:
      try:
        existing_bounds = load_bounds(existing_bounds_path)
        print(f"   Loading existing bounds: {existing_bounds_path}")
      except Exception as e:
        print(f"   Warning: Could not load existing bounds: {e}")

    # Generate HTML
    html_content = generate_editor_html(
      config=config,
      quadrant_polygons=quadrant_polygons,
      pending_polygons=pending_polygons,
      seed_lat=seed_lat,
      seed_lng=seed_lng,
      center_lat=center_lat,
      center_lng=center_lng,
      nyc_boundary_geojson=nyc_boundary,
      existing_bounds=existing_bounds,
      bounds_dir=get_bounds_dir(),
    )

    # Setup handler with HTML content
    BoundsEditorHandler.html_content = html_content

    # Start server
    server = HTTPServer(("localhost", port), BoundsEditorHandler)
    url = f"http://localhost:{port}/"

    print(f"\n✅ Bounds editor running at: {url}")
    print("   Press Ctrl+C to stop")

    # Open browser
    webbrowser.open(url)

    # Run server
    server.serve_forever()

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create or edit generation boundaries with an interactive polygon editor."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/nyc"),
    help="Path to the generation directory (default: generations/nyc)",
  )
  parser.add_argument(
    "--load",
    "-l",
    type=Path,
    help="Path to an existing bounds JSON file to edit",
  )
  parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=8765,
    help="Port to run the editor server on (default: 8765)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  existing_bounds_path = args.load.resolve() if args.load else None

  try:
    run_editor(generation_dir, existing_bounds_path, args.port)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except KeyboardInterrupt:
    print("\n\n👋 Bounds editor stopped")
    return 0
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())
