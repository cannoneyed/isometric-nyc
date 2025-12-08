"""
Simple web app to view generated tiles in an nx×ny grid.

Usage:
  uv run python src/isometric_nyc/e2e_generation/view_generations.py <generation_dir>

Then open http://localhost:8080/?x=0&y=0 in your browser.

URL Parameters:
  x, y   - Starting coordinates (default: 0, 0)
  nx, ny - Grid size nx×ny (default: 2, max: 20)
  lines  - Show grid lines: 1=on, 0=off (default: 1)
  coords - Show coordinates: 1=on, 0=off (default: 1)
  render - Show renders instead of generations: 1=renders, 0=generations (default: 0)

Keyboard shortcuts:
  Arrow keys - Navigate the grid
  L          - Toggle lines
  C          - Toggle coords
  G          - Toggle render/generation mode
"""

import argparse
import sqlite3
from pathlib import Path

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Will be set by main()
GENERATION_DIR: Path | None = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Generated Tiles Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
      background: #1a1a2e;
      color: #eee;
      min-height: 100vh;
      padding: 20px;
    }
    
    h1 {
      font-size: 1.5rem;
      margin-bottom: 20px;
      color: #00d9ff;
    }
    
    .controls {
      margin-bottom: 20px;
      display: flex;
      gap: 15px;
      align-items: center;
      flex-wrap: wrap;
    }
    
    .controls label {
      color: #888;
    }
    
    .controls input[type="number"] {
      width: 60px;
      padding: 8px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #16213e;
      color: #fff;
      font-family: inherit;
    }
    
    .controls button {
      padding: 8px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-weight: bold;
    }
    
    .controls button:hover {
      background: #00b8d4;
    }
    
    .toggle-group {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: 10px;
      padding-left: 15px;
      border-left: 1px solid #333;
    }
    
    .toggle-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      user-select: none;
    }
    
    .toggle-group input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: #00d9ff;
      cursor: pointer;
    }
    
    .grid-container {
      display: inline-block;
      border-radius: 8px;
      overflow: hidden;
    }
    
    .grid-container.show-lines {
      border: 2px solid #333;
    }
    
    .grid {
      display: grid;
      grid-template-columns: repeat({{ nx }}, {{ size_px }}px);
      grid-auto-rows: {{ size_px }}px;
      background: #333;
    }
    
    .grid-container.show-lines .grid {
      gap: 2px;
    }
    
    .grid-container:not(.show-lines) .grid {
      gap: 0;
      background: transparent;
    }
    
    .grid-container:not(.show-lines) {
      border: none;
    }
    
    .tile {
      position: relative;
      background: #2a2a4a;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .tile img {
      display: block;
      max-width: 100%;
      height: auto;
    }
    
    .tile.placeholder {
      background: #3a3a5a;
      min-width: {{ size_px }}px;
      min-height: {{ size_px }}px;
    }
    
    .tile .coords {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(0, 0, 0, 0.7);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #00d9ff;
      transition: opacity 0.2s;
    }
    
    .tile.placeholder .coords {
      color: #666;
    }
    
    .grid-container:not(.show-coords) .tile .coords {
      opacity: 0;
    }
    
    /* Tool button styles */
    .tools-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .tools-label {
      color: #666;
      font-size: 0.85rem;
    }
    
    .tool-btn {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      background: #333;
      color: #888;
      border: 1px solid #444;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-size: 0.85rem;
      transition: all 0.2s;
    }
    
    .tool-btn:hover {
      background: #444;
      color: #fff;
      border-color: #555;
    }
    
    .tool-btn.active {
      background: #00d9ff;
      color: #1a1a2e;
      border-color: #00d9ff;
    }
    
    .tool-btn svg {
      width: 14px;
      height: 14px;
    }
    
    /* Selection styles */
    .tile.selected {
      outline: 3px solid #ff3333;
      outline-offset: -3px;
      z-index: 10;
    }
    
    .grid-container.show-lines .tile.selected {
      outline-color: #ff3333;
    }
    
    .tile.selectable {
      cursor: pointer;
    }
    
    .tile.placeholder.selected {
      background: rgba(255, 51, 51, 0.15);
    }
    
    /* Selection status bar */
    .selection-status {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 15px;
      padding: 8px 12px;
      background: rgba(255, 51, 51, 0.1);
      border: 1px solid rgba(255, 51, 51, 0.3);
      border-radius: 6px;
      font-size: 0.9rem;
      color: #ff6666;
    }
    
    .selection-status.empty {
      background: transparent;
      border-color: #333;
      color: #666;
    }
    
    .selection-limit {
      color: #888;
      font-size: 0.8rem;
    }
    
    .deselect-btn {
      padding: 4px 10px;
      background: #ff3333;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-size: 0.8rem;
      margin-left: auto;
      transition: all 0.2s;
    }
    
    .deselect-btn:hover:not(:disabled) {
      background: #ff5555;
    }
    
    .deselect-btn:disabled {
      background: #444;
      color: #666;
      cursor: not-allowed;
    }
    
    .generate-btn {
      padding: 6px 16px;
      background: #00d9ff;
      color: #1a1a2e;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-family: inherit;
      font-size: 0.85rem;
      font-weight: bold;
      transition: all 0.2s;
    }
    
    .generate-btn:hover:not(:disabled) {
      background: #00b8d4;
    }
    
    .generate-btn:disabled {
      background: #444;
      color: #666;
      cursor: not-allowed;
      font-weight: normal;
    }
    
    .info {
      margin-top: 20px;
      color: #666;
      font-size: 0.85rem;
    }
    

  </style>
</head>
<body>
  <h1>🎨 Generated Tiles Viewer</h1>
  
  <div class="controls">
    <label>X: <input type="number" id="x" value="{{ x }}"></label>
    <label>Y: <input type="number" id="y" value="{{ y }}"></label>
    <label>NX: <input type="number" id="nx" value="{{ nx }}" min="1" max="20"></label>
    <label>NY: <input type="number" id="ny" value="{{ ny }}" min="1" max="20"></label>
    <label>Size: <input type="number" id="sizePx" value="{{ size_px }}" step="32"></label>
    <button onclick="goTo()">Go</button>
    
    
    <div class="toggle-group">
      <label>
        <input type="checkbox" id="showLines" {% if show_lines %}checked{% endif %} onchange="toggleLines()">
        Lines
      </label>
      <label>
        <input type="checkbox" id="showCoords" {% if show_coords %}checked{% endif %} onchange="toggleCoords()">
        Coords
      </label>
      <label>
        <input type="checkbox" id="showRender" {% if show_render %}checked{% endif %} onchange="toggleRender()">
        Renders
      </label>
    </div>
    
    <div class="toggle-group tools-group">
      <span class="tools-label">Tools:</span>
      <button id="selectTool" class="tool-btn" onclick="toggleSelectTool()" title="Select quadrants">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"></path>
          <path d="M13 13l6 6"></path>
        </svg>
        Select
      </button>
    </div>
  </div>
  
  <div class="selection-status" id="selectionStatus">
    <span id="selectionCount">0 quadrants selected</span>
    <span class="selection-limit">(max 4)</span>
    <button id="deselectAllBtn" class="deselect-btn" onclick="deselectAll()" disabled>Deselect All</button>
    <button id="generateBtn" class="generate-btn" onclick="generateSelected()" disabled>Generate</button>
  </div>
  
  <div class="grid-container {% if show_lines %}show-lines{% endif %} {% if show_coords %}show-coords{% endif %}" id="gridContainer">
    <div class="grid">
      {% for dy in range(ny) %}
        {% for dx in range(nx) %}
          {% set qx = x + dx %}
          {% set qy = y + dy %}
          {% set has_gen = tiles.get((dx, dy), False) %}
          <div class="tile {% if not has_gen %}placeholder{% endif %}" data-coords="{{ qx }},{{ qy }}">
            <span class="coords">({{ qx }}, {{ qy }})</span>
            {% if has_gen %}
              <img src="/tile/{{ qx }}/{{ qy }}?render={{ '1' if show_render else '0' }}" alt="Tile {{ qx }},{{ qy }}">
            {% endif %}
          </div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>
  
  <div class="info">
    <p>Showing {{ nx }}×{{ ny }} quadrants from ({{ x }}, {{ y }}) through ({{ x + nx - 1 }}, {{ y + ny - 1 }})</p>
    <p>Generation dir: {{ generation_dir }}</p>
  </div>
  
  <script>
    function getParams() {
      const x = document.getElementById('x').value;
      const y = document.getElementById('y').value;
      const nx = document.getElementById('nx').value;
      const ny = document.getElementById('ny').value;
      const sizePx = document.getElementById('sizePx').value;
      const showLines = document.getElementById('showLines').checked ? '1' : '0';
      const showCoords = document.getElementById('showCoords').checked ? '1' : '0';
      const showRender = document.getElementById('showRender').checked ? '1' : '0';
      return { x, y, nx, ny, sizePx, showLines, showCoords, showRender };
    }
    
    function goTo() {
      const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    function navigate(dx, dy) {
      const params = getParams();
      const x = parseInt(params.x) + dx;
      const y = parseInt(params.y) + dy;
      window.location.href = `?x=${x}&y=${y}&nx=${params.nx}&ny=${params.ny}&size=${params.sizePx}&lines=${params.showLines}&coords=${params.showCoords}&render=${params.showRender}`;
    }
    
    function toggleLines() {
      const container = document.getElementById('gridContainer');
      const showLines = document.getElementById('showLines').checked;
      container.classList.toggle('show-lines', showLines);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('lines', showLines ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleCoords() {
      const container = document.getElementById('gridContainer');
      const showCoords = document.getElementById('showCoords').checked;
      container.classList.toggle('show-coords', showCoords);
      
      // Update URL without reload
      const url = new URL(window.location);
      url.searchParams.set('coords', showCoords ? '1' : '0');
      history.replaceState({}, '', url);
    }
    
    function toggleRender() {
      // This requires a page reload to fetch different data
      const { x, y, nx, ny, sizePx, showLines, showCoords, showRender } = getParams();
      window.location.href = `?x=${x}&y=${y}&nx=${nx}&ny=${ny}&size=${sizePx}&lines=${showLines}&coords=${showCoords}&render=${showRender}`;
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      
      switch(e.key) {
        case 'ArrowLeft': navigate(-1, 0); break;
        case 'ArrowRight': navigate(1, 0); break;
        case 'ArrowUp': navigate(0, -1); break;
        case 'ArrowDown': navigate(0, 1); break;
        case 'l': case 'L':
          document.getElementById('showLines').click();
          break;
        case 'c': case 'C':
          document.getElementById('showCoords').click();
          break;
        case 'g': case 'G':
          document.getElementById('showRender').click();
          break;
        case 's': case 'S':
          toggleSelectTool();
          break;
        case 'Escape':
          if (selectToolActive) toggleSelectTool();
          break;
      }
    });
    
    // Select tool state
    let selectToolActive = false;
    const selectedQuadrants = new Set();
    const MAX_SELECTION = 4;
    
    function toggleSelectTool() {
      selectToolActive = !selectToolActive;
      const btn = document.getElementById('selectTool');
      const tiles = document.querySelectorAll('.tile');
      
      if (selectToolActive) {
        btn.classList.add('active');
        tiles.forEach(tile => tile.classList.add('selectable'));
      } else {
        btn.classList.remove('active');
        tiles.forEach(tile => tile.classList.remove('selectable'));
      }
    }
    
    function updateSelectionStatus() {
      const count = selectedQuadrants.size;
      const countEl = document.getElementById('selectionCount');
      const statusEl = document.getElementById('selectionStatus');
      const deselectBtn = document.getElementById('deselectAllBtn');
      const generateBtn = document.getElementById('generateBtn');
      
      countEl.textContent = `${count} quadrant${count !== 1 ? 's' : ''} selected`;
      statusEl.classList.toggle('empty', count === 0);
      deselectBtn.disabled = count === 0;
      generateBtn.disabled = count === 0;
    }
    
    function generateSelected() {
      if (selectedQuadrants.size === 0) return;
      
      const coords = Array.from(selectedQuadrants);
      console.log('Generate requested for:', coords);
      // TODO: Implement generation
    }
    
    function deselectAll() {
      selectedQuadrants.clear();
      document.querySelectorAll('.tile.selected').forEach(tile => {
        tile.classList.remove('selected');
      });
      updateSelectionStatus();
      console.log('Deselected all quadrants');
    }
    
    function toggleTileSelection(tileEl, qx, qy) {
      if (!selectToolActive) return;
      
      const key = `${qx},${qy}`;
      if (selectedQuadrants.has(key)) {
        selectedQuadrants.delete(key);
        tileEl.classList.remove('selected');
        console.log(`Deselected quadrant (${qx}, ${qy})`);
      } else {
        // Check if we've hit the max selection limit
        if (selectedQuadrants.size >= MAX_SELECTION) {
          console.log(`Cannot select more than ${MAX_SELECTION} quadrants`);
          return;
        }
        selectedQuadrants.add(key);
        tileEl.classList.add('selected');
        console.log(`Selected quadrant (${qx}, ${qy})`);
      }
      
      updateSelectionStatus();
      
      // Log current selection
      if (selectedQuadrants.size > 0) {
        console.log('Selected:', Array.from(selectedQuadrants).join('; '));
      }
    }
    
    // Setup tile click handlers
    document.querySelectorAll('.tile').forEach(tile => {
      tile.addEventListener('click', (e) => {
        if (!selectToolActive) return;
        e.preventDefault();
        e.stopPropagation();
        
        const coords = tile.dataset.coords.split(',').map(Number);
        toggleTileSelection(tile, coords[0], coords[1]);
      });
    });
    
    // Initialize selection status
    updateSelectionStatus();
  </script>
</body>
</html>
"""


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def get_quadrant_generation(x: int, y: int) -> bytes | None:
  """Get the generation bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT generation FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_render(x: int, y: int) -> bytes | None:
  """Get the render bytes for a quadrant."""
  conn = get_db_connection()
  try:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT render FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
      (x, y),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else None
  finally:
    conn.close()


def get_quadrant_data(x: int, y: int, use_render: bool = False) -> bytes | None:
  """Get either render or generation bytes for a quadrant."""
  if use_render:
    return get_quadrant_render(x, y)
  return get_quadrant_generation(x, y)


@app.route("/")
def index():
  """Main page showing nx×ny grid of tiles."""
  x = request.args.get("x", 0, type=int)
  y = request.args.get("y", 0, type=int)
  # Default fallback if nx/ny not present
  n_default = request.args.get("n", 2, type=int)
  nx = request.args.get("nx", n_default, type=int)
  ny = request.args.get("ny", n_default, type=int)
  size_px = request.args.get("size", 256, type=int)
  show_lines = request.args.get("lines", "1") == "1"
  show_coords = request.args.get("coords", "1") == "1"
  show_render = request.args.get("render", "0") == "1"

  # Clamp nx/ny to reasonable bounds
  nx = max(1, min(nx, 20))
  ny = max(1, min(ny, 20))

  # Check which tiles have data (generation or render based on mode)
  tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      data = get_quadrant_data(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = data is not None

  return render_template_string(
    HTML_TEMPLATE,
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    generation_dir=str(GENERATION_DIR),
  )


@app.route("/tile/<x>/<y>")
def tile(x: str, y: str):
  """Serve a tile image (generation or render based on query param)."""
  try:
    qx, qy = int(x), int(y)
  except ValueError:
    return Response("Invalid coordinates", status=400)

  use_render = request.args.get("render", "0") == "1"
  data = get_quadrant_data(qx, qy, use_render=use_render)

  if data is None:
    return Response("Not found", status=404)

  return Response(data, mimetype="image/png")


def main():
  global GENERATION_DIR

  parser = argparse.ArgumentParser(description="View generated tiles in a grid.")
  parser.add_argument(
    "generation_dir",
    type=Path,
    help="Path to the generation directory containing quadrants.db",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to run the server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Server: http://{args.host}:{args.port}/")
  print("   Press Ctrl+C to stop")

  app.run(host=args.host, port=args.port, debug=True)
  return 0


if __name__ == "__main__":
  exit(main())