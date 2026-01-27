"""
Quick CSV editor for omni.csv files.

Allows viewing images and editing prompts.

Usage:
  uv run python src/isometric_nyc/synthetic_data/csv_editor_app.py <path_to_dataset_dir>

Example:
  uv run python src/isometric_nyc/synthetic_data/csv_editor_app.py synthetic_data/datasets/noised_terrain
"""

import argparse
import csv
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

app = Flask(__name__)

# Global state
CSV_PATH = None
DATASET_DIR = None
CSV_DATA = []


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>CSV Prompt Editor</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #1a1a1a;
      color: #e0e0e0;
      padding: 20px;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
    }

    h1 {
      margin-bottom: 20px;
      color: #fff;
    }

    .controls {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      align-items: center;
    }

    button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
    }

    button:hover {
      background: #2563eb;
    }

    button:disabled {
      background: #4b5563;
      cursor: not-allowed;
    }

    .save-btn {
      background: #10b981;
      margin-left: auto;
    }

    .save-btn:hover {
      background: #059669;
    }

    .status {
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 14px;
    }

    .status.success {
      background: #065f46;
      color: #d1fae5;
    }

    .row-info {
      color: #9ca3af;
      font-size: 14px;
    }

    .editor {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }

    .image-panel {
      background: #262626;
      border-radius: 8px;
      padding: 20px;
    }

    .image-panel h2 {
      font-size: 14px;
      text-transform: uppercase;
      color: #9ca3af;
      margin-bottom: 10px;
      letter-spacing: 0.5px;
    }

    .image-container {
      width: 512px;
      height: 512px;
      background: #1a1a1a;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .image-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: pixelated;
    }

    .prompt-editor {
      grid-column: 1 / -1;
      background: #262626;
      border-radius: 8px;
      padding: 20px;
    }

    .prompt-editor h2 {
      font-size: 14px;
      text-transform: uppercase;
      color: #9ca3af;
      margin-bottom: 10px;
      letter-spacing: 0.5px;
    }

    textarea {
      width: 100%;
      min-height: 100px;
      background: #1a1a1a;
      border: 2px solid #404040;
      border-radius: 6px;
      padding: 12px;
      color: #e0e0e0;
      font-family: inherit;
      font-size: 14px;
      resize: vertical;
    }

    textarea:focus {
      outline: none;
      border-color: #3b82f6;
    }

    .metadata {
      grid-column: 1 / -1;
      background: #262626;
      border-radius: 8px;
      padding: 20px;
      font-size: 13px;
      color: #9ca3af;
    }

    .metadata-row {
      display: flex;
      gap: 10px;
      margin-bottom: 5px;
    }

    .metadata-label {
      font-weight: 600;
      min-width: 120px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>CSV Prompt Editor</h1>

    <div class="controls">
      <button id="prevBtn" onclick="navigate(-1)">← Previous</button>
      <button id="nextBtn" onclick="navigate(1)">Next →</button>
      <span class="row-info" id="rowInfo">Row 1 of 0</span>
      <button class="save-btn" onclick="saveChanges()">Save CSV</button>
      <div id="status"></div>
    </div>

    <div class="editor">
      <div class="image-panel">
        <h2>Generation</h2>
        <div class="image-container">
          <img id="genImage" src="" alt="Generation">
        </div>
      </div>

      <div class="image-panel">
        <h2>Omni (Training Input)</h2>
        <div class="image-container">
          <img id="omniImage" src="" alt="Omni">
        </div>
      </div>

      <div class="prompt-editor">
        <h2>Prompt</h2>
        <textarea id="promptText" placeholder="Enter prompt..."></textarea>
      </div>

      <div class="metadata">
        <div class="metadata-row">
          <span class="metadata-label">Generation Path:</span>
          <span id="genPath"></span>
        </div>
        <div class="metadata-row">
          <span class="metadata-label">Omni Path:</span>
          <span id="omniPath"></span>
        </div>
      </div>
    </div>
  </div>

  <script>
    let csvData = [];
    let currentIndex = 0;

    async function loadData() {
      const response = await fetch('/api/data');
      csvData = await response.json();
      displayRow();
    }

    function displayRow() {
      if (csvData.length === 0) return;

      const row = csvData[currentIndex];

      document.getElementById('genImage').src = '/image/' + encodeURIComponent(row.generation);
      document.getElementById('omniImage').src = '/image/' + encodeURIComponent(row.omni);
      document.getElementById('promptText').value = row.prompt;
      document.getElementById('genPath').textContent = row.generation;
      document.getElementById('omniPath').textContent = row.omni;

      document.getElementById('rowInfo').textContent = `Row ${currentIndex + 1} of ${csvData.length}`;
      document.getElementById('prevBtn').disabled = currentIndex === 0;
      document.getElementById('nextBtn').disabled = currentIndex === csvData.length - 1;

      // Clear status when navigating
      document.getElementById('status').innerHTML = '';
    }

    function navigate(delta) {
      // Save current prompt before navigating
      csvData[currentIndex].prompt = document.getElementById('promptText').value;

      currentIndex += delta;
      currentIndex = Math.max(0, Math.min(currentIndex, csvData.length - 1));
      displayRow();
    }

    async function saveChanges() {
      // Save current prompt
      csvData[currentIndex].prompt = document.getElementById('promptText').value;

      const response = await fetch('/api/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(csvData),
      });

      const result = await response.json();

      const statusEl = document.getElementById('status');
      if (result.success) {
        statusEl.innerHTML = '<span class="status success">✓ Saved</span>';
        setTimeout(() => {
          statusEl.innerHTML = '';
        }, 2000);
      } else {
        statusEl.innerHTML = '<span class="status error">✗ Error saving</span>';
      }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft' && !e.target.matches('textarea')) {
        navigate(-1);
      } else if (e.key === 'ArrowRight' && !e.target.matches('textarea')) {
        navigate(1);
      } else if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        saveChanges();
      }
    });

    loadData();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
  return render_template_string(HTML_TEMPLATE)


@app.route("/api/data")
def get_data():
  return jsonify(CSV_DATA)


@app.route("/api/save", methods=["POST"])
def save_data():
  global CSV_DATA
  try:
    new_data = request.json
    CSV_DATA = new_data

    # Write back to CSV
    with open(CSV_PATH, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=["omni", "generation", "prompt"])
      writer.writeheader()
      writer.writerows(CSV_DATA)

    return jsonify({"success": True})
  except Exception as e:
    print(f"Error saving: {e}")
    return jsonify({"success": False, "error": str(e)}), 500


@app.route("/image/<path:filename>")
def serve_image(filename):
  """Serve images from the dataset directory."""
  file_path = DATASET_DIR / filename
  return send_from_directory(file_path.parent, file_path.name)


def main():
  global CSV_PATH, DATASET_DIR, CSV_DATA

  parser = argparse.ArgumentParser(description="CSV prompt editor for omni.csv")
  parser.add_argument(
    "dataset_dir",
    type=Path,
    help="Path to dataset directory containing omni.csv",
  )
  parser.add_argument(
    "--port", type=int, default=5000, help="Port to run the server on (default: 5000)"
  )
  args = parser.parse_args()

  DATASET_DIR = args.dataset_dir.resolve()
  CSV_PATH = DATASET_DIR / "omni.csv"

  if not DATASET_DIR.exists():
    print(f"❌ Dataset directory not found: {DATASET_DIR}")
    sys.exit(1)

  if not CSV_PATH.exists():
    print(f"❌ CSV file not found: {CSV_PATH}")
    sys.exit(1)

  # Load CSV data
  with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    CSV_DATA = list(reader)

  print("=" * 60)
  print("🎨 CSV PROMPT EDITOR")
  print(f"   Dataset: {DATASET_DIR}")
  print(f"   CSV: {CSV_PATH}")
  print(f"   Rows: {len(CSV_DATA)}")
  print("=" * 60)
  print(f"\n✨ Server running at http://localhost:{args.port}")
  print("   Press Ctrl+C to stop\n")
  print("Keyboard shortcuts:")
  print("   ← → : Navigate between rows")
  print("   Cmd/Ctrl + S : Save changes")
  print("=" * 60)

  app.run(debug=True, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
  main()
