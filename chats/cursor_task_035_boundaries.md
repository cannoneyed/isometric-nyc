# Task 035 boundaries
_Exported on 12/21/2025 at 11:26:08 EST from Cursor (2.2.43)_

---

**User**

Implement @tasks/035_boundaries.md 

---

**Cursor**

I'll implement this task step by step. Let me first explore the existing codebase to understand the structure and find any existing NYC boundary definitions.

Let me look at more files to understand the frontend structure:

Now I have a clear understanding of the codebase. Let me create a task list and start implementing:

```json
{
  "type": "FeatureCollection",
  "name": "NYC Boroughs",
  "description": "Simplified NYC borough boundaries (source: NYC Open Data, simplified using Douglas-Peucker algorithm)",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "Queens"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-73.8205, 40.801], [-73.8137, 40.7971], [-73.7964, 40.7957], [-73.7928, 40.7888],
          [-73.7817, 40.7911], [-73.7833, 40.7949], [-73.7769, 40.7964], [-73.7706, 40.7885],
          [-73.7763, 40.7885], [-73.7668, 40.7798], [-73.7651, 40.7738], [-73.7447, 40.7566],
          [-73.7551, 40.7675], [-73.7555, 40.7777], [-73.7508, 40.7829], [-73.7013, 40.7508],
          [-73.7, 40.7392], [-73.7077, 40.7278], [-73.7303, 40.7222], [-73.7269, 40.71],
          [-73.7256, 40.6796], [-73.7283, 40.663], [-73.7252, 40.652], [-73.7414, 40.6469],
          [-73.74, 40.6351], [-73.7437, 40.639], [-73.747, 40.6369], [-73.7473, 40.6432],
          [-73.7548, 40.6478], [-73.7474, 40.6414], [-73.7486, 40.6354], [-73.765, 40.6294],
          [-73.7708, 40.62], [-73.7717, 40.6236], [-73.7752, 40.6193], [-73.7717, 40.6236],
          [-73.7792, 40.6273], [-73.7852, 40.6201], [-73.7809, 40.6136], [-73.7914, 40.6076],
          [-73.8008, 40.6119], [-73.7946, 40.6226], [-73.7897, 40.6226], [-73.7826, 40.6302],
          [-73.7895, 40.6343], [-73.7981, 40.631], [-73.7908, 40.6343], [-73.8212, 40.6491],
          [-73.8223, 40.6594], [-73.8113, 40.6606], [-73.8235, 40.6601], [-73.8231, 40.648],
          [-73.8309, 40.6516], [-73.8302, 40.6598], [-73.8319, 40.6549], [-73.831, 40.6582],
          [-73.8334, 40.6581], [-73.8311, 40.6486], [-73.8343, 40.6481], [-73.8387, 40.6625],
          [-73.8359, 40.6455], [-73.8495, 40.6441], [-73.8524, 40.6474], [-73.8494, 40.6514],
          [-73.8559, 40.6513], [-73.8614, 40.6564], [-73.8596, 40.6585], [-73.8632, 40.6583],
          [-73.8557, 40.6639], [-73.8576, 40.6717], [-73.8604, 40.6713], [-73.866, 40.6819],
          [-73.8689, 40.6952], [-73.8965, 40.6823], [-73.9058, 40.6941], [-73.9043, 40.6957],
          [-73.9219, 40.7094], [-73.9241, 40.7151], [-73.9206, 40.716], [-73.9252, 40.7223],
          [-73.9204, 40.7237], [-73.938, 40.7305], [-73.9459, 40.7375], [-73.9386, 40.7429],
          [-73.9472, 40.7379], [-73.9626, 40.7388], [-73.9568, 40.7488], [-73.9532, 40.7477],
          [-73.9562, 40.7497], [-73.9435, 40.7645], [-73.9349, 40.77], [-73.9378, 40.7736],
          [-73.9364, 40.7769], [-73.9286, 40.7766], [-73.9099, 40.7909], [-73.8959, 40.7856],
          [-73.9026, 40.7803], [-73.8944, 40.7842], [-73.8953, 40.7816], [-73.8917, 40.7822],
          [-73.8925, 40.7774], [-73.8918, 40.7749], [-73.8901, 40.7778], [-73.8895, 40.7735],
          [-73.8839, 40.7741], [-73.8849, 40.78], [-73.8787, 40.7806], [-73.8797, 40.7829],
          [-73.8752, 40.7815], [-73.8679, 40.7874], [-73.8726, 40.7808], [-73.8551, 40.7722],
          [-73.8582, 40.7703], [-73.8562, 40.7689], [-73.8628, 40.7668], [-73.8507, 40.7594],
          [-73.8391, 40.766], [-73.8478, 40.7668], [-73.8513, 40.7718], [-73.849, 40.778],
          [-73.8524, 40.7792], [-73.8492, 40.7823], [-73.8553, 40.782], [-73.8595, 40.7855],
          [-73.8528, 40.7882], [-73.8537, 40.794], [-73.8379, 40.7985], [-73.8403, 40.7976],
          [-73.8373, 40.789], [-73.8329, 40.7885], [-73.8279, 40.7929], [-73.8295, 40.7968],
          [-73.8205, 40.801]
        ]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Brooklyn"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-73.9544, 40.7391], [-73.9292, 40.7274], [-73.9246, 40.7192], [-73.9322, 40.715],
          [-73.9333, 40.711], [-73.9304, 40.7087], [-73.9309, 40.7149], [-73.9236, 40.7173],
          [-73.9219, 40.7094], [-73.9043, 40.6957], [-73.9058, 40.6941], [-73.8965, 40.6823],
          [-73.8684, 40.6947], [-73.866, 40.6819], [-73.8604, 40.6713], [-73.8576, 40.6717],
          [-73.8557, 40.6639], [-73.8633, 40.6569], [-73.857, 40.6506], [-73.8572, 40.6437],
          [-73.8656, 40.6388], [-73.8793, 40.6546], [-73.868, 40.6381], [-73.8744, 40.6367],
          [-73.891, 40.6494], [-73.8771, 40.636], [-73.8836, 40.6309], [-73.8837, 40.6276],
          [-73.8875, 40.6283], [-73.8958, 40.6225], [-73.9175, 40.6314], [-73.8965, 40.6214],
          [-73.8959, 40.6145], [-73.891, 40.6128], [-73.894, 40.6112], [-73.8897, 40.6115],
          [-73.8933, 40.6067], [-73.8998, 40.6055], [-73.9013, 40.6117], [-73.9088, 40.6174],
          [-73.9016, 40.6058], [-73.9137, 40.604], [-73.9171, 40.6088], [-73.9142, 40.6149],
          [-73.9195, 40.6077], [-73.9089, 40.6016], [-73.884, 40.6056], [-73.8767, 40.5849],
          [-73.8797, 40.5801], [-73.8955, 40.5768], [-73.899, 40.5877], [-73.9115, 40.5861],
          [-73.9151, 40.5988], [-73.9215, 40.6019], [-73.9249, 40.5998], [-73.9302, 40.604],
          [-73.9321, 40.6027], [-73.9142, 40.5892], [-73.9248, 40.5856], [-73.9281, 40.5877],
          [-73.924, 40.5911], [-73.9286, 40.5886], [-73.9326, 40.5949], [-73.9313, 40.5868],
          [-73.9135, 40.5861], [-73.9125, 40.5834], [-73.9173, 40.5832], [-73.9115, 40.5818],
          [-73.9535, 40.583], [-73.9326, 40.5813], [-73.9311, 40.5762], [-74.0021, 40.5696],
          [-74.012, 40.5749], [-74.0119, 40.5797], [-74.006, 40.582], [-73.9884, 40.5788],
          [-73.986, 40.5817], [-73.9916, 40.5811], [-73.9903, 40.5846], [-74.0002, 40.5831],
          [-73.9941, 40.5887], [-74.0004, 40.5865], [-73.9984, 40.5935], [-74.036, 40.6093],
          [-74.0416, 40.6206], [-74.0368, 40.639], [-74.0388, 40.6396], [-74.0341, 40.6443],
          [-74.0282, 40.644], [-74.026, 40.6463], [-74.0294, 40.6488], [-74.0247, 40.6471],
          [-74.0261, 40.651], [-74.0229, 40.6512], [-74.0254, 40.6527], [-74.0212, 40.6508],
          [-74.0232, 40.6541], [-74.0179, 40.6541], [-74.0198, 40.6565], [-74.0167, 40.6553],
          [-74.0191, 40.6574], [-74.015, 40.657], [-74.0175, 40.6595], [-74.0155, 40.6608],
          [-74.0086, 40.6595], [-74.0096, 40.6633], [-74.0035, 40.6622], [-74.0077, 40.6648],
          [-73.9995, 40.6644], [-74.0029, 40.6665], [-73.9988, 40.6716], [-74.0029, 40.6673],
          [-74.007, 40.6661], [-74.0054, 40.6709], [-74.0075, 40.6671], [-74.0169, 40.6648],
          [-74.0193, 40.6716], [-74.0165, 40.6649], [-74.0103, 40.6691], [-74.0157, 40.671],
          [-74.013, 40.6733], [-74.0163, 40.6729], [-74.015, 40.6747], [-74.0188, 40.6723],
          [-74.0193, 40.6796], [-74.0128, 40.6836], [-74.013, 40.6804], [-74.0097, 40.6833],
          [-74.0119, 40.6839], [-74.001, 40.6901], [-73.9946, 40.7044], [-73.9783, 40.706],
          [-73.9757, 40.6996], [-73.9708, 40.7], [-73.9728, 40.7027], [-73.9695, 40.7005],
          [-73.9746, 40.7077], [-73.9711, 40.7059], [-73.9723, 40.7091], [-73.9675, 40.7034],
          [-73.97, 40.7105], [-73.9673, 40.717], [-73.9621, 40.7251], [-73.9579, 40.7246],
          [-73.9616, 40.7259], [-73.9623, 40.734], [-73.9544, 40.7391]
        ], [[-73.9039, 40.6248], [-73.9039, 40.6248]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Staten Island"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-74.0822, 40.6483], [-74.0717, 40.645], [-74.0703, 40.6425], [-74.0731, 40.6419],
          [-74.0702, 40.642], [-74.0729, 40.6409], [-74.07, 40.6411], [-74.0731, 40.6405],
          [-74.0735, 40.6372], [-74.0705, 40.6372], [-74.0734, 40.6369], [-74.073, 40.6303],
          [-74.068, 40.6288], [-74.0729, 40.63], [-74.0723, 40.6246], [-74.0538, 40.6059],
          [-74.0522, 40.5998], [-74.0734, 40.5783], [-74.091, 40.5667], [-74.1048, 40.553],
          [-74.1073, 40.5541], [-74.1128, 40.5479], [-74.1225, 40.5448], [-74.1369, 40.5293],
          [-74.1405, 40.535], [-74.1341, 40.5355], [-74.128, 40.5407], [-74.1307, 40.5463],
          [-74.1379, 40.546], [-74.1423, 40.5423], [-74.1413, 40.5371], [-74.1776, 40.5192],
          [-74.1848, 40.5193], [-74.1952, 40.51], [-74.1994, 40.513], [-74.2076, 40.5118],
          [-74.2176, 40.5033], [-74.2468, 40.4961], [-74.2532, 40.5001], [-74.2556, 40.5077],
          [-74.2535, 40.5117], [-74.2494, 40.517], [-74.2398, 40.5201], [-74.244, 40.5249],
          [-74.242, 40.5343], [-74.248, 40.5431], [-74.2307, 40.5556], [-74.2186, 40.5547],
          [-74.212, 40.5584], [-74.2046, 40.5893], [-74.1975, 40.5968], [-74.2028, 40.6083],
          [-74.2008, 40.6303], [-74.1946, 40.6374], [-74.18, 40.6453], [-74.1796, 40.6419],
          [-74.1744, 40.6451], [-74.1713, 40.6429], [-74.173, 40.6405], [-74.1661, 40.6424],
          [-74.1528, 40.6371], [-74.1514, 40.6394], [-74.1486, 40.6374], [-74.1344, 40.6419],
          [-74.127, 40.6393], [-74.1097, 40.6455], [-74.0822, 40.6483]
        ]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Bronx"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-73.8729, 40.9044], [-73.8595, 40.9005], [-73.8511, 40.9104], [-73.8535, 40.9075],
          [-73.8414, 40.9042], [-73.8384, 40.8941], [-73.793, 40.8834], [-73.7974, 40.8751],
          [-73.8025, 40.8705], [-73.8034, 40.8725], [-73.803, 40.8692], [-73.8084, 40.8712],
          [-73.8031, 40.8658], [-73.7948, 40.8732], [-73.7929, 40.8793], [-73.7896, 40.8813],
          [-73.7844, 40.8784], [-73.7872, 40.8727], [-73.7836, 40.8738], [-73.7832, 40.8707],
          [-73.7914, 40.8681], [-73.7919, 40.8608], [-73.7987, 40.8549], [-73.8003, 40.8481],
          [-73.8042, 40.8533], [-73.8048, 40.8616], [-73.8077, 40.8585], [-73.8151, 40.8631],
          [-73.8123, 40.8542], [-73.8164, 40.8541], [-73.8171, 40.8513], [-73.815, 40.8455],
          [-73.8117, 40.8467], [-73.8144, 40.8431], [-73.8182, 40.8443], [-73.8147, 40.8414],
          [-73.8149, 40.831], [-73.8099, 40.8282], [-73.814, 40.8242], [-73.807, 40.8254],
          [-73.8046, 40.819], [-73.7974, 40.8158], [-73.803, 40.8124], [-73.807, 40.8165],
          [-73.8051, 40.8129], [-73.7909, 40.8073], [-73.7906, 40.8046], [-73.8162, 40.8138],
          [-73.8302, 40.8108], [-73.8316, 40.8049], [-73.8374, 40.8062], [-73.8423, 40.8291],
          [-73.8389, 40.8337], [-73.8397, 40.8396], [-73.8431, 40.8298], [-73.84, 40.8199],
          [-73.8462, 40.8107], [-73.8554, 40.8143], [-73.8491, 40.8123], [-73.8481, 40.8049],
          [-73.8561, 40.8046], [-73.8592, 40.8063], [-73.859, 40.8103], [-73.8594, 40.8082],
          [-73.8677, 40.8106], [-73.8765, 40.8161], [-73.8681, 40.8068], [-73.8723, 40.8002],
          [-73.8951, 40.8069], [-73.9022, 40.8049], [-73.9117, 40.7966], [-73.9276, 40.8027],
          [-73.9329, 40.8101], [-73.9331, 40.8352], [-73.9067, 40.8758], [-73.9103, 40.879],
          [-73.9178, 40.8757], [-73.9249, 40.8789], [-73.9115, 40.9138], [-73.8729, 40.9044]
        ]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Manhattan"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-73.9264, 40.8776], [-73.9224, 40.8768], [-73.9213, 40.8731], [-73.9209, 40.8755],
          [-73.9104, 40.8715], [-73.9194, 40.8589], [-73.9216, 40.8601], [-73.9349, 40.8352],
          [-73.9343, 40.8096], [-73.929, 40.8011], [-73.9293, 40.7959], [-73.9432, 40.7833],
          [-73.9429, 40.7747], [-73.9714, 40.7438], [-73.9745, 40.7359], [-73.9715, 40.7274],
          [-73.9767, 40.7115], [-73.9981, 40.7085], [-74.0083, 40.7007], [-74.0151, 40.7003],
          [-74.0193, 40.7061], [-74.0167, 40.7186], [-74.0131, 40.7189], [-74.0117, 40.7259],
          [-74.0152, 40.7264], [-74.0115, 40.7264], [-74.0144, 40.7285], [-74.0109, 40.733],
          [-74.014, 40.7333], [-74.0107, 40.7334], [-74.0127, 40.7407], [-74.0095, 40.7407],
          [-74.0121, 40.7436], [-74.0089, 40.7439], [-74.0118, 40.7456], [-74.0095, 40.7458],
          [-74.0116, 40.7466], [-74.0092, 40.7468], [-74.0114, 40.7476], [-74.0091, 40.7478],
          [-74.01, 40.7529], [-74.0048, 40.7578], [-74.007, 40.7592], [-74.001, 40.7634],
          [-74.0031, 40.7647], [-73.9996, 40.7636], [-74.0023, 40.7656], [-73.9983, 40.7656],
          [-74.0016, 40.767], [-73.9981, 40.7659], [-74.0006, 40.7682], [-73.9972, 40.7671],
          [-73.9997, 40.7695], [-73.9971, 40.7687], [-73.999, 40.7705], [-73.9949, 40.7715],
          [-73.9967, 40.7735], [-73.9942, 40.7725], [-73.9962, 40.774], [-73.9923, 40.7751],
          [-73.9889, 40.7797], [-73.9916, 40.7796], [-73.9582, 40.8223], [-73.9461, 40.8439],
          [-73.947, 40.8505], [-73.9419, 40.8539], [-73.9264, 40.8776]
        ]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "Queens (South)"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-73.7667, 40.6143], [-73.7602, 40.6108], [-73.7616, 40.609], [-73.7458, 40.612],
          [-73.7382, 40.6027], [-73.7381, 40.5977], [-73.7418, 40.5967], [-73.7381, 40.5974],
          [-73.7376, 40.5944], [-73.8135, 40.5835], [-73.8703, 40.5642], [-73.9069, 40.5556],
          [-73.9407, 40.5418], [-73.9399, 40.5546], [-73.9262, 40.5616], [-73.912, 40.5659],
          [-73.9015, 40.563], [-73.8926, 40.5686], [-73.8778, 40.5688], [-73.8504, 40.5821],
          [-73.8393, 40.5819], [-73.8215, 40.5868], [-73.8096, 40.5936], [-73.8057, 40.5908],
          [-73.808, 40.5942], [-73.8024, 40.5926], [-73.8052, 40.5969], [-73.803, 40.5988],
          [-73.7864, 40.6032], [-73.7903, 40.5992], [-73.7898, 40.5947], [-73.7786, 40.6096],
          [-73.7742, 40.6089], [-73.7747, 40.6043], [-73.7813, 40.5978], [-73.775, 40.6017],
          [-73.7736, 40.5981], [-73.7681, 40.5978], [-73.7712, 40.5994], [-73.769, 40.6094],
          [-73.7742, 40.6118], [-73.7667, 40.6143]
        ]]
      }
    }
  ]
}

```

```python
"""
Bounds utilities for loading and managing geographic boundary files.
"""

import json
from pathlib import Path
from typing import Any

# Default bounds directory
BOUNDS_DIR = Path(__file__).parent

# Default NYC boundary file
DEFAULT_BOUNDS_FILE = BOUNDS_DIR / "nyc.json"


def load_bounds(bounds_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load a boundary GeoJSON file.

    Args:
        bounds_path: Path to the bounds JSON file. If None, loads the default
                    NYC boundary.

    Returns:
        GeoJSON dictionary with the boundary features.

    Raises:
        FileNotFoundError: If the bounds file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if bounds_path is None:
        bounds_path = DEFAULT_BOUNDS_FILE
    elif isinstance(bounds_path, str):
        bounds_path = Path(bounds_path)

    if not bounds_path.exists():
        raise FileNotFoundError(f"Bounds file not found: {bounds_path}")

    with open(bounds_path) as f:
        return json.load(f)


def get_bounds_dir() -> Path:
    """Get the path to the bounds directory."""
    return BOUNDS_DIR


def list_bounds_files() -> list[Path]:
    """List all available bounds JSON files."""
    return list(BOUNDS_DIR.glob("*.json"))


def save_bounds(
    bounds: dict[str, Any],
    name: str,
    bounds_dir: Path | None = None,
) -> Path:
    """
    Save a boundary GeoJSON to a file.

    Args:
        bounds: GeoJSON dictionary to save.
        name: Name for the bounds file (without .json extension).
        bounds_dir: Directory to save to. Defaults to the standard bounds dir.

    Returns:
        Path to the saved file.
    """
    if bounds_dir is None:
        bounds_dir = BOUNDS_DIR

    # Ensure .json extension
    if not name.endswith(".json"):
        name = f"{name}.json"

    output_path = bounds_dir / name

    with open(output_path, "w") as f:
        json.dump(bounds, f, indent=2)

    return output_path

```

```python
"""
Debug Map - Visualize generated quadrants on a real map of NYC.

This script creates an interactive web visualization showing which quadrants
have been generated by overlaying polygons on a real map using Leaflet.js.
Hovering over quadrants shows their x,y coordinates.

Usage:
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir] --bounds path/to/bounds.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --bounds: Path to a custom bounds GeoJSON file (default: NYC boundary)

Output:
  Creates debug_map.html in the generation directory
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.shared import calculate_offset
# fmt: on


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_generated_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int, int]]:
  """
  Get all quadrants that have been generated (have generation data).

  Returns list of (quadrant_x, quadrant_y, water_status) tuples.
  water_status: -1 = explicit not water, 0 = unset, 1 = water
  """
  cursor = conn.cursor()

  # Check if is_water column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  has_water_column = "is_water" in columns

  if has_water_column:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y, COALESCE(is_water, 0)
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
  else:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], 0) for row in cursor.fetchall()]


def get_pending_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int]]:
  """
  Get all unique quadrants from pending queue items.

  Returns list of (quadrant_x, quadrant_y) tuples.
  """
  cursor = conn.cursor()

  # Check if generation_queue table exists
  cursor.execute("""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='generation_queue'
  """)
  if not cursor.fetchone():
    return []

  # Get all pending queue items and extract their quadrants
  cursor.execute("""
    SELECT quadrants FROM generation_queue
    WHERE status = 'pending'
  """)

  pending_set: set[tuple[int, int]] = set()
  for (quadrants_json,) in cursor.fetchall():
    quadrants = json.loads(quadrants_json)
    for qx, qy in quadrants:
      pending_set.add((qx, qy))

  return sorted(pending_set)


def calculate_quadrant_corners(
  config: dict, quadrant_x: int, quadrant_y: int
) -> list[tuple[float, float]]:
  """
  Calculate the 4 geographic corners of a quadrant.

  The isometric projection means the quadrant appears as a parallelogram
  in geographic space. Returns corners in order: TL, TR, BR, BL.

  Args:
    config: Generation config dictionary
    quadrant_x: X coordinate of the quadrant
    quadrant_y: Y coordinate of the quadrant

  Returns:
    List of 4 (lat, lng) tuples representing the corners
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant is half a tile in size
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  # Calculate the pixel offsets for the 4 corners of this quadrant
  # Quadrant (0,0) has its bottom-right corner at the seed position
  # So we need to calculate corners relative to the seed
  base_x_px = quadrant_x * quadrant_width_px
  base_y_px = -quadrant_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels (relative to the quadrant's anchor)
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL: left and up from anchor
    (0, quadrant_height_px),  # TR: up from anchor
    (0, 0),  # BR: at anchor
    (-quadrant_width_px, 0),  # BL: left from anchor
  ]

  corners = []
  for dx, dy in corner_offsets:
    shift_x_px = base_x_px + dx
    shift_y_px = base_y_px + dy

    lat, lng = calculate_offset(
      seed_lat,
      seed_lng,
      shift_x_px,
      shift_y_px,
      view_height_meters,
      height_px,
      azimuth,
      elevation,
    )
    corners.append((lat, lng))

  return corners


def generate_html(
  config: dict,
  quadrant_polygons: list[tuple[int, int, list[tuple[float, float]], int]],
  pending_polygons: list[tuple[int, int, list[tuple[float, float]]]],
  seed_lat: float,
  seed_lng: float,
  center_lat: float,
  center_lng: float,
) -> str:
  """Generate the HTML content for the debug map."""

  # Convert quadrant data to JSON for JavaScript
  # water_status: -1 = explicit not water, 0 = unset, 1 = water
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

  # Convert pending quadrant data to JSON for JavaScript
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

  # NYC boundary GeoJSON for the map
  nyc_boundary_json = json.dumps(NYC_BOUNDARY_GEOJSON)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Debug Map - {config.get("name", "Generation")}</title>
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

    .info-panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      color: #fff;
      min-width: 180px;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .info-panel h3 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 8px;
    }}

    .coords {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 28px;
      font-weight: 600;
      color: #fff;
      display: flex;
      gap: 8px;
      align-items: baseline;
    }}

    .coords .label {{
      font-size: 14px;
      color: rgba(255, 255, 255, 0.4);
    }}

    .coords .x {{
      color: #ff6b6b;
    }}

    .coords .y {{
      color: #4ecdc4;
    }}

    .coords .separator {{
      color: rgba(255, 255, 255, 0.3);
      font-size: 20px;
    }}

    .no-hover {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: rgba(255, 255, 255, 0.3);
      font-style: italic;
    }}

    .stats-panel {{
      position: fixed;
      bottom: 16px;
      left: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .stats-panel .title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}

    .stats-panel .count {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: rgba(255, 255, 255, 0.6);
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

    .legend-swatch.quadrant {{
      background: rgba(255, 107, 107, 0.35);
      border: 2px solid rgba(255, 107, 107, 0.7);
    }}

    .legend-swatch.water {{
      background: rgba(59, 130, 246, 0.35);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.protected {{
      background: rgba(220, 38, 38, 0.25);
      border: 2px dashed rgba(220, 38, 38, 0.7);
    }}

    .legend-swatch.seed {{
      background: #10b981;
      border: 2px solid #fff;
      border-radius: 50%;
    }}

    .legend-swatch.nyc-boundary {{
      background: rgba(59, 130, 246, 0.15);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.pending {{
      background: rgba(168, 85, 247, 0.35);
      border: 2px solid rgba(168, 85, 247, 0.7);
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="info-panel">
    <h3>Quadrant</h3>
    <div id="coords-display" class="no-hover">Hover over a tile</div>
  </div>

  <div class="stats-panel">
    <div class="title">{config.get("name", "Debug Map")}</div>
    <div class="count">{len(quadrant_polygons)} quadrants generated</div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch nyc-boundary"></div>
      <span>NYC Boundary</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch quadrant"></div>
      <span>Generated Quadrant</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch water"></div>
      <span>Water Tile</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch protected"></div>
      <span>Protected (Not Water)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch pending"></div>
      <span>Pending Generation</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch seed"></div>
      <span>Seed Point</span>
    </div>
  </div>

  <script>
    const quadrants = {quadrants_json};
    const pendingQuadrants = {pending_json};
    const nycBoundary = {nyc_boundary_json};
    const seedLat = {seed_lat};
    const seedLng = {seed_lng};
    const centerLat = {center_lat};
    const centerLng = {center_lng};

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView([centerLat, centerLng], 15);

    // Add dark tile layer
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }}).addTo(map);

    // Add NYC boundary layer (underneath quadrants)
    const nycBoundaryStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.1,
      color: '#3b82f6',
      weight: 2,
      opacity: 0.6
    }};

    L.geoJSON(nycBoundary, {{
      style: nycBoundaryStyle
    }}).addTo(map);

    const coordsDisplay = document.getElementById('coords-display');

    // Style for normal state (red for land)
    const normalStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.25,
      color: '#ff6b6b',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for hover state (land)
    const hoverStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for water tiles (blue)
    const waterStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.35,
      color: '#3b82f6',
      weight: 1.5,
      opacity: 0.7
    }};

    // Style for water hover state
    const waterHoverStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.6,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for protected (explicit not water) tiles
    const protectedStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.2,
      color: '#dc2626',
      weight: 1.5,
      opacity: 0.6,
      dashArray: '5, 5'
    }};

    // Style for protected hover state
    const protectedHoverStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.4,
      color: '#fff',
      weight: 2,
      opacity: 1,
      dashArray: '5, 5'
    }};

    // Style for pending quadrants (purple)
    const pendingStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.25,
      color: '#a855f7',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for pending hover state
    const pendingHoverStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Helper to get style based on water_status
    function getStyle(waterStatus) {{
      if (waterStatus === 1) return waterStyle;
      if (waterStatus === -1) return protectedStyle;
      return normalStyle;
    }}

    function getHoverStyle(waterStatus) {{
      if (waterStatus === 1) return waterHoverStyle;
      if (waterStatus === -1) return protectedHoverStyle;
      return hoverStyle;
    }}

    // Add quadrant polygons
    quadrants.forEach(q => {{
      const style = getStyle(q.water_status);
      const hStyle = getHoverStyle(q.water_status);
      const polygon = L.polygon(q.corners, style).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(hStyle);
        let statusLabel = '';
        if (q.water_status === 1) {{
          statusLabel = ' <span style="color: #3b82f6;">💧</span>';
        }} else if (q.water_status === -1) {{
          statusLabel = ' <span style="color: #dc2626;">🛡️</span>';
        }}
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            ${{statusLabel}}
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(style);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add pending quadrant polygons (purple)
    pendingQuadrants.forEach(q => {{
      const polygon = L.polygon(q.corners, pendingStyle).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(pendingHoverStyle);
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            <span style="color: #a855f7;">⏳</span>
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(pendingStyle);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add seed point marker
    const seedIcon = L.divIcon({{
      className: 'seed-marker',
      html: `<div style="
        width: 14px;
        height: 14px;
        background: #10b981;
        border: 3px solid #fff;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      "></div>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    }});

    L.marker([seedLat, seedLng], {{ icon: seedIcon }})
      .addTo(map)
      .bindPopup('Seed Point');

    // Fit map to quadrant bounds if we have quadrants
    const allQuadrantCorners = [...quadrants, ...pendingQuadrants];
    if (allQuadrantCorners.length > 0) {{
      const allLatLngs = allQuadrantCorners.flatMap(q => q.corners);
      const bounds = L.latLngBounds(allLatLngs);
      map.fitBounds(bounds.pad(0.1));
    }}
  </script>
</body>
</html>
"""
  return html


def create_debug_map(
  generation_dir: Path,
  output_path: Path | None = None,
) -> Path:
  """
  Create an interactive debug map showing generated quadrants overlaid on NYC.

  Args:
    generation_dir: Path to the generation directory
    output_path: Optional output path (default: generation_dir/debug_map.html)

  Returns:
    Path to the generated HTML file
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"📂 Loaded generation config: {config.get('name', 'unnamed')}")
    print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
    print(
      f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
      f"elevation={config['camera_elevation_degrees']}°"
    )

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"\n📊 Found {len(generated)} generated quadrants")

    if not generated:
      print("⚠️  No generated quadrants found. Creating empty map.")

    # Get pending quadrants from the generation queue
    pending = get_pending_quadrants(conn)
    # Filter out any pending quadrants that are already generated
    generated_set = {(qx, qy) for qx, qy, _ in generated}
    pending = [(qx, qy) for qx, qy in pending if (qx, qy) not in generated_set]
    if pending:
      print(f"   ⏳ {len(pending)} pending quadrants in queue")

    # Calculate corners for all generated quadrants
    print("\n🔲 Calculating quadrant geographic footprints...")
    quadrant_polygons = []
    all_lats = []
    all_lngs = []
    water_count = 0
    protected_count = 0

    for qx, qy, water_status in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners, water_status))

      if water_status == 1:
        water_count += 1
      elif water_status == -1:
        protected_count += 1

      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate corners for pending quadrants
    pending_polygons = []
    for qx, qy in pending:
      corners = calculate_quadrant_corners(config, qx, qy)
      pending_polygons.append((qx, qy, corners))
      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    if water_count > 0:
      print(f"   💧 {water_count} water tiles")
    if protected_count > 0:
      print(f"   🛡️  {protected_count} protected tiles")

    # Calculate center for map
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]

    if all_lats and all_lngs:
      center_lat = (min(all_lats) + max(all_lats)) / 2
      center_lng = (min(all_lngs) + max(all_lngs)) / 2
    else:
      center_lat = seed_lat
      center_lng = seed_lng

    print("\n🗺️  Map center:")
    print(f"   Lat: {center_lat:.6f}")
    print(f"   Lng: {center_lng:.6f}")

    # Generate HTML
    print("\n🎨 Generating interactive map...")
    html_content = generate_html(
      config=config,
      quadrant_polygons=quadrant_polygons,
      pending_polygons=pending_polygons,
      seed_lat=seed_lat,
      seed_lng=seed_lng,
      center_lat=center_lat,
      center_lng=center_lng,
    )

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.html"

    # Save HTML file
    print(f"\n💾 Saving to {output_path}")
    output_path.write_text(html_content)

    print(f"\n✅ Debug map created: {output_path}")
    print(f"   Open in browser: file://{output_path.resolve()}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create an interactive debug map showing generated quadrants."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: generation_dir/debug_map.html)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None

  print(f"\n{'=' * 60}")
  print("🗺️  Debug Map Generator (Interactive)")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")

  try:
    create_debug_map(generation_dir, output_path)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

```python
"""
Debug Map - Visualize generated quadrants on a real map of NYC.

This script creates an interactive web visualization showing which quadrants
have been generated by overlaying polygons on a real map using Leaflet.js.
Hovering over quadrants shows their x,y coordinates.

Usage:
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir] --bounds path/to/bounds.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --bounds: Path to a custom bounds GeoJSON file (default: NYC boundary)

Output:
  Creates debug_map.html in the generation directory
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.shared import calculate_offset
# fmt: on


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_generated_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int, int]]:
  """
  Get all quadrants that have been generated (have generation data).

  Returns list of (quadrant_x, quadrant_y, water_status) tuples.
  water_status: -1 = explicit not water, 0 = unset, 1 = water
  """
  cursor = conn.cursor()

  # Check if is_water column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  has_water_column = "is_water" in columns

  if has_water_column:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y, COALESCE(is_water, 0)
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
  else:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], 0) for row in cursor.fetchall()]


def get_pending_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int]]:
  """
  Get all unique quadrants from pending queue items.

  Returns list of (quadrant_x, quadrant_y) tuples.
  """
  cursor = conn.cursor()

  # Check if generation_queue table exists
  cursor.execute("""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='generation_queue'
  """)
  if not cursor.fetchone():
    return []

  # Get all pending queue items and extract their quadrants
  cursor.execute("""
    SELECT quadrants FROM generation_queue
    WHERE status = 'pending'
  """)

  pending_set: set[tuple[int, int]] = set()
  for (quadrants_json,) in cursor.fetchall():
    quadrants = json.loads(quadrants_json)
    for qx, qy in quadrants:
      pending_set.add((qx, qy))

  return sorted(pending_set)


def calculate_quadrant_corners(
  config: dict, quadrant_x: int, quadrant_y: int
) -> list[tuple[float, float]]:
  """
  Calculate the 4 geographic corners of a quadrant.

  The isometric projection means the quadrant appears as a parallelogram
  in geographic space. Returns corners in order: TL, TR, BR, BL.

  Args:
    config: Generation config dictionary
    quadrant_x: X coordinate of the quadrant
    quadrant_y: Y coordinate of the quadrant

  Returns:
    List of 4 (lat, lng) tuples representing the corners
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant is half a tile in size
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  # Calculate the pixel offsets for the 4 corners of this quadrant
  # Quadrant (0,0) has its bottom-right corner at the seed position
  # So we need to calculate corners relative to the seed
  base_x_px = quadrant_x * quadrant_width_px
  base_y_px = -quadrant_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels (relative to the quadrant's anchor)
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL: left and up from anchor
    (0, quadrant_height_px),  # TR: up from anchor
    (0, 0),  # BR: at anchor
    (-quadrant_width_px, 0),  # BL: left from anchor
  ]

  corners = []
  for dx, dy in corner_offsets:
    shift_x_px = base_x_px + dx
    shift_y_px = base_y_px + dy

    lat, lng = calculate_offset(
      seed_lat,
      seed_lng,
      shift_x_px,
      shift_y_px,
      view_height_meters,
      height_px,
      azimuth,
      elevation,
    )
    corners.append((lat, lng))

  return corners


def generate_html(
  config: dict,
  quadrant_polygons: list[tuple[int, int, list[tuple[float, float]], int]],
  pending_polygons: list[tuple[int, int, list[tuple[float, float]]]],
  seed_lat: float,
  seed_lng: float,
  center_lat: float,
  center_lng: float,
  boundary_geojson: dict | None = None,
) -> str:
  """Generate the HTML content for the debug map."""

  # Convert quadrant data to JSON for JavaScript
  # water_status: -1 = explicit not water, 0 = unset, 1 = water
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

  # Convert pending quadrant data to JSON for JavaScript
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

  # Boundary GeoJSON for the map (default to NYC if not provided)
  if boundary_geojson is None:
    boundary_geojson = load_bounds()
  nyc_boundary_json = json.dumps(boundary_geojson)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Debug Map - {config.get("name", "Generation")}</title>
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

    .info-panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      color: #fff;
      min-width: 180px;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .info-panel h3 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 8px;
    }}

    .coords {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 28px;
      font-weight: 600;
      color: #fff;
      display: flex;
      gap: 8px;
      align-items: baseline;
    }}

    .coords .label {{
      font-size: 14px;
      color: rgba(255, 255, 255, 0.4);
    }}

    .coords .x {{
      color: #ff6b6b;
    }}

    .coords .y {{
      color: #4ecdc4;
    }}

    .coords .separator {{
      color: rgba(255, 255, 255, 0.3);
      font-size: 20px;
    }}

    .no-hover {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: rgba(255, 255, 255, 0.3);
      font-style: italic;
    }}

    .stats-panel {{
      position: fixed;
      bottom: 16px;
      left: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .stats-panel .title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}

    .stats-panel .count {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: rgba(255, 255, 255, 0.6);
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

    .legend-swatch.quadrant {{
      background: rgba(255, 107, 107, 0.35);
      border: 2px solid rgba(255, 107, 107, 0.7);
    }}

    .legend-swatch.water {{
      background: rgba(59, 130, 246, 0.35);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.protected {{
      background: rgba(220, 38, 38, 0.25);
      border: 2px dashed rgba(220, 38, 38, 0.7);
    }}

    .legend-swatch.seed {{
      background: #10b981;
      border: 2px solid #fff;
      border-radius: 50%;
    }}

    .legend-swatch.nyc-boundary {{
      background: rgba(59, 130, 246, 0.15);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.pending {{
      background: rgba(168, 85, 247, 0.35);
      border: 2px solid rgba(168, 85, 247, 0.7);
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="info-panel">
    <h3>Quadrant</h3>
    <div id="coords-display" class="no-hover">Hover over a tile</div>
  </div>

  <div class="stats-panel">
    <div class="title">{config.get("name", "Debug Map")}</div>
    <div class="count">{len(quadrant_polygons)} quadrants generated</div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch nyc-boundary"></div>
      <span>NYC Boundary</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch quadrant"></div>
      <span>Generated Quadrant</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch water"></div>
      <span>Water Tile</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch protected"></div>
      <span>Protected (Not Water)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch pending"></div>
      <span>Pending Generation</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch seed"></div>
      <span>Seed Point</span>
    </div>
  </div>

  <script>
    const quadrants = {quadrants_json};
    const pendingQuadrants = {pending_json};
    const nycBoundary = {nyc_boundary_json};
    const seedLat = {seed_lat};
    const seedLng = {seed_lng};
    const centerLat = {center_lat};
    const centerLng = {center_lng};

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView([centerLat, centerLng], 15);

    // Add dark tile layer
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }}).addTo(map);

    // Add NYC boundary layer (underneath quadrants)
    const nycBoundaryStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.1,
      color: '#3b82f6',
      weight: 2,
      opacity: 0.6
    }};

    L.geoJSON(nycBoundary, {{
      style: nycBoundaryStyle
    }}).addTo(map);

    const coordsDisplay = document.getElementById('coords-display');

    // Style for normal state (red for land)
    const normalStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.25,
      color: '#ff6b6b',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for hover state (land)
    const hoverStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for water tiles (blue)
    const waterStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.35,
      color: '#3b82f6',
      weight: 1.5,
      opacity: 0.7
    }};

    // Style for water hover state
    const waterHoverStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.6,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for protected (explicit not water) tiles
    const protectedStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.2,
      color: '#dc2626',
      weight: 1.5,
      opacity: 0.6,
      dashArray: '5, 5'
    }};

    // Style for protected hover state
    const protectedHoverStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.4,
      color: '#fff',
      weight: 2,
      opacity: 1,
      dashArray: '5, 5'
    }};

    // Style for pending quadrants (purple)
    const pendingStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.25,
      color: '#a855f7',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for pending hover state
    const pendingHoverStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Helper to get style based on water_status
    function getStyle(waterStatus) {{
      if (waterStatus === 1) return waterStyle;
      if (waterStatus === -1) return protectedStyle;
      return normalStyle;
    }}

    function getHoverStyle(waterStatus) {{
      if (waterStatus === 1) return waterHoverStyle;
      if (waterStatus === -1) return protectedHoverStyle;
      return hoverStyle;
    }}

    // Add quadrant polygons
    quadrants.forEach(q => {{
      const style = getStyle(q.water_status);
      const hStyle = getHoverStyle(q.water_status);
      const polygon = L.polygon(q.corners, style).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(hStyle);
        let statusLabel = '';
        if (q.water_status === 1) {{
          statusLabel = ' <span style="color: #3b82f6;">💧</span>';
        }} else if (q.water_status === -1) {{
          statusLabel = ' <span style="color: #dc2626;">🛡️</span>';
        }}
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            ${{statusLabel}}
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(style);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add pending quadrant polygons (purple)
    pendingQuadrants.forEach(q => {{
      const polygon = L.polygon(q.corners, pendingStyle).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(pendingHoverStyle);
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            <span style="color: #a855f7;">⏳</span>
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(pendingStyle);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add seed point marker
    const seedIcon = L.divIcon({{
      className: 'seed-marker',
      html: `<div style="
        width: 14px;
        height: 14px;
        background: #10b981;
        border: 3px solid #fff;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      "></div>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    }});

    L.marker([seedLat, seedLng], {{ icon: seedIcon }})
      .addTo(map)
      .bindPopup('Seed Point');

    // Fit map to quadrant bounds if we have quadrants
    const allQuadrantCorners = [...quadrants, ...pendingQuadrants];
    if (allQuadrantCorners.length > 0) {{
      const allLatLngs = allQuadrantCorners.flatMap(q => q.corners);
      const bounds = L.latLngBounds(allLatLngs);
      map.fitBounds(bounds.pad(0.1));
    }}
  </script>
</body>
</html>
"""
  return html


def create_debug_map(
  generation_dir: Path,
  output_path: Path | None = None,
) -> Path:
  """
  Create an interactive debug map showing generated quadrants overlaid on NYC.

  Args:
    generation_dir: Path to the generation directory
    output_path: Optional output path (default: generation_dir/debug_map.html)

  Returns:
    Path to the generated HTML file
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"📂 Loaded generation config: {config.get('name', 'unnamed')}")
    print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
    print(
      f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
      f"elevation={config['camera_elevation_degrees']}°"
    )

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"\n📊 Found {len(generated)} generated quadrants")

    if not generated:
      print("⚠️  No generated quadrants found. Creating empty map.")

    # Get pending quadrants from the generation queue
    pending = get_pending_quadrants(conn)
    # Filter out any pending quadrants that are already generated
    generated_set = {(qx, qy) for qx, qy, _ in generated}
    pending = [(qx, qy) for qx, qy in pending if (qx, qy) not in generated_set]
    if pending:
      print(f"   ⏳ {len(pending)} pending quadrants in queue")

    # Calculate corners for all generated quadrants
    print("\n🔲 Calculating quadrant geographic footprints...")
    quadrant_polygons = []
    all_lats = []
    all_lngs = []
    water_count = 0
    protected_count = 0

    for qx, qy, water_status in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners, water_status))

      if water_status == 1:
        water_count += 1
      elif water_status == -1:
        protected_count += 1

      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate corners for pending quadrants
    pending_polygons = []
    for qx, qy in pending:
      corners = calculate_quadrant_corners(config, qx, qy)
      pending_polygons.append((qx, qy, corners))
      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    if water_count > 0:
      print(f"   💧 {water_count} water tiles")
    if protected_count > 0:
      print(f"   🛡️  {protected_count} protected tiles")

    # Calculate center for map
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]

    if all_lats and all_lngs:
      center_lat = (min(all_lats) + max(all_lats)) / 2
      center_lng = (min(all_lngs) + max(all_lngs)) / 2
    else:
      center_lat = seed_lat
      center_lng = seed_lng

    print("\n🗺️  Map center:")
    print(f"   Lat: {center_lat:.6f}")
    print(f"   Lng: {center_lng:.6f}")

    # Generate HTML
    print("\n🎨 Generating interactive map...")
    html_content = generate_html(
      config=config,
      quadrant_polygons=quadrant_polygons,
      pending_polygons=pending_polygons,
      seed_lat=seed_lat,
      seed_lng=seed_lng,
      center_lat=center_lat,
      center_lng=center_lng,
    )

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.html"

    # Save HTML file
    print(f"\n💾 Saving to {output_path}")
    output_path.write_text(html_content)

    print(f"\n✅ Debug map created: {output_path}")
    print(f"   Open in browser: file://{output_path.resolve()}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create an interactive debug map showing generated quadrants."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: generation_dir/debug_map.html)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None

  print(f"\n{'=' * 60}")
  print("🗺️  Debug Map Generator (Interactive)")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")

  try:
    create_debug_map(generation_dir, output_path)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

```python
"""
Debug Map - Visualize generated quadrants on a real map of NYC.

This script creates an interactive web visualization showing which quadrants
have been generated by overlaying polygons on a real map using Leaflet.js.
Hovering over quadrants shows their x,y coordinates.

Usage:
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir] --bounds path/to/bounds.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --bounds: Path to a custom bounds GeoJSON file (default: NYC boundary)

Output:
  Creates debug_map.html in the generation directory
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.shared import calculate_offset
# fmt: on


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_generated_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int, int]]:
  """
  Get all quadrants that have been generated (have generation data).

  Returns list of (quadrant_x, quadrant_y, water_status) tuples.
  water_status: -1 = explicit not water, 0 = unset, 1 = water
  """
  cursor = conn.cursor()

  # Check if is_water column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  has_water_column = "is_water" in columns

  if has_water_column:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y, COALESCE(is_water, 0)
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
  else:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], 0) for row in cursor.fetchall()]


def get_pending_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int]]:
  """
  Get all unique quadrants from pending queue items.

  Returns list of (quadrant_x, quadrant_y) tuples.
  """
  cursor = conn.cursor()

  # Check if generation_queue table exists
  cursor.execute("""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='generation_queue'
  """)
  if not cursor.fetchone():
    return []

  # Get all pending queue items and extract their quadrants
  cursor.execute("""
    SELECT quadrants FROM generation_queue
    WHERE status = 'pending'
  """)

  pending_set: set[tuple[int, int]] = set()
  for (quadrants_json,) in cursor.fetchall():
    quadrants = json.loads(quadrants_json)
    for qx, qy in quadrants:
      pending_set.add((qx, qy))

  return sorted(pending_set)


def calculate_quadrant_corners(
  config: dict, quadrant_x: int, quadrant_y: int
) -> list[tuple[float, float]]:
  """
  Calculate the 4 geographic corners of a quadrant.

  The isometric projection means the quadrant appears as a parallelogram
  in geographic space. Returns corners in order: TL, TR, BR, BL.

  Args:
    config: Generation config dictionary
    quadrant_x: X coordinate of the quadrant
    quadrant_y: Y coordinate of the quadrant

  Returns:
    List of 4 (lat, lng) tuples representing the corners
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant is half a tile in size
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  # Calculate the pixel offsets for the 4 corners of this quadrant
  # Quadrant (0,0) has its bottom-right corner at the seed position
  # So we need to calculate corners relative to the seed
  base_x_px = quadrant_x * quadrant_width_px
  base_y_px = -quadrant_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels (relative to the quadrant's anchor)
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL: left and up from anchor
    (0, quadrant_height_px),  # TR: up from anchor
    (0, 0),  # BR: at anchor
    (-quadrant_width_px, 0),  # BL: left from anchor
  ]

  corners = []
  for dx, dy in corner_offsets:
    shift_x_px = base_x_px + dx
    shift_y_px = base_y_px + dy

    lat, lng = calculate_offset(
      seed_lat,
      seed_lng,
      shift_x_px,
      shift_y_px,
      view_height_meters,
      height_px,
      azimuth,
      elevation,
    )
    corners.append((lat, lng))

  return corners


def generate_html(
  config: dict,
  quadrant_polygons: list[tuple[int, int, list[tuple[float, float]], int]],
  pending_polygons: list[tuple[int, int, list[tuple[float, float]]]],
  seed_lat: float,
  seed_lng: float,
  center_lat: float,
  center_lng: float,
  boundary_geojson: dict | None = None,
) -> str:
  """Generate the HTML content for the debug map."""

  # Convert quadrant data to JSON for JavaScript
  # water_status: -1 = explicit not water, 0 = unset, 1 = water
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

  # Convert pending quadrant data to JSON for JavaScript
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

  # Boundary GeoJSON for the map (default to NYC if not provided)
  if boundary_geojson is None:
    boundary_geojson = load_bounds()
  nyc_boundary_json = json.dumps(boundary_geojson)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Debug Map - {config.get("name", "Generation")}</title>
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

    .info-panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      color: #fff;
      min-width: 180px;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .info-panel h3 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 8px;
    }}

    .coords {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 28px;
      font-weight: 600;
      color: #fff;
      display: flex;
      gap: 8px;
      align-items: baseline;
    }}

    .coords .label {{
      font-size: 14px;
      color: rgba(255, 255, 255, 0.4);
    }}

    .coords .x {{
      color: #ff6b6b;
    }}

    .coords .y {{
      color: #4ecdc4;
    }}

    .coords .separator {{
      color: rgba(255, 255, 255, 0.3);
      font-size: 20px;
    }}

    .no-hover {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: rgba(255, 255, 255, 0.3);
      font-style: italic;
    }}

    .stats-panel {{
      position: fixed;
      bottom: 16px;
      left: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .stats-panel .title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}

    .stats-panel .count {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: rgba(255, 255, 255, 0.6);
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

    .legend-swatch.quadrant {{
      background: rgba(255, 107, 107, 0.35);
      border: 2px solid rgba(255, 107, 107, 0.7);
    }}

    .legend-swatch.water {{
      background: rgba(59, 130, 246, 0.35);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.protected {{
      background: rgba(220, 38, 38, 0.25);
      border: 2px dashed rgba(220, 38, 38, 0.7);
    }}

    .legend-swatch.seed {{
      background: #10b981;
      border: 2px solid #fff;
      border-radius: 50%;
    }}

    .legend-swatch.nyc-boundary {{
      background: rgba(59, 130, 246, 0.15);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.pending {{
      background: rgba(168, 85, 247, 0.35);
      border: 2px solid rgba(168, 85, 247, 0.7);
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="info-panel">
    <h3>Quadrant</h3>
    <div id="coords-display" class="no-hover">Hover over a tile</div>
  </div>

  <div class="stats-panel">
    <div class="title">{config.get("name", "Debug Map")}</div>
    <div class="count">{len(quadrant_polygons)} quadrants generated</div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch nyc-boundary"></div>
      <span>NYC Boundary</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch quadrant"></div>
      <span>Generated Quadrant</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch water"></div>
      <span>Water Tile</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch protected"></div>
      <span>Protected (Not Water)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch pending"></div>
      <span>Pending Generation</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch seed"></div>
      <span>Seed Point</span>
    </div>
  </div>

  <script>
    const quadrants = {quadrants_json};
    const pendingQuadrants = {pending_json};
    const nycBoundary = {nyc_boundary_json};
    const seedLat = {seed_lat};
    const seedLng = {seed_lng};
    const centerLat = {center_lat};
    const centerLng = {center_lng};

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView([centerLat, centerLng], 15);

    // Add dark tile layer
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }}).addTo(map);

    // Add NYC boundary layer (underneath quadrants)
    const nycBoundaryStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.1,
      color: '#3b82f6',
      weight: 2,
      opacity: 0.6
    }};

    L.geoJSON(nycBoundary, {{
      style: nycBoundaryStyle
    }}).addTo(map);

    const coordsDisplay = document.getElementById('coords-display');

    // Style for normal state (red for land)
    const normalStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.25,
      color: '#ff6b6b',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for hover state (land)
    const hoverStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for water tiles (blue)
    const waterStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.35,
      color: '#3b82f6',
      weight: 1.5,
      opacity: 0.7
    }};

    // Style for water hover state
    const waterHoverStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.6,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for protected (explicit not water) tiles
    const protectedStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.2,
      color: '#dc2626',
      weight: 1.5,
      opacity: 0.6,
      dashArray: '5, 5'
    }};

    // Style for protected hover state
    const protectedHoverStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.4,
      color: '#fff',
      weight: 2,
      opacity: 1,
      dashArray: '5, 5'
    }};

    // Style for pending quadrants (purple)
    const pendingStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.25,
      color: '#a855f7',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for pending hover state
    const pendingHoverStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Helper to get style based on water_status
    function getStyle(waterStatus) {{
      if (waterStatus === 1) return waterStyle;
      if (waterStatus === -1) return protectedStyle;
      return normalStyle;
    }}

    function getHoverStyle(waterStatus) {{
      if (waterStatus === 1) return waterHoverStyle;
      if (waterStatus === -1) return protectedHoverStyle;
      return hoverStyle;
    }}

    // Add quadrant polygons
    quadrants.forEach(q => {{
      const style = getStyle(q.water_status);
      const hStyle = getHoverStyle(q.water_status);
      const polygon = L.polygon(q.corners, style).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(hStyle);
        let statusLabel = '';
        if (q.water_status === 1) {{
          statusLabel = ' <span style="color: #3b82f6;">💧</span>';
        }} else if (q.water_status === -1) {{
          statusLabel = ' <span style="color: #dc2626;">🛡️</span>';
        }}
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            ${{statusLabel}}
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(style);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add pending quadrant polygons (purple)
    pendingQuadrants.forEach(q => {{
      const polygon = L.polygon(q.corners, pendingStyle).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(pendingHoverStyle);
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            <span style="color: #a855f7;">⏳</span>
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(pendingStyle);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add seed point marker
    const seedIcon = L.divIcon({{
      className: 'seed-marker',
      html: `<div style="
        width: 14px;
        height: 14px;
        background: #10b981;
        border: 3px solid #fff;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      "></div>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    }});

    L.marker([seedLat, seedLng], {{ icon: seedIcon }})
      .addTo(map)
      .bindPopup('Seed Point');

    // Fit map to quadrant bounds if we have quadrants
    const allQuadrantCorners = [...quadrants, ...pendingQuadrants];
    if (allQuadrantCorners.length > 0) {{
      const allLatLngs = allQuadrantCorners.flatMap(q => q.corners);
      const bounds = L.latLngBounds(allLatLngs);
      map.fitBounds(bounds.pad(0.1));
    }}
  </script>
</body>
</html>
"""
  return html


def create_debug_map(
  generation_dir: Path,
  output_path: Path | None = None,
  bounds_path: Path | None = None,
) -> Path:
  """
  Create an interactive debug map showing generated quadrants overlaid on NYC.

  Args:
    generation_dir: Path to the generation directory
    output_path: Optional output path (default: generation_dir/debug_map.html)
    bounds_path: Optional path to custom bounds GeoJSON file (default: NYC)

  Returns:
    Path to the generated HTML file
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Load boundary GeoJSON
  boundary_geojson = load_bounds(bounds_path)
  bounds_name = bounds_path.name if bounds_path else "NYC (default)"
  print(f"📍 Using boundary: {bounds_name}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"📂 Loaded generation config: {config.get('name', 'unnamed')}")
    print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
    print(
      f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
      f"elevation={config['camera_elevation_degrees']}°"
    )

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"\n📊 Found {len(generated)} generated quadrants")

    if not generated:
      print("⚠️  No generated quadrants found. Creating empty map.")

    # Get pending quadrants from the generation queue
    pending = get_pending_quadrants(conn)
    # Filter out any pending quadrants that are already generated
    generated_set = {(qx, qy) for qx, qy, _ in generated}
    pending = [(qx, qy) for qx, qy in pending if (qx, qy) not in generated_set]
    if pending:
      print(f"   ⏳ {len(pending)} pending quadrants in queue")

    # Calculate corners for all generated quadrants
    print("\n🔲 Calculating quadrant geographic footprints...")
    quadrant_polygons = []
    all_lats = []
    all_lngs = []
    water_count = 0
    protected_count = 0

    for qx, qy, water_status in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners, water_status))

      if water_status == 1:
        water_count += 1
      elif water_status == -1:
        protected_count += 1

      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate corners for pending quadrants
    pending_polygons = []
    for qx, qy in pending:
      corners = calculate_quadrant_corners(config, qx, qy)
      pending_polygons.append((qx, qy, corners))
      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    if water_count > 0:
      print(f"   💧 {water_count} water tiles")
    if protected_count > 0:
      print(f"   🛡️  {protected_count} protected tiles")

    # Calculate center for map
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]

    if all_lats and all_lngs:
      center_lat = (min(all_lats) + max(all_lats)) / 2
      center_lng = (min(all_lngs) + max(all_lngs)) / 2
    else:
      center_lat = seed_lat
      center_lng = seed_lng

    print("\n🗺️  Map center:")
    print(f"   Lat: {center_lat:.6f}")
    print(f"   Lng: {center_lng:.6f}")

    # Generate HTML
    print("\n🎨 Generating interactive map...")
    html_content = generate_html(
      config=config,
      quadrant_polygons=quadrant_polygons,
      pending_polygons=pending_polygons,
      seed_lat=seed_lat,
      seed_lng=seed_lng,
      center_lat=center_lat,
      center_lng=center_lng,
    )

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.html"

    # Save HTML file
    print(f"\n💾 Saving to {output_path}")
    output_path.write_text(html_content)

    print(f"\n✅ Debug map created: {output_path}")
    print(f"   Open in browser: file://{output_path.resolve()}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create an interactive debug map showing generated quadrants."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: generation_dir/debug_map.html)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None

  print(f"\n{'=' * 60}")
  print("🗺️  Debug Map Generator (Interactive)")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")

  try:
    create_debug_map(generation_dir, output_path)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

```python
"""
Debug Map - Visualize generated quadrants on a real map of NYC.

This script creates an interactive web visualization showing which quadrants
have been generated by overlaying polygons on a real map using Leaflet.js.
Hovering over quadrants shows their x,y coordinates.

Usage:
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir] --bounds path/to/bounds.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --bounds: Path to a custom bounds GeoJSON file (default: NYC boundary)

Output:
  Creates debug_map.html in the generation directory
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.shared import calculate_offset
# fmt: on


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_generated_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int, int]]:
  """
  Get all quadrants that have been generated (have generation data).

  Returns list of (quadrant_x, quadrant_y, water_status) tuples.
  water_status: -1 = explicit not water, 0 = unset, 1 = water
  """
  cursor = conn.cursor()

  # Check if is_water column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  has_water_column = "is_water" in columns

  if has_water_column:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y, COALESCE(is_water, 0)
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
  else:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], 0) for row in cursor.fetchall()]


def get_pending_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int]]:
  """
  Get all unique quadrants from pending queue items.

  Returns list of (quadrant_x, quadrant_y) tuples.
  """
  cursor = conn.cursor()

  # Check if generation_queue table exists
  cursor.execute("""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='generation_queue'
  """)
  if not cursor.fetchone():
    return []

  # Get all pending queue items and extract their quadrants
  cursor.execute("""
    SELECT quadrants FROM generation_queue
    WHERE status = 'pending'
  """)

  pending_set: set[tuple[int, int]] = set()
  for (quadrants_json,) in cursor.fetchall():
    quadrants = json.loads(quadrants_json)
    for qx, qy in quadrants:
      pending_set.add((qx, qy))

  return sorted(pending_set)


def calculate_quadrant_corners(
  config: dict, quadrant_x: int, quadrant_y: int
) -> list[tuple[float, float]]:
  """
  Calculate the 4 geographic corners of a quadrant.

  The isometric projection means the quadrant appears as a parallelogram
  in geographic space. Returns corners in order: TL, TR, BR, BL.

  Args:
    config: Generation config dictionary
    quadrant_x: X coordinate of the quadrant
    quadrant_y: Y coordinate of the quadrant

  Returns:
    List of 4 (lat, lng) tuples representing the corners
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant is half a tile in size
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  # Calculate the pixel offsets for the 4 corners of this quadrant
  # Quadrant (0,0) has its bottom-right corner at the seed position
  # So we need to calculate corners relative to the seed
  base_x_px = quadrant_x * quadrant_width_px
  base_y_px = -quadrant_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels (relative to the quadrant's anchor)
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL: left and up from anchor
    (0, quadrant_height_px),  # TR: up from anchor
    (0, 0),  # BR: at anchor
    (-quadrant_width_px, 0),  # BL: left from anchor
  ]

  corners = []
  for dx, dy in corner_offsets:
    shift_x_px = base_x_px + dx
    shift_y_px = base_y_px + dy

    lat, lng = calculate_offset(
      seed_lat,
      seed_lng,
      shift_x_px,
      shift_y_px,
      view_height_meters,
      height_px,
      azimuth,
      elevation,
    )
    corners.append((lat, lng))

  return corners


def generate_html(
  config: dict,
  quadrant_polygons: list[tuple[int, int, list[tuple[float, float]], int]],
  pending_polygons: list[tuple[int, int, list[tuple[float, float]]]],
  seed_lat: float,
  seed_lng: float,
  center_lat: float,
  center_lng: float,
  boundary_geojson: dict | None = None,
) -> str:
  """Generate the HTML content for the debug map."""

  # Convert quadrant data to JSON for JavaScript
  # water_status: -1 = explicit not water, 0 = unset, 1 = water
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

  # Convert pending quadrant data to JSON for JavaScript
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

  # Boundary GeoJSON for the map (default to NYC if not provided)
  if boundary_geojson is None:
    boundary_geojson = load_bounds()
  nyc_boundary_json = json.dumps(boundary_geojson)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Debug Map - {config.get("name", "Generation")}</title>
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

    .info-panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      color: #fff;
      min-width: 180px;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .info-panel h3 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 8px;
    }}

    .coords {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 28px;
      font-weight: 600;
      color: #fff;
      display: flex;
      gap: 8px;
      align-items: baseline;
    }}

    .coords .label {{
      font-size: 14px;
      color: rgba(255, 255, 255, 0.4);
    }}

    .coords .x {{
      color: #ff6b6b;
    }}

    .coords .y {{
      color: #4ecdc4;
    }}

    .coords .separator {{
      color: rgba(255, 255, 255, 0.3);
      font-size: 20px;
    }}

    .no-hover {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: rgba(255, 255, 255, 0.3);
      font-style: italic;
    }}

    .stats-panel {{
      position: fixed;
      bottom: 16px;
      left: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .stats-panel .title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}

    .stats-panel .count {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: rgba(255, 255, 255, 0.6);
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

    .legend-swatch.quadrant {{
      background: rgba(255, 107, 107, 0.35);
      border: 2px solid rgba(255, 107, 107, 0.7);
    }}

    .legend-swatch.water {{
      background: rgba(59, 130, 246, 0.35);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.protected {{
      background: rgba(220, 38, 38, 0.25);
      border: 2px dashed rgba(220, 38, 38, 0.7);
    }}

    .legend-swatch.seed {{
      background: #10b981;
      border: 2px solid #fff;
      border-radius: 50%;
    }}

    .legend-swatch.nyc-boundary {{
      background: rgba(59, 130, 246, 0.15);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.pending {{
      background: rgba(168, 85, 247, 0.35);
      border: 2px solid rgba(168, 85, 247, 0.7);
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="info-panel">
    <h3>Quadrant</h3>
    <div id="coords-display" class="no-hover">Hover over a tile</div>
  </div>

  <div class="stats-panel">
    <div class="title">{config.get("name", "Debug Map")}</div>
    <div class="count">{len(quadrant_polygons)} quadrants generated</div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch nyc-boundary"></div>
      <span>NYC Boundary</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch quadrant"></div>
      <span>Generated Quadrant</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch water"></div>
      <span>Water Tile</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch protected"></div>
      <span>Protected (Not Water)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch pending"></div>
      <span>Pending Generation</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch seed"></div>
      <span>Seed Point</span>
    </div>
  </div>

  <script>
    const quadrants = {quadrants_json};
    const pendingQuadrants = {pending_json};
    const nycBoundary = {nyc_boundary_json};
    const seedLat = {seed_lat};
    const seedLng = {seed_lng};
    const centerLat = {center_lat};
    const centerLng = {center_lng};

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView([centerLat, centerLng], 15);

    // Add dark tile layer
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }}).addTo(map);

    // Add NYC boundary layer (underneath quadrants)
    const nycBoundaryStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.1,
      color: '#3b82f6',
      weight: 2,
      opacity: 0.6
    }};

    L.geoJSON(nycBoundary, {{
      style: nycBoundaryStyle
    }}).addTo(map);

    const coordsDisplay = document.getElementById('coords-display');

    // Style for normal state (red for land)
    const normalStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.25,
      color: '#ff6b6b',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for hover state (land)
    const hoverStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for water tiles (blue)
    const waterStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.35,
      color: '#3b82f6',
      weight: 1.5,
      opacity: 0.7
    }};

    // Style for water hover state
    const waterHoverStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.6,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for protected (explicit not water) tiles
    const protectedStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.2,
      color: '#dc2626',
      weight: 1.5,
      opacity: 0.6,
      dashArray: '5, 5'
    }};

    // Style for protected hover state
    const protectedHoverStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.4,
      color: '#fff',
      weight: 2,
      opacity: 1,
      dashArray: '5, 5'
    }};

    // Style for pending quadrants (purple)
    const pendingStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.25,
      color: '#a855f7',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for pending hover state
    const pendingHoverStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Helper to get style based on water_status
    function getStyle(waterStatus) {{
      if (waterStatus === 1) return waterStyle;
      if (waterStatus === -1) return protectedStyle;
      return normalStyle;
    }}

    function getHoverStyle(waterStatus) {{
      if (waterStatus === 1) return waterHoverStyle;
      if (waterStatus === -1) return protectedHoverStyle;
      return hoverStyle;
    }}

    // Add quadrant polygons
    quadrants.forEach(q => {{
      const style = getStyle(q.water_status);
      const hStyle = getHoverStyle(q.water_status);
      const polygon = L.polygon(q.corners, style).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(hStyle);
        let statusLabel = '';
        if (q.water_status === 1) {{
          statusLabel = ' <span style="color: #3b82f6;">💧</span>';
        }} else if (q.water_status === -1) {{
          statusLabel = ' <span style="color: #dc2626;">🛡️</span>';
        }}
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            ${{statusLabel}}
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(style);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add pending quadrant polygons (purple)
    pendingQuadrants.forEach(q => {{
      const polygon = L.polygon(q.corners, pendingStyle).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(pendingHoverStyle);
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            <span style="color: #a855f7;">⏳</span>
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(pendingStyle);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add seed point marker
    const seedIcon = L.divIcon({{
      className: 'seed-marker',
      html: `<div style="
        width: 14px;
        height: 14px;
        background: #10b981;
        border: 3px solid #fff;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      "></div>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    }});

    L.marker([seedLat, seedLng], {{ icon: seedIcon }})
      .addTo(map)
      .bindPopup('Seed Point');

    // Fit map to quadrant bounds if we have quadrants
    const allQuadrantCorners = [...quadrants, ...pendingQuadrants];
    if (allQuadrantCorners.length > 0) {{
      const allLatLngs = allQuadrantCorners.flatMap(q => q.corners);
      const bounds = L.latLngBounds(allLatLngs);
      map.fitBounds(bounds.pad(0.1));
    }}
  </script>
</body>
</html>
"""
  return html


def create_debug_map(
  generation_dir: Path,
  output_path: Path | None = None,
  bounds_path: Path | None = None,
) -> Path:
  """
  Create an interactive debug map showing generated quadrants overlaid on NYC.

  Args:
    generation_dir: Path to the generation directory
    output_path: Optional output path (default: generation_dir/debug_map.html)
    bounds_path: Optional path to custom bounds GeoJSON file (default: NYC)

  Returns:
    Path to the generated HTML file
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Load boundary GeoJSON
  boundary_geojson = load_bounds(bounds_path)
  bounds_name = bounds_path.name if bounds_path else "NYC (default)"
  print(f"📍 Using boundary: {bounds_name}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"📂 Loaded generation config: {config.get('name', 'unnamed')}")
    print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
    print(
      f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
      f"elevation={config['camera_elevation_degrees']}°"
    )

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"\n📊 Found {len(generated)} generated quadrants")

    if not generated:
      print("⚠️  No generated quadrants found. Creating empty map.")

    # Get pending quadrants from the generation queue
    pending = get_pending_quadrants(conn)
    # Filter out any pending quadrants that are already generated
    generated_set = {(qx, qy) for qx, qy, _ in generated}
    pending = [(qx, qy) for qx, qy in pending if (qx, qy) not in generated_set]
    if pending:
      print(f"   ⏳ {len(pending)} pending quadrants in queue")

    # Calculate corners for all generated quadrants
    print("\n🔲 Calculating quadrant geographic footprints...")
    quadrant_polygons = []
    all_lats = []
    all_lngs = []
    water_count = 0
    protected_count = 0

    for qx, qy, water_status in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners, water_status))

      if water_status == 1:
        water_count += 1
      elif water_status == -1:
        protected_count += 1

      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate corners for pending quadrants
    pending_polygons = []
    for qx, qy in pending:
      corners = calculate_quadrant_corners(config, qx, qy)
      pending_polygons.append((qx, qy, corners))
      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    if water_count > 0:
      print(f"   💧 {water_count} water tiles")
    if protected_count > 0:
      print(f"   🛡️  {protected_count} protected tiles")

    # Calculate center for map
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]

    if all_lats and all_lngs:
      center_lat = (min(all_lats) + max(all_lats)) / 2
      center_lng = (min(all_lngs) + max(all_lngs)) / 2
    else:
      center_lat = seed_lat
      center_lng = seed_lng

    print("\n🗺️  Map center:")
    print(f"   Lat: {center_lat:.6f}")
    print(f"   Lng: {center_lng:.6f}")

    # Generate HTML
    print("\n🎨 Generating interactive map...")
    html_content = generate_html(
      config=config,
      quadrant_polygons=quadrant_polygons,
      pending_polygons=pending_polygons,
      seed_lat=seed_lat,
      seed_lng=seed_lng,
      center_lat=center_lat,
      center_lng=center_lng,
      boundary_geojson=boundary_geojson,
    )

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.html"

    # Save HTML file
    print(f"\n💾 Saving to {output_path}")
    output_path.write_text(html_content)

    print(f"\n✅ Debug map created: {output_path}")
    print(f"   Open in browser: file://{output_path.resolve()}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create an interactive debug map showing generated quadrants."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: generation_dir/debug_map.html)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None

  print(f"\n{'=' * 60}")
  print("🗺️  Debug Map Generator (Interactive)")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")

  try:
    create_debug_map(generation_dir, output_path)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

```python
"""
Debug Map - Visualize generated quadrants on a real map of NYC.

This script creates an interactive web visualization showing which quadrants
have been generated by overlaying polygons on a real map using Leaflet.js.
Hovering over quadrants shows their x,y coordinates.

Usage:
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/debug_map.py [generation_dir] --bounds path/to/bounds.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --bounds: Path to a custom bounds GeoJSON file (default: NYC boundary)

Output:
  Creates debug_map.html in the generation directory
"""

import argparse
import json
import sqlite3
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.shared import calculate_offset
# fmt: on


def get_generation_config(conn: sqlite3.Connection) -> dict:
  """Get the generation config from the metadata table."""
  cursor = conn.cursor()
  cursor.execute("SELECT value FROM metadata WHERE key = 'generation_config'")
  row = cursor.fetchone()
  if not row:
    raise ValueError("generation_config not found in metadata")
  return json.loads(row[0])


def get_generated_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int, int]]:
  """
  Get all quadrants that have been generated (have generation data).

  Returns list of (quadrant_x, quadrant_y, water_status) tuples.
  water_status: -1 = explicit not water, 0 = unset, 1 = water
  """
  cursor = conn.cursor()

  # Check if is_water column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  has_water_column = "is_water" in columns

  if has_water_column:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y, COALESCE(is_water, 0)
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
  else:
    cursor.execute("""
      SELECT quadrant_x, quadrant_y
      FROM quadrants
      WHERE generation IS NOT NULL
      ORDER BY quadrant_x, quadrant_y
    """)
    return [(row[0], row[1], 0) for row in cursor.fetchall()]


def get_pending_quadrants(
  conn: sqlite3.Connection,
) -> list[tuple[int, int]]:
  """
  Get all unique quadrants from pending queue items.

  Returns list of (quadrant_x, quadrant_y) tuples.
  """
  cursor = conn.cursor()

  # Check if generation_queue table exists
  cursor.execute("""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='generation_queue'
  """)
  if not cursor.fetchone():
    return []

  # Get all pending queue items and extract their quadrants
  cursor.execute("""
    SELECT quadrants FROM generation_queue
    WHERE status = 'pending'
  """)

  pending_set: set[tuple[int, int]] = set()
  for (quadrants_json,) in cursor.fetchall():
    quadrants = json.loads(quadrants_json)
    for qx, qy in quadrants:
      pending_set.add((qx, qy))

  return sorted(pending_set)


def calculate_quadrant_corners(
  config: dict, quadrant_x: int, quadrant_y: int
) -> list[tuple[float, float]]:
  """
  Calculate the 4 geographic corners of a quadrant.

  The isometric projection means the quadrant appears as a parallelogram
  in geographic space. Returns corners in order: TL, TR, BR, BL.

  Args:
    config: Generation config dictionary
    quadrant_x: X coordinate of the quadrant
    quadrant_y: Y coordinate of the quadrant

  Returns:
    List of 4 (lat, lng) tuples representing the corners
  """
  seed_lat = config["seed"]["lat"]
  seed_lng = config["seed"]["lng"]
  width_px = config["width_px"]
  height_px = config["height_px"]
  view_height_meters = config["view_height_meters"]
  azimuth = config["camera_azimuth_degrees"]
  elevation = config["camera_elevation_degrees"]
  tile_step = config.get("tile_step", 0.5)

  # Each quadrant is half a tile in size
  quadrant_width_px = width_px * tile_step
  quadrant_height_px = height_px * tile_step

  # Calculate the pixel offsets for the 4 corners of this quadrant
  # Quadrant (0,0) has its bottom-right corner at the seed position
  # So we need to calculate corners relative to the seed
  base_x_px = quadrant_x * quadrant_width_px
  base_y_px = -quadrant_y * quadrant_height_px  # Negative because y increases down

  # Corner offsets in pixels (relative to the quadrant's anchor)
  # TL, TR, BR, BL - going clockwise
  corner_offsets = [
    (-quadrant_width_px, quadrant_height_px),  # TL: left and up from anchor
    (0, quadrant_height_px),  # TR: up from anchor
    (0, 0),  # BR: at anchor
    (-quadrant_width_px, 0),  # BL: left from anchor
  ]

  corners = []
  for dx, dy in corner_offsets:
    shift_x_px = base_x_px + dx
    shift_y_px = base_y_px + dy

    lat, lng = calculate_offset(
      seed_lat,
      seed_lng,
      shift_x_px,
      shift_y_px,
      view_height_meters,
      height_px,
      azimuth,
      elevation,
    )
    corners.append((lat, lng))

  return corners


def generate_html(
  config: dict,
  quadrant_polygons: list[tuple[int, int, list[tuple[float, float]], int]],
  pending_polygons: list[tuple[int, int, list[tuple[float, float]]]],
  seed_lat: float,
  seed_lng: float,
  center_lat: float,
  center_lng: float,
  boundary_geojson: dict | None = None,
) -> str:
  """Generate the HTML content for the debug map."""

  # Convert quadrant data to JSON for JavaScript
  # water_status: -1 = explicit not water, 0 = unset, 1 = water
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

  # Convert pending quadrant data to JSON for JavaScript
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

  # Boundary GeoJSON for the map (default to NYC if not provided)
  if boundary_geojson is None:
    boundary_geojson = load_bounds()
  nyc_boundary_json = json.dumps(boundary_geojson)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Debug Map - {config.get("name", "Generation")}</title>
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

    .info-panel {{
      position: fixed;
      top: 16px;
      right: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px 20px;
      color: #fff;
      min-width: 180px;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .info-panel h3 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: rgba(255, 255, 255, 0.5);
      margin-bottom: 8px;
    }}

    .coords {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 28px;
      font-weight: 600;
      color: #fff;
      display: flex;
      gap: 8px;
      align-items: baseline;
    }}

    .coords .label {{
      font-size: 14px;
      color: rgba(255, 255, 255, 0.4);
    }}

    .coords .x {{
      color: #ff6b6b;
    }}

    .coords .y {{
      color: #4ecdc4;
    }}

    .coords .separator {{
      color: rgba(255, 255, 255, 0.3);
      font-size: 20px;
    }}

    .no-hover {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: rgba(255, 255, 255, 0.3);
      font-style: italic;
    }}

    .stats-panel {{
      position: fixed;
      bottom: 16px;
      left: 16px;
      background: rgba(10, 10, 15, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      z-index: 1000;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    .stats-panel .title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}

    .stats-panel .count {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: rgba(255, 255, 255, 0.6);
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

    .legend-swatch.quadrant {{
      background: rgba(255, 107, 107, 0.35);
      border: 2px solid rgba(255, 107, 107, 0.7);
    }}

    .legend-swatch.water {{
      background: rgba(59, 130, 246, 0.35);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.protected {{
      background: rgba(220, 38, 38, 0.25);
      border: 2px dashed rgba(220, 38, 38, 0.7);
    }}

    .legend-swatch.seed {{
      background: #10b981;
      border: 2px solid #fff;
      border-radius: 50%;
    }}

    .legend-swatch.nyc-boundary {{
      background: rgba(59, 130, 246, 0.15);
      border: 2px solid rgba(59, 130, 246, 0.7);
    }}

    .legend-swatch.pending {{
      background: rgba(168, 85, 247, 0.35);
      border: 2px solid rgba(168, 85, 247, 0.7);
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <div class="info-panel">
    <h3>Quadrant</h3>
    <div id="coords-display" class="no-hover">Hover over a tile</div>
  </div>

  <div class="stats-panel">
    <div class="title">{config.get("name", "Debug Map")}</div>
    <div class="count">{len(quadrant_polygons)} quadrants generated</div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch nyc-boundary"></div>
      <span>NYC Boundary</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch quadrant"></div>
      <span>Generated Quadrant</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch water"></div>
      <span>Water Tile</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch protected"></div>
      <span>Protected (Not Water)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch pending"></div>
      <span>Pending Generation</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch seed"></div>
      <span>Seed Point</span>
    </div>
  </div>

  <script>
    const quadrants = {quadrants_json};
    const pendingQuadrants = {pending_json};
    const nycBoundary = {nyc_boundary_json};
    const seedLat = {seed_lat};
    const seedLng = {seed_lng};
    const centerLat = {center_lat};
    const centerLng = {center_lng};

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView([centerLat, centerLng], 15);

    // Add dark tile layer
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }}).addTo(map);

    // Add NYC boundary layer (underneath quadrants)
    const nycBoundaryStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.1,
      color: '#3b82f6',
      weight: 2,
      opacity: 0.6
    }};

    L.geoJSON(nycBoundary, {{
      style: nycBoundaryStyle
    }}).addTo(map);

    const coordsDisplay = document.getElementById('coords-display');

    // Style for normal state (red for land)
    const normalStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.25,
      color: '#ff6b6b',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for hover state (land)
    const hoverStyle = {{
      fillColor: '#ff6b6b',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for water tiles (blue)
    const waterStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.35,
      color: '#3b82f6',
      weight: 1.5,
      opacity: 0.7
    }};

    // Style for water hover state
    const waterHoverStyle = {{
      fillColor: '#3b82f6',
      fillOpacity: 0.6,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Style for protected (explicit not water) tiles
    const protectedStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.2,
      color: '#dc2626',
      weight: 1.5,
      opacity: 0.6,
      dashArray: '5, 5'
    }};

    // Style for protected hover state
    const protectedHoverStyle = {{
      fillColor: '#dc2626',
      fillOpacity: 0.4,
      color: '#fff',
      weight: 2,
      opacity: 1,
      dashArray: '5, 5'
    }};

    // Style for pending quadrants (purple)
    const pendingStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.25,
      color: '#a855f7',
      weight: 1.5,
      opacity: 0.6
    }};

    // Style for pending hover state
    const pendingHoverStyle = {{
      fillColor: '#a855f7',
      fillOpacity: 0.5,
      color: '#fff',
      weight: 2,
      opacity: 1
    }};

    // Helper to get style based on water_status
    function getStyle(waterStatus) {{
      if (waterStatus === 1) return waterStyle;
      if (waterStatus === -1) return protectedStyle;
      return normalStyle;
    }}

    function getHoverStyle(waterStatus) {{
      if (waterStatus === 1) return waterHoverStyle;
      if (waterStatus === -1) return protectedHoverStyle;
      return hoverStyle;
    }}

    // Add quadrant polygons
    quadrants.forEach(q => {{
      const style = getStyle(q.water_status);
      const hStyle = getHoverStyle(q.water_status);
      const polygon = L.polygon(q.corners, style).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(hStyle);
        let statusLabel = '';
        if (q.water_status === 1) {{
          statusLabel = ' <span style="color: #3b82f6;">💧</span>';
        }} else if (q.water_status === -1) {{
          statusLabel = ' <span style="color: #dc2626;">🛡️</span>';
        }}
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            ${{statusLabel}}
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(style);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add pending quadrant polygons (purple)
    pendingQuadrants.forEach(q => {{
      const polygon = L.polygon(q.corners, pendingStyle).addTo(map);

      polygon.on('mouseover', function(e) {{
        this.setStyle(pendingHoverStyle);
        coordsDisplay.innerHTML = `
          <div class="coords">
            <span class="label">x:</span><span class="x">${{q.x}}</span>
            <span class="separator">,</span>
            <span class="label">y:</span><span class="y">${{q.y}}</span>
            <span style="color: #a855f7;">⏳</span>
          </div>
        `;
      }});

      polygon.on('mouseout', function(e) {{
        this.setStyle(pendingStyle);
        coordsDisplay.innerHTML = '<span class="no-hover">Hover over a tile</span>';
      }});
    }});

    // Add seed point marker
    const seedIcon = L.divIcon({{
      className: 'seed-marker',
      html: `<div style="
        width: 14px;
        height: 14px;
        background: #10b981;
        border: 3px solid #fff;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      "></div>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    }});

    L.marker([seedLat, seedLng], {{ icon: seedIcon }})
      .addTo(map)
      .bindPopup('Seed Point');

    // Fit map to quadrant bounds if we have quadrants
    const allQuadrantCorners = [...quadrants, ...pendingQuadrants];
    if (allQuadrantCorners.length > 0) {{
      const allLatLngs = allQuadrantCorners.flatMap(q => q.corners);
      const bounds = L.latLngBounds(allLatLngs);
      map.fitBounds(bounds.pad(0.1));
    }}
  </script>
</body>
</html>
"""
  return html


def create_debug_map(
  generation_dir: Path,
  output_path: Path | None = None,
  bounds_path: Path | None = None,
) -> Path:
  """
  Create an interactive debug map showing generated quadrants overlaid on NYC.

  Args:
    generation_dir: Path to the generation directory
    output_path: Optional output path (default: generation_dir/debug_map.html)
    bounds_path: Optional path to custom bounds GeoJSON file (default: NYC)

  Returns:
    Path to the generated HTML file
  """
  db_path = generation_dir / "quadrants.db"
  if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

  # Load boundary GeoJSON
  boundary_geojson = load_bounds(bounds_path)
  bounds_name = bounds_path.name if bounds_path else "NYC (default)"
  print(f"📍 Using boundary: {bounds_name}")

  # Connect to database
  conn = sqlite3.connect(db_path)

  try:
    # Load config
    config = get_generation_config(conn)
    print(f"📂 Loaded generation config: {config.get('name', 'unnamed')}")
    print(f"   Seed: {config['seed']['lat']:.6f}, {config['seed']['lng']:.6f}")
    print(
      f"   Camera: azimuth={config['camera_azimuth_degrees']}°, "
      f"elevation={config['camera_elevation_degrees']}°"
    )

    # Get generated quadrants
    generated = get_generated_quadrants(conn)
    print(f"\n📊 Found {len(generated)} generated quadrants")

    if not generated:
      print("⚠️  No generated quadrants found. Creating empty map.")

    # Get pending quadrants from the generation queue
    pending = get_pending_quadrants(conn)
    # Filter out any pending quadrants that are already generated
    generated_set = {(qx, qy) for qx, qy, _ in generated}
    pending = [(qx, qy) for qx, qy in pending if (qx, qy) not in generated_set]
    if pending:
      print(f"   ⏳ {len(pending)} pending quadrants in queue")

    # Calculate corners for all generated quadrants
    print("\n🔲 Calculating quadrant geographic footprints...")
    quadrant_polygons = []
    all_lats = []
    all_lngs = []
    water_count = 0
    protected_count = 0

    for qx, qy, water_status in generated:
      corners = calculate_quadrant_corners(config, qx, qy)
      quadrant_polygons.append((qx, qy, corners, water_status))

      if water_status == 1:
        water_count += 1
      elif water_status == -1:
        protected_count += 1

      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    # Calculate corners for pending quadrants
    pending_polygons = []
    for qx, qy in pending:
      corners = calculate_quadrant_corners(config, qx, qy)
      pending_polygons.append((qx, qy, corners))
      for lat, lng in corners:
        all_lats.append(lat)
        all_lngs.append(lng)

    if water_count > 0:
      print(f"   💧 {water_count} water tiles")
    if protected_count > 0:
      print(f"   🛡️  {protected_count} protected tiles")

    # Calculate center for map
    seed_lat = config["seed"]["lat"]
    seed_lng = config["seed"]["lng"]

    if all_lats and all_lngs:
      center_lat = (min(all_lats) + max(all_lats)) / 2
      center_lng = (min(all_lngs) + max(all_lngs)) / 2
    else:
      center_lat = seed_lat
      center_lng = seed_lng

    print("\n🗺️  Map center:")
    print(f"   Lat: {center_lat:.6f}")
    print(f"   Lng: {center_lng:.6f}")

    # Generate HTML
    print("\n🎨 Generating interactive map...")
    html_content = generate_html(
      config=config,
      quadrant_polygons=quadrant_polygons,
      pending_polygons=pending_polygons,
      seed_lat=seed_lat,
      seed_lng=seed_lng,
      center_lat=center_lat,
      center_lng=center_lng,
      boundary_geojson=boundary_geojson,
    )

    # Set output path
    if output_path is None:
      output_path = generation_dir / "debug_map.html"

    # Save HTML file
    print(f"\n💾 Saving to {output_path}")
    output_path.write_text(html_content)

    print(f"\n✅ Debug map created: {output_path}")
    print(f"   Open in browser: file://{output_path.resolve()}")
    return output_path

  finally:
    conn.close()


def main():
  parser = argparse.ArgumentParser(
    description="Create an interactive debug map showing generated quadrants."
  )
  parser.add_argument(
    "generation_dir",
    type=Path,
    nargs="?",
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
  )
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    help="Output path for the debug map (default: generation_dir/debug_map.html)",
  )
  parser.add_argument(
    "--bounds",
    "-b",
    type=Path,
    help="Path to custom bounds GeoJSON file (default: NYC boundary)",
  )

  args = parser.parse_args()

  # Resolve paths
  generation_dir = args.generation_dir.resolve()

  if not generation_dir.exists():
    print(f"❌ Error: Directory not found: {generation_dir}")
    return 1

  output_path = args.output.resolve() if args.output else None
  bounds_path = args.bounds.resolve() if args.bounds else None

  print(f"\n{'=' * 60}")
  print("🗺️  Debug Map Generator (Interactive)")
  print(f"{'=' * 60}")
  print(f"   Generation dir: {generation_dir}")
  if bounds_path:
    print(f"   Bounds file: {bounds_path}")

  try:
    create_debug_map(generation_dir, output_path, bounds_path)
    return 0
  except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    return 1
  except ValueError as e:
    print(f"❌ Validation error: {e}")
    return 1
  except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise


if __name__ == "__main__":
  exit(main())

```

```python
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
  R          - Toggle render/generation mode
  G          - Generate selected quadrants
  S          - Toggle select tool
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.make_rectangle_plan import (
  Point,
  RectBounds,
  create_rectangle_plan,
  get_plan_summary,
  validate_plan,
)
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cancel_processing_items,
  cancel_queue_item_by_id,
  clear_completed_items,
  clear_pending_queue,
  get_all_processing_items,
  get_next_pending_item_for_available_model,
  get_pending_queue,
  get_queue_position_for_model,
  get_queue_status,
  get_queue_status_by_model,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
  reset_all_processing_items,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
  latlng_to_quadrant_coords,
)

from isometric_nyc.e2e_generation.bounds import load_bounds

# Global boundary GeoJSON - loaded at startup
BOUNDARY_GEOJSON: dict | None = None
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)


# =============================================================================
# Logging Configuration - Suppress noisy tile request logs
# =============================================================================
class TileRequestFilter(logging.Filter):
  """Filter out noisy tile and static file requests from logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    message = record.getMessage()
    # Filter out tile requests, static files, and api/status polling
    if "/tile/" in message:
      return False
    if "/static/" in message:
      return False
    if "/api/status" in message:
      return False
    return True


# Apply filter to werkzeug logger (Flask's HTTP request logger)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(TileRequestFilter())

# Generation lock - protects generation_state updates
generation_lock = threading.Lock()

# Per-model generation states
# Key is model_id (None for default), value is state dict
model_generation_states: dict[str | None, dict] = {}

# Legacy global generation_state for backwards compatibility with API
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
  "current_item_id": None,
  "model_id": None,
}

# Track which models are currently processing
busy_models: set[str | None] = set()
busy_models_lock = threading.Lock()

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Cancellation flag - set to True to cancel all generations
generation_cancelled = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def ensure_flagged_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the flagged column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "flagged" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'flagged' column to quadrants table")


def ensure_is_water_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


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


def get_quadrant_info(x: int, y: int, use_render: bool = False) -> dict:
  """
  Get info about a quadrant including whether it has data, is flagged, and water status.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed)
     1: Water tile
  """
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      water_status = row[2]
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": water_status == 1,  # True if water
        "is_explicit_not_water": water_status == -1,  # True if explicitly not water
        "water_status": water_status,  # Raw value: -1, 0, or 1
      }
    return {
      "has_data": False,
      "flagged": False,
      "is_water": False,
      "is_explicit_not_water": False,
      "water_status": 0,
    }
  finally:
    conn.close()


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

  # Check which tiles have data, flagged status, and water status
  tiles = {}
  flagged_tiles = {}
  water_tiles = {}
  explicit_not_water_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      water_tiles[(dx, dy)] = info["is_water"]
      explicit_not_water_tiles[(dx, dy)] = info["is_explicit_not_water"]

  # Get model configuration for the frontend
  models_config = []
  default_model_id = None
  if APP_CONFIG:
    models_config = [m.to_dict() for m in APP_CONFIG.models]
    default_model_id = APP_CONFIG.default_model_id

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    flagged_tiles=flagged_tiles,
    water_tiles=water_tiles,
    explicit_not_water_tiles=explicit_not_water_tiles,
    generation_dir=str(GENERATION_DIR),
    models_config=json.dumps(models_config),
    default_model_id=default_model_id,
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

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = shared_get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation.
    prompt: Optional additional prompt text for generation

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants,
    prompt=prompt,
  )


def render_quadrant_with_renderer(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
) -> bytes | None:
  """
  Render a quadrant using the global web renderer.

  Returns the PNG bytes of the rendered quadrant.
  """
  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    save_quadrant_render,
  )
  from isometric_nyc.e2e_generation.web_renderer import get_web_renderer

  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  renderer = get_web_renderer(port=WEB_SERVER_PORT)

  # Render the tile
  quadrant_images = renderer.render_quadrant(
    quadrant_x=x,
    quadrant_y=y,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    camera_azimuth_degrees=config["camera_azimuth_degrees"],
    camera_elevation_degrees=config["camera_elevation_degrees"],
    view_height_meters=config.get("view_height_meters", 200),
  )

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), png_bytes in quadrant_images.items():
    qx, qy = x + dx, y + dy
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


def process_queue_item_from_db(item_id: int) -> dict:
  """Process a single queue item from the database."""
  global generation_state

  conn = get_db_connection()
  try:
    # Get item details
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT item_type, quadrants, model_id, context_quadrants, prompt
      FROM generation_queue
      WHERE id = ?
      """,
      (item_id,),
    )
    row = cursor.fetchone()
    if not row:
      return {"success": False, "error": "Item not found"}

    item_type = QueueItemType(row[0])
    quadrants = json.loads(row[1])
    model_id = row[2]
    context_quadrants_raw = json.loads(row[3]) if row[3] else None
    prompt = row[4]

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Calculate context lazily if not explicitly provided
    # This ensures we use the most up-to-date context based on what's
    # actually generated at execution time (not queue time)
    if context_quadrants_raw:
      context_quadrants = [(q[0], q[1]) for q in context_quadrants_raw]
      print(
        f"   📋 Using explicit context from queue: {len(context_quadrants)} quadrant(s)"
      )
    else:
      # Calculate context lazily based on current generation state
      context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
      if context_quadrants:
        print(f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s)")
      else:
        print(
          "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
        )

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = (
      "starting" if item_type == QueueItemType.GENERATE else "rendering"
    )
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(
      f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}"
    )
    if model_id:
      print(f"   Model: {model_id}")
    if context_quadrants:
      print(f"   Context: {context_quadrants}")
    if prompt:
      print(f"   Prompt: {prompt}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt
        )

        if result["success"]:
          print(f"✅ Generation complete: {result['message']}")
          generation_state["status"] = "complete"
          generation_state["message"] = result["message"]
          mark_item_complete(conn, item_id, result["message"])
          return result

        # Generation failed
        if gen_attempt < max_generation_retries:
          print(
            f"⚠️  Generation failed (attempt {gen_attempt}/{max_generation_retries}): "
            f"{result['error']}"
          )
          print(f"⏳ Waiting {generation_retry_delay}s before retrying generation...")
          update_generation_state(
            "retrying",
            f"Generation failed, retrying (attempt {gen_attempt + 1}/{max_generation_retries})...",
          )
          time.sleep(generation_retry_delay)
        else:
          # All retries exhausted
          print(
            f"❌ Generation failed after {max_generation_retries} attempts: "
            f"{result['error']}"
          )
          generation_state["status"] = "error"
          generation_state["error"] = result["error"]
          mark_item_error(conn, item_id, result["error"])
          return result

      # Should not reach here, but just in case
      return result

    else:  # render
      update_generation_state("rendering", "Starting render...")

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant_with_renderer(conn, config, qx, qy)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      result_message = f"Rendered {rendered_count} quadrant(s)"
      update_generation_state("complete", result_message)
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")
      mark_item_complete(conn, item_id, result_message)

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    mark_item_error(conn, item_id, str(e))
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def process_model_item(item_id: int, model_id: str | None):
  """Process a single queue item for a specific model in its own thread."""
  global generation_state, generation_cancelled

  try:
    # Check cancellation before starting
    if generation_cancelled:
      print(f"⚠️  Item {item_id} cancelled before processing")
      return

    process_queue_item_from_db(item_id)

  except Exception as e:
    print(f"❌ Model worker error for {model_id}: {e}")
    traceback.print_exc()
  finally:
    # Remove model from busy set
    with busy_models_lock:
      busy_models.discard(model_id)

    # Update global state if this was the active model
    with generation_lock:
      if generation_state.get("model_id") == model_id:
        generation_state["is_generating"] = False
        generation_state["current_item_id"] = None

    # Remove from per-model states
    if model_id in model_generation_states:
      del model_generation_states[model_id]


def queue_worker():
  """Background worker that processes the generation queue from the database.

  This worker supports parallel processing of different models - each model
  can have one active generation at a time, but different models can run
  concurrently.
  """
  global generation_state, queue_worker_running, generation_cancelled

  print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
      # Check if we were cancelled
      if generation_cancelled:
        print("⚠️  Generation cancelled, resetting flags...")
        generation_cancelled = False
        with generation_lock:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None
        with busy_models_lock:
          busy_models.clear()
        model_generation_states.clear()
        time.sleep(0.5)
        continue

      conn = get_db_connection()

      # Get current busy models
      with busy_models_lock:
        current_busy = busy_models.copy()

      # Get next pending item for an available model
      item = get_next_pending_item_for_available_model(conn, current_busy)

      if item is None:
        # No items available (either queue empty or all models busy)
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      model_id = item.model_id
      conn.close()
      conn = None

      # Mark this model as busy
      with busy_models_lock:
        if model_id in busy_models:
          # Another thread grabbed this model, skip
          continue
        busy_models.add(model_id)

      # Update global state for display (use most recent)
      with generation_lock:
        generation_state["is_generating"] = True
        generation_state["model_id"] = model_id

      # Initialize per-model state
      model_generation_states[model_id] = {
        "is_generating": True,
        "item_id": item_id,
        "started_at": time.time(),
      }

      # Spawn a thread to process this model's item
      model_name = model_id or "default"
      worker_thread = threading.Thread(
        target=process_model_item,
        args=(item_id, model_id),
        name=f"model-worker-{model_name}",
        daemon=True,
      )
      worker_thread.start()

      print(f"🚀 Started worker for model '{model_name}' (item {item_id})")

      # Small delay before checking for more work
      time.sleep(0.2)

    except Exception as e:
      print(f"❌ Queue worker error: {e}")
      traceback.print_exc()
      time.sleep(1.0)
    finally:
      if conn:
        conn.close()

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue_db(
  quadrants: list[tuple[int, int]],
  item_type: str,
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
      context_quadrants,
      prompt,
    )

    # Get model-specific queue position
    model_position = get_queue_position_for_model(conn, queue_item.id, model_id)

    # Get total queue length for backwards compatibility
    pending = get_pending_queue(conn)
    total_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": model_position,  # Position within this model's queue
      "total_position": total_position,  # Overall queue position
      "model_id": model_id,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {model_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)
    model_status = get_queue_status_by_model(conn)

    # Get list of currently busy models
    with busy_models_lock:
      active_models = list(busy_models)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
      # Per-model queue info
      "queue_by_model": model_status["by_model"],
      "processing_models": model_status["processing_models"],
      # All currently processing models (for parallel processing)
      "active_models": active_models,
      "active_model_count": len(active_models),
      # All quadrants being processed across all models
      "all_processing_quadrants": model_status["all_processing_quadrants"],
    }

    # Set is_generating based on whether any models are active
    response["is_generating"] = len(active_models) > 0

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
  """
  API endpoint to clear all items from the generation queue,
  including cancelling any currently running generation.

  Returns:
    {
      "success": true,
      "cleared_count": N,
      "cancelled_count": M,
      "message": "Cleared N pending item(s), cancelled M in-progress item(s)"
    }
  """
  global generation_state, generation_cancelled

  print(f"\n{'=' * 60}")
  print("🗑️  Clear queue request received")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # Clear pending items
    cleared_count = clear_pending_queue(conn)

    # Cancel any in-progress items
    cancelled_count = cancel_processing_items(conn)

    # Set the cancellation flag so the worker knows to stop
    if cancelled_count > 0:
      generation_cancelled = True

    # Clear busy models
    with busy_models_lock:
      busy_models.clear()

    # Clear per-model states
    model_generation_states.clear()

    # Reset the generation state
    generation_state["is_generating"] = False
    generation_state["quadrants"] = []
    generation_state["status"] = "idle"
    generation_state["message"] = "Queue cleared"
    generation_state["error"] = None
    generation_state["current_item_id"] = None

    if cleared_count > 0 or cancelled_count > 0:
      print(
        f"✅ Cleared {cleared_count} pending, cancelled {cancelled_count} in-progress"
      )
    else:
      print("ℹ️  Queue was already empty")

    message_parts = []
    if cleared_count > 0:
      message_parts.append(f"Cleared {cleared_count} pending item(s)")
    if cancelled_count > 0:
      message_parts.append(f"cancelled {cancelled_count} in-progress item(s)")
    message = ", ".join(message_parts) if message_parts else "Queue was already empty"

    return jsonify(
      {
        "success": True,
        "cleared_count": cleared_count,
        "cancelled_count": cancelled_count,
        "message": message,
      }
    )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/queue/cancel/<int:item_id>", methods=["POST"])
def api_cancel_queue_item(item_id: int):
  """
  API endpoint to cancel a specific queue item by its ID.

  Returns:
    {
      "success": true,
      "cancelled": true,
      "item_id": N,
      "message": "Cancelled queue item N"
    }
  """
  global generation_cancelled

  print(f"\n{'=' * 60}")
  print(f"🗑️  Cancel queue item request received: item_id={item_id}")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # First check if this item was processing (not just pending)
    cursor = conn.cursor()
    cursor.execute(
      "SELECT status, model_id FROM generation_queue WHERE id = ?",
      (item_id,),
    )
    row = cursor.fetchone()
    was_processing = row and row[0] == "processing"
    cancelled_model_id = row[1] if row else None

    cancelled = cancel_queue_item_by_id(conn, item_id)

    if cancelled:
      print(f"✅ Cancelled queue item {item_id}")

      # Only set the global cancellation flag if this was a PROCESSING item
      # Pending items just get marked as cancelled in the database
      if was_processing:
        generation_cancelled = True
        print("   ⚠️  Item was processing, signaling cancellation")

        # Also remove this model from busy set so it can pick up new work
        if cancelled_model_id:
          with busy_models_lock:
            busy_models.discard(cancelled_model_id)

      return jsonify(
        {
          "success": True,
          "cancelled": True,
          "item_id": item_id,
          "message": f"Cancelled queue item {item_id}",
        }
      )
    else:
      print(f"ℹ️  Queue item {item_id} not found or already completed")
      return jsonify(
        {
          "success": True,
          "cancelled": False,
          "item_id": item_id,
          "message": f"Queue item {item_id} not found or already completed",
        }
      )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify(
    {
      "models": [m.to_dict() for m in APP_CONFIG.models],
      "default_model_id": APP_CONFIG.default_model_id,
    }
  )


@app.route("/api/nyc-boundary")
def api_nyc_boundary():
  """
  API endpoint to get the NYC boundary GeoJSON with coordinate transformation info.

  Returns the NYC borough boundaries along with the generation config needed
  to transform lat/lng coordinates to quadrant (x, y) coordinates.
  """
  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    # Pre-compute boundary points in quadrant coordinates for the frontend
    # This avoids complex math in JavaScript
    boundary_in_quadrants = {"type": "FeatureCollection", "features": []}

    for feature in NYC_BOUNDARY_GEOJSON["features"]:
      new_feature = {
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": {"type": feature["geometry"]["type"], "coordinates": []},
      }

      # Process each ring of the polygon
      for ring in feature["geometry"]["coordinates"]:
        new_ring = []
        for coord in ring:
          lng, lat = coord[0], coord[1]
          qx, qy = latlng_to_quadrant_coords(config, lat, lng)
          new_ring.append([qx, qy])
        new_feature["geometry"]["coordinates"].append(new_ring)

      boundary_in_quadrants["features"].append(new_feature)

    return jsonify(
      {
        "boundary": boundary_in_quadrants,
        "seed": config["seed"],
      }
    )
  finally:
    conn.close()


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/delete-render", methods=["POST"])
def api_delete_render():
  """API endpoint to delete render data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the render column (set to NULL) but keep the row
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted render data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/flag", methods=["POST"])
def api_flag():
  """API endpoint to flag/unflag selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get flag value (default to True/1 for flagging, False/0 for unflagging)
  flag_value = 1 if data.get("flag", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the flagged column exists
    ensure_flagged_column_exists(conn)

    flagged_count = 0
    for qx, qy in quadrants:
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET flagged = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (flag_value, qx, qy),
      )
      if cursor.rowcount > 0:
        flagged_count += 1

    conn.commit()

    action = "Flagged" if flag_value else "Unflagged"
    return jsonify(
      {
        "success": True,
        "message": f"{action} {flagged_count} quadrant{'s' if flagged_count != 1 else ''}",
        "count": flagged_count,
        "flagged": bool(flag_value),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/water", methods=["POST"])
def api_water():
  """
  API endpoint to mark/unmark selected quadrants as water tiles.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed by script)
     1: Water tile

  Request body:
    {
      "quadrants": [[x, y], ...],
      "is_water": true/false,  // true=water(1), false=not water(0)
      "explicit_not_water": true  // Optional: if true, sets to -1 (protected)
    }
  """
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Determine water value:
  # - explicit_not_water=true → -1 (protected from auto-detection)
  # - is_water=true → 1 (water)
  # - is_water=false → 0 (not water, can be auto-changed)
  if data.get("explicit_not_water", False):
    water_value = -1
    action = "Marked as explicitly NOT water (protected)"
  elif data.get("is_water", True):
    water_value = 1
    action = "Marked as water"
  else:
    water_value = 0
    action = "Unmarked as water"

  conn = get_db_connection()

  try:
    # Ensure the is_water column exists
    ensure_is_water_column_exists(conn)

    water_count = 0
    for qx, qy in quadrants:
      # First ensure the quadrant exists in the database
      cursor = conn.execute(
        "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (qx, qy),
      )
      if cursor.fetchone() is None:
        # Quadrant doesn't exist, skip it
        continue

      cursor = conn.execute(
        """
        UPDATE quadrants
        SET is_water = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (water_value, qx, qy),
      )
      if cursor.rowcount > 0:
        water_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "water_status": water_value,
        "is_water": water_value == 1,
        "is_explicit_not_water": water_value == -1,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  print(f"\n{'=' * 60}")
  print(f"🎨 Render request: {selected_quadrants}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "render")
  return jsonify(result), 202  # 202 Accepted


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """
  API endpoint to generate tiles for selected quadrants.

  Request body:
    {
      "quadrants": [[x, y], ...] or [{"x": x, "y": y}, ...],
      "model_id": "optional-model-id",
      "context": [[x, y], ...] or [{"x": x, "y": y}, ...]  // Optional context quadrants
    }

  The context quadrants are used to provide surrounding pixel art context for
  the generation. If a context quadrant has an existing generation, that will
  be used; otherwise the render content will be used.
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Get optional model_id from request
  model_id = data.get("model_id")

  # Convert quadrants to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Parse optional context quadrants
  context_quadrants = None
  context_raw = data.get("context")
  if context_raw:
    if not isinstance(context_raw, list):
      return jsonify(
        {
          "success": False,
          "error": "context must be a list of quadrant coordinates",
        }
      ), 400

    context_quadrants = []
    for q in context_raw:
      if isinstance(q, list) and len(q) == 2:
        context_quadrants.append((int(q[0]), int(q[1])))
      elif isinstance(q, dict) and "x" in q and "y" in q:
        context_quadrants.append((int(q["x"]), int(q["y"])))
      else:
        return jsonify(
          {
            "success": False,
            "error": f"Invalid context quadrant format: {q}",
          }
        ), 400

  # Parse optional prompt
  prompt = data.get("prompt")
  if prompt and not isinstance(prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "prompt must be a string",
      }
    ), 400

  # Clean up prompt (strip whitespace, None if empty)
  if prompt:
    prompt = prompt.strip()
    if not prompt:
      prompt = None

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt
  )
  return jsonify(result), 202  # 202 Accepted


# =============================================================================
# Rectangle Generation API
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def load_queued_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """
  Load all quadrants from pending and processing queue items.

  These quadrants are scheduled for generation and should be considered
  when planning new rectangles to avoid seam issues.
  """
  queued: set[Point] = set()

  # Get pending items
  pending_items = get_pending_queue(conn)
  for item in pending_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  # Get processing items
  processing_items = get_all_processing_items(conn)
  for item in processing_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  return queued


@app.route("/api/generate-rectangle", methods=["POST"])
def api_generate_rectangle():
  """
  API endpoint to generate all quadrants within a rectangle.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "model_id": "optional-model-id"    // Optional model ID
    }

  Returns:
    {
      "success": true,
      "plan_summary": {...},
      "queued_count": N,
      "message": "Queued N generation steps"
    }
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl = Point(int(tl_raw[0]), int(tl_raw[1]))
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl = Point(int(tl_raw["x"]), int(tl_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br = Point(int(br_raw[0]), int(br_raw[1]))
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br = Point(int(br_raw["x"]), int(br_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  # Get optional model_id
  model_id = data.get("model_id")

  print(f"\n{'=' * 60}")
  print(f"📐 Rectangle generation request: ({tl.x},{tl.y}) to ({br.x},{br.y})")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Load existing generated quadrants and pending/processing quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)
    queued = load_queued_quadrants(conn)

    if queued:
      print(f"   Considering {len(queued)} queued/processing quadrant(s) for seam avoidance")

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated, queued)

    # Validate the plan
    is_valid, errors = validate_plan(plan)
    if not is_valid:
      print(f"❌ Invalid plan generated: {errors}")
      return jsonify(
        {
          "success": False,
          "error": f"Internal error: invalid plan generated - {errors}",
        }
      ), 500

    # Get plan summary for response
    summary = get_plan_summary(plan)

    if len(plan.steps) == 0:
      print("ℹ️  No quadrants to generate (all already generated)")
      return jsonify(
        {
          "success": True,
          "plan_summary": summary,
          "queued_count": 0,
          "message": "No quadrants to generate - all already generated",
        }
      )

    # Queue all generation steps
    queued_count = 0
    for step in plan.steps:
      quadrants = [(q.x, q.y) for q in step.quadrants]
      add_to_queue(conn, QueueItemType.GENERATE, quadrants, model_id)
      queued_count += 1

    # Ensure queue worker is running
    start_queue_worker()

    print(f"✅ Queued {queued_count} generation step(s)")
    print(f"   Steps by type: {summary['steps_by_type']}")

    return jsonify(
      {
        "success": True,
        "plan_summary": summary,
        "queued_count": queued_count,
        "message": f"Queued {queued_count} generation step(s) for {summary['total_quadrants']} quadrant(s)",
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG

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
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to app_config.json (default: looks in the e2e_generation directory)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Load app configuration
  APP_CONFIG = load_app_config(args.config)
  print(f"📦 Loaded {len(APP_CONFIG.models)} model(s) from configuration")
  for model in APP_CONFIG.models:
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Reset any items that were mid-processing when server shut down
    # These will be retried automatically
    reset_count = reset_all_processing_items(conn)
    if reset_count > 0:
      print(f"🔄 Reset {reset_count} interrupted generation(s) - will be retried")
    # Clean up old completed items
    deleted_count = clear_completed_items(conn)
    if deleted_count > 0:
      print(f"🧹 Cleaned up {deleted_count} old completed queue item(s)")
  finally:
    conn.close()

  # Start the queue worker
  start_queue_worker()

  # Start the global web renderer
  try:
    start_global_renderer(port=WEB_SERVER_PORT)
  except Exception as e:
    print(f"⚠️  Failed to start web renderer: {e}")
    print("   Rendering will start on demand")

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web renderer
    print("🛑 Stopping web renderer...")
    stop_global_renderer()

  return 0


if __name__ == "__main__":
  exit(main())

```

```python
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
  R          - Toggle render/generation mode
  G          - Generate selected quadrants
  S          - Toggle select tool
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.make_rectangle_plan import (
  Point,
  RectBounds,
  create_rectangle_plan,
  get_plan_summary,
  validate_plan,
)
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cancel_processing_items,
  cancel_queue_item_by_id,
  clear_completed_items,
  clear_pending_queue,
  get_all_processing_items,
  get_next_pending_item_for_available_model,
  get_pending_queue,
  get_queue_position_for_model,
  get_queue_status,
  get_queue_status_by_model,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
  reset_all_processing_items,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
  latlng_to_quadrant_coords,
)

from isometric_nyc.e2e_generation.bounds import load_bounds

# Global boundary GeoJSON - loaded at startup
BOUNDARY_GEOJSON: dict | None = None
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)


# =============================================================================
# Logging Configuration - Suppress noisy tile request logs
# =============================================================================
class TileRequestFilter(logging.Filter):
  """Filter out noisy tile and static file requests from logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    message = record.getMessage()
    # Filter out tile requests, static files, and api/status polling
    if "/tile/" in message:
      return False
    if "/static/" in message:
      return False
    if "/api/status" in message:
      return False
    return True


# Apply filter to werkzeug logger (Flask's HTTP request logger)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(TileRequestFilter())

# Generation lock - protects generation_state updates
generation_lock = threading.Lock()

# Per-model generation states
# Key is model_id (None for default), value is state dict
model_generation_states: dict[str | None, dict] = {}

# Legacy global generation_state for backwards compatibility with API
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
  "current_item_id": None,
  "model_id": None,
}

# Track which models are currently processing
busy_models: set[str | None] = set()
busy_models_lock = threading.Lock()

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Cancellation flag - set to True to cancel all generations
generation_cancelled = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def ensure_flagged_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the flagged column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "flagged" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'flagged' column to quadrants table")


def ensure_is_water_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


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


def get_quadrant_info(x: int, y: int, use_render: bool = False) -> dict:
  """
  Get info about a quadrant including whether it has data, is flagged, and water status.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed)
     1: Water tile
  """
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      water_status = row[2]
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": water_status == 1,  # True if water
        "is_explicit_not_water": water_status == -1,  # True if explicitly not water
        "water_status": water_status,  # Raw value: -1, 0, or 1
      }
    return {
      "has_data": False,
      "flagged": False,
      "is_water": False,
      "is_explicit_not_water": False,
      "water_status": 0,
    }
  finally:
    conn.close()


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

  # Check which tiles have data, flagged status, and water status
  tiles = {}
  flagged_tiles = {}
  water_tiles = {}
  explicit_not_water_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      water_tiles[(dx, dy)] = info["is_water"]
      explicit_not_water_tiles[(dx, dy)] = info["is_explicit_not_water"]

  # Get model configuration for the frontend
  models_config = []
  default_model_id = None
  if APP_CONFIG:
    models_config = [m.to_dict() for m in APP_CONFIG.models]
    default_model_id = APP_CONFIG.default_model_id

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    flagged_tiles=flagged_tiles,
    water_tiles=water_tiles,
    explicit_not_water_tiles=explicit_not_water_tiles,
    generation_dir=str(GENERATION_DIR),
    models_config=json.dumps(models_config),
    default_model_id=default_model_id,
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

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = shared_get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation.
    prompt: Optional additional prompt text for generation

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants,
    prompt=prompt,
  )


def render_quadrant_with_renderer(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
) -> bytes | None:
  """
  Render a quadrant using the global web renderer.

  Returns the PNG bytes of the rendered quadrant.
  """
  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    save_quadrant_render,
  )
  from isometric_nyc.e2e_generation.web_renderer import get_web_renderer

  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  renderer = get_web_renderer(port=WEB_SERVER_PORT)

  # Render the tile
  quadrant_images = renderer.render_quadrant(
    quadrant_x=x,
    quadrant_y=y,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    camera_azimuth_degrees=config["camera_azimuth_degrees"],
    camera_elevation_degrees=config["camera_elevation_degrees"],
    view_height_meters=config.get("view_height_meters", 200),
  )

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), png_bytes in quadrant_images.items():
    qx, qy = x + dx, y + dy
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


def process_queue_item_from_db(item_id: int) -> dict:
  """Process a single queue item from the database."""
  global generation_state

  conn = get_db_connection()
  try:
    # Get item details
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT item_type, quadrants, model_id, context_quadrants, prompt
      FROM generation_queue
      WHERE id = ?
      """,
      (item_id,),
    )
    row = cursor.fetchone()
    if not row:
      return {"success": False, "error": "Item not found"}

    item_type = QueueItemType(row[0])
    quadrants = json.loads(row[1])
    model_id = row[2]
    context_quadrants_raw = json.loads(row[3]) if row[3] else None
    prompt = row[4]

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Calculate context lazily if not explicitly provided
    # This ensures we use the most up-to-date context based on what's
    # actually generated at execution time (not queue time)
    if context_quadrants_raw:
      context_quadrants = [(q[0], q[1]) for q in context_quadrants_raw]
      print(
        f"   📋 Using explicit context from queue: {len(context_quadrants)} quadrant(s)"
      )
    else:
      # Calculate context lazily based on current generation state
      context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
      if context_quadrants:
        print(f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s)")
      else:
        print(
          "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
        )

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = (
      "starting" if item_type == QueueItemType.GENERATE else "rendering"
    )
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(
      f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}"
    )
    if model_id:
      print(f"   Model: {model_id}")
    if context_quadrants:
      print(f"   Context: {context_quadrants}")
    if prompt:
      print(f"   Prompt: {prompt}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt
        )

        if result["success"]:
          print(f"✅ Generation complete: {result['message']}")
          generation_state["status"] = "complete"
          generation_state["message"] = result["message"]
          mark_item_complete(conn, item_id, result["message"])
          return result

        # Generation failed
        if gen_attempt < max_generation_retries:
          print(
            f"⚠️  Generation failed (attempt {gen_attempt}/{max_generation_retries}): "
            f"{result['error']}"
          )
          print(f"⏳ Waiting {generation_retry_delay}s before retrying generation...")
          update_generation_state(
            "retrying",
            f"Generation failed, retrying (attempt {gen_attempt + 1}/{max_generation_retries})...",
          )
          time.sleep(generation_retry_delay)
        else:
          # All retries exhausted
          print(
            f"❌ Generation failed after {max_generation_retries} attempts: "
            f"{result['error']}"
          )
          generation_state["status"] = "error"
          generation_state["error"] = result["error"]
          mark_item_error(conn, item_id, result["error"])
          return result

      # Should not reach here, but just in case
      return result

    else:  # render
      update_generation_state("rendering", "Starting render...")

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant_with_renderer(conn, config, qx, qy)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      result_message = f"Rendered {rendered_count} quadrant(s)"
      update_generation_state("complete", result_message)
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")
      mark_item_complete(conn, item_id, result_message)

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    mark_item_error(conn, item_id, str(e))
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def process_model_item(item_id: int, model_id: str | None):
  """Process a single queue item for a specific model in its own thread."""
  global generation_state, generation_cancelled

  try:
    # Check cancellation before starting
    if generation_cancelled:
      print(f"⚠️  Item {item_id} cancelled before processing")
      return

    process_queue_item_from_db(item_id)

  except Exception as e:
    print(f"❌ Model worker error for {model_id}: {e}")
    traceback.print_exc()
  finally:
    # Remove model from busy set
    with busy_models_lock:
      busy_models.discard(model_id)

    # Update global state if this was the active model
    with generation_lock:
      if generation_state.get("model_id") == model_id:
        generation_state["is_generating"] = False
        generation_state["current_item_id"] = None

    # Remove from per-model states
    if model_id in model_generation_states:
      del model_generation_states[model_id]


def queue_worker():
  """Background worker that processes the generation queue from the database.

  This worker supports parallel processing of different models - each model
  can have one active generation at a time, but different models can run
  concurrently.
  """
  global generation_state, queue_worker_running, generation_cancelled

  print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
      # Check if we were cancelled
      if generation_cancelled:
        print("⚠️  Generation cancelled, resetting flags...")
        generation_cancelled = False
        with generation_lock:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None
        with busy_models_lock:
          busy_models.clear()
        model_generation_states.clear()
        time.sleep(0.5)
        continue

      conn = get_db_connection()

      # Get current busy models
      with busy_models_lock:
        current_busy = busy_models.copy()

      # Get next pending item for an available model
      item = get_next_pending_item_for_available_model(conn, current_busy)

      if item is None:
        # No items available (either queue empty or all models busy)
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      model_id = item.model_id
      conn.close()
      conn = None

      # Mark this model as busy
      with busy_models_lock:
        if model_id in busy_models:
          # Another thread grabbed this model, skip
          continue
        busy_models.add(model_id)

      # Update global state for display (use most recent)
      with generation_lock:
        generation_state["is_generating"] = True
        generation_state["model_id"] = model_id

      # Initialize per-model state
      model_generation_states[model_id] = {
        "is_generating": True,
        "item_id": item_id,
        "started_at": time.time(),
      }

      # Spawn a thread to process this model's item
      model_name = model_id or "default"
      worker_thread = threading.Thread(
        target=process_model_item,
        args=(item_id, model_id),
        name=f"model-worker-{model_name}",
        daemon=True,
      )
      worker_thread.start()

      print(f"🚀 Started worker for model '{model_name}' (item {item_id})")

      # Small delay before checking for more work
      time.sleep(0.2)

    except Exception as e:
      print(f"❌ Queue worker error: {e}")
      traceback.print_exc()
      time.sleep(1.0)
    finally:
      if conn:
        conn.close()

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue_db(
  quadrants: list[tuple[int, int]],
  item_type: str,
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
      context_quadrants,
      prompt,
    )

    # Get model-specific queue position
    model_position = get_queue_position_for_model(conn, queue_item.id, model_id)

    # Get total queue length for backwards compatibility
    pending = get_pending_queue(conn)
    total_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": model_position,  # Position within this model's queue
      "total_position": total_position,  # Overall queue position
      "model_id": model_id,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {model_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)
    model_status = get_queue_status_by_model(conn)

    # Get list of currently busy models
    with busy_models_lock:
      active_models = list(busy_models)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
      # Per-model queue info
      "queue_by_model": model_status["by_model"],
      "processing_models": model_status["processing_models"],
      # All currently processing models (for parallel processing)
      "active_models": active_models,
      "active_model_count": len(active_models),
      # All quadrants being processed across all models
      "all_processing_quadrants": model_status["all_processing_quadrants"],
    }

    # Set is_generating based on whether any models are active
    response["is_generating"] = len(active_models) > 0

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
  """
  API endpoint to clear all items from the generation queue,
  including cancelling any currently running generation.

  Returns:
    {
      "success": true,
      "cleared_count": N,
      "cancelled_count": M,
      "message": "Cleared N pending item(s), cancelled M in-progress item(s)"
    }
  """
  global generation_state, generation_cancelled

  print(f"\n{'=' * 60}")
  print("🗑️  Clear queue request received")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # Clear pending items
    cleared_count = clear_pending_queue(conn)

    # Cancel any in-progress items
    cancelled_count = cancel_processing_items(conn)

    # Set the cancellation flag so the worker knows to stop
    if cancelled_count > 0:
      generation_cancelled = True

    # Clear busy models
    with busy_models_lock:
      busy_models.clear()

    # Clear per-model states
    model_generation_states.clear()

    # Reset the generation state
    generation_state["is_generating"] = False
    generation_state["quadrants"] = []
    generation_state["status"] = "idle"
    generation_state["message"] = "Queue cleared"
    generation_state["error"] = None
    generation_state["current_item_id"] = None

    if cleared_count > 0 or cancelled_count > 0:
      print(
        f"✅ Cleared {cleared_count} pending, cancelled {cancelled_count} in-progress"
      )
    else:
      print("ℹ️  Queue was already empty")

    message_parts = []
    if cleared_count > 0:
      message_parts.append(f"Cleared {cleared_count} pending item(s)")
    if cancelled_count > 0:
      message_parts.append(f"cancelled {cancelled_count} in-progress item(s)")
    message = ", ".join(message_parts) if message_parts else "Queue was already empty"

    return jsonify(
      {
        "success": True,
        "cleared_count": cleared_count,
        "cancelled_count": cancelled_count,
        "message": message,
      }
    )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/queue/cancel/<int:item_id>", methods=["POST"])
def api_cancel_queue_item(item_id: int):
  """
  API endpoint to cancel a specific queue item by its ID.

  Returns:
    {
      "success": true,
      "cancelled": true,
      "item_id": N,
      "message": "Cancelled queue item N"
    }
  """
  global generation_cancelled

  print(f"\n{'=' * 60}")
  print(f"🗑️  Cancel queue item request received: item_id={item_id}")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # First check if this item was processing (not just pending)
    cursor = conn.cursor()
    cursor.execute(
      "SELECT status, model_id FROM generation_queue WHERE id = ?",
      (item_id,),
    )
    row = cursor.fetchone()
    was_processing = row and row[0] == "processing"
    cancelled_model_id = row[1] if row else None

    cancelled = cancel_queue_item_by_id(conn, item_id)

    if cancelled:
      print(f"✅ Cancelled queue item {item_id}")

      # Only set the global cancellation flag if this was a PROCESSING item
      # Pending items just get marked as cancelled in the database
      if was_processing:
        generation_cancelled = True
        print("   ⚠️  Item was processing, signaling cancellation")

        # Also remove this model from busy set so it can pick up new work
        if cancelled_model_id:
          with busy_models_lock:
            busy_models.discard(cancelled_model_id)

      return jsonify(
        {
          "success": True,
          "cancelled": True,
          "item_id": item_id,
          "message": f"Cancelled queue item {item_id}",
        }
      )
    else:
      print(f"ℹ️  Queue item {item_id} not found or already completed")
      return jsonify(
        {
          "success": True,
          "cancelled": False,
          "item_id": item_id,
          "message": f"Queue item {item_id} not found or already completed",
        }
      )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify(
    {
      "models": [m.to_dict() for m in APP_CONFIG.models],
      "default_model_id": APP_CONFIG.default_model_id,
    }
  )


@app.route("/api/nyc-boundary")
def api_nyc_boundary():
  """
  API endpoint to get the NYC boundary GeoJSON with coordinate transformation info.

  Returns the NYC borough boundaries along with the generation config needed
  to transform lat/lng coordinates to quadrant (x, y) coordinates.
  """
  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    # Pre-compute boundary points in quadrant coordinates for the frontend
    # This avoids complex math in JavaScript
    boundary_in_quadrants = {"type": "FeatureCollection", "features": []}

    for feature in BOUNDARY_GEOJSON["features"]:
      new_feature = {
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": {"type": feature["geometry"]["type"], "coordinates": []},
      }

      # Process each ring of the polygon
      for ring in feature["geometry"]["coordinates"]:
        new_ring = []
        for coord in ring:
          lng, lat = coord[0], coord[1]
          qx, qy = latlng_to_quadrant_coords(config, lat, lng)
          new_ring.append([qx, qy])
        new_feature["geometry"]["coordinates"].append(new_ring)

      boundary_in_quadrants["features"].append(new_feature)

    return jsonify(
      {
        "boundary": boundary_in_quadrants,
        "seed": config["seed"],
      }
    )
  finally:
    conn.close()


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/delete-render", methods=["POST"])
def api_delete_render():
  """API endpoint to delete render data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the render column (set to NULL) but keep the row
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted render data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/flag", methods=["POST"])
def api_flag():
  """API endpoint to flag/unflag selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get flag value (default to True/1 for flagging, False/0 for unflagging)
  flag_value = 1 if data.get("flag", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the flagged column exists
    ensure_flagged_column_exists(conn)

    flagged_count = 0
    for qx, qy in quadrants:
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET flagged = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (flag_value, qx, qy),
      )
      if cursor.rowcount > 0:
        flagged_count += 1

    conn.commit()

    action = "Flagged" if flag_value else "Unflagged"
    return jsonify(
      {
        "success": True,
        "message": f"{action} {flagged_count} quadrant{'s' if flagged_count != 1 else ''}",
        "count": flagged_count,
        "flagged": bool(flag_value),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/water", methods=["POST"])
def api_water():
  """
  API endpoint to mark/unmark selected quadrants as water tiles.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed by script)
     1: Water tile

  Request body:
    {
      "quadrants": [[x, y], ...],
      "is_water": true/false,  // true=water(1), false=not water(0)
      "explicit_not_water": true  // Optional: if true, sets to -1 (protected)
    }
  """
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Determine water value:
  # - explicit_not_water=true → -1 (protected from auto-detection)
  # - is_water=true → 1 (water)
  # - is_water=false → 0 (not water, can be auto-changed)
  if data.get("explicit_not_water", False):
    water_value = -1
    action = "Marked as explicitly NOT water (protected)"
  elif data.get("is_water", True):
    water_value = 1
    action = "Marked as water"
  else:
    water_value = 0
    action = "Unmarked as water"

  conn = get_db_connection()

  try:
    # Ensure the is_water column exists
    ensure_is_water_column_exists(conn)

    water_count = 0
    for qx, qy in quadrants:
      # First ensure the quadrant exists in the database
      cursor = conn.execute(
        "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (qx, qy),
      )
      if cursor.fetchone() is None:
        # Quadrant doesn't exist, skip it
        continue

      cursor = conn.execute(
        """
        UPDATE quadrants
        SET is_water = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (water_value, qx, qy),
      )
      if cursor.rowcount > 0:
        water_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "water_status": water_value,
        "is_water": water_value == 1,
        "is_explicit_not_water": water_value == -1,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  print(f"\n{'=' * 60}")
  print(f"🎨 Render request: {selected_quadrants}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "render")
  return jsonify(result), 202  # 202 Accepted


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """
  API endpoint to generate tiles for selected quadrants.

  Request body:
    {
      "quadrants": [[x, y], ...] or [{"x": x, "y": y}, ...],
      "model_id": "optional-model-id",
      "context": [[x, y], ...] or [{"x": x, "y": y}, ...]  // Optional context quadrants
    }

  The context quadrants are used to provide surrounding pixel art context for
  the generation. If a context quadrant has an existing generation, that will
  be used; otherwise the render content will be used.
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Get optional model_id from request
  model_id = data.get("model_id")

  # Convert quadrants to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Parse optional context quadrants
  context_quadrants = None
  context_raw = data.get("context")
  if context_raw:
    if not isinstance(context_raw, list):
      return jsonify(
        {
          "success": False,
          "error": "context must be a list of quadrant coordinates",
        }
      ), 400

    context_quadrants = []
    for q in context_raw:
      if isinstance(q, list) and len(q) == 2:
        context_quadrants.append((int(q[0]), int(q[1])))
      elif isinstance(q, dict) and "x" in q and "y" in q:
        context_quadrants.append((int(q["x"]), int(q["y"])))
      else:
        return jsonify(
          {
            "success": False,
            "error": f"Invalid context quadrant format: {q}",
          }
        ), 400

  # Parse optional prompt
  prompt = data.get("prompt")
  if prompt and not isinstance(prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "prompt must be a string",
      }
    ), 400

  # Clean up prompt (strip whitespace, None if empty)
  if prompt:
    prompt = prompt.strip()
    if not prompt:
      prompt = None

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt
  )
  return jsonify(result), 202  # 202 Accepted


# =============================================================================
# Rectangle Generation API
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def load_queued_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """
  Load all quadrants from pending and processing queue items.

  These quadrants are scheduled for generation and should be considered
  when planning new rectangles to avoid seam issues.
  """
  queued: set[Point] = set()

  # Get pending items
  pending_items = get_pending_queue(conn)
  for item in pending_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  # Get processing items
  processing_items = get_all_processing_items(conn)
  for item in processing_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  return queued


@app.route("/api/generate-rectangle", methods=["POST"])
def api_generate_rectangle():
  """
  API endpoint to generate all quadrants within a rectangle.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "model_id": "optional-model-id"    // Optional model ID
    }

  Returns:
    {
      "success": true,
      "plan_summary": {...},
      "queued_count": N,
      "message": "Queued N generation steps"
    }
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl = Point(int(tl_raw[0]), int(tl_raw[1]))
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl = Point(int(tl_raw["x"]), int(tl_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br = Point(int(br_raw[0]), int(br_raw[1]))
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br = Point(int(br_raw["x"]), int(br_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  # Get optional model_id
  model_id = data.get("model_id")

  print(f"\n{'=' * 60}")
  print(f"📐 Rectangle generation request: ({tl.x},{tl.y}) to ({br.x},{br.y})")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Load existing generated quadrants and pending/processing quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)
    queued = load_queued_quadrants(conn)

    if queued:
      print(f"   Considering {len(queued)} queued/processing quadrant(s) for seam avoidance")

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated, queued)

    # Validate the plan
    is_valid, errors = validate_plan(plan)
    if not is_valid:
      print(f"❌ Invalid plan generated: {errors}")
      return jsonify(
        {
          "success": False,
          "error": f"Internal error: invalid plan generated - {errors}",
        }
      ), 500

    # Get plan summary for response
    summary = get_plan_summary(plan)

    if len(plan.steps) == 0:
      print("ℹ️  No quadrants to generate (all already generated)")
      return jsonify(
        {
          "success": True,
          "plan_summary": summary,
          "queued_count": 0,
          "message": "No quadrants to generate - all already generated",
        }
      )

    # Queue all generation steps
    queued_count = 0
    for step in plan.steps:
      quadrants = [(q.x, q.y) for q in step.quadrants]
      add_to_queue(conn, QueueItemType.GENERATE, quadrants, model_id)
      queued_count += 1

    # Ensure queue worker is running
    start_queue_worker()

    print(f"✅ Queued {queued_count} generation step(s)")
    print(f"   Steps by type: {summary['steps_by_type']}")

    return jsonify(
      {
        "success": True,
        "plan_summary": summary,
        "queued_count": queued_count,
        "message": f"Queued {queued_count} generation step(s) for {summary['total_quadrants']} quadrant(s)",
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG

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
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to app_config.json (default: looks in the e2e_generation directory)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Load app configuration
  APP_CONFIG = load_app_config(args.config)
  print(f"📦 Loaded {len(APP_CONFIG.models)} model(s) from configuration")
  for model in APP_CONFIG.models:
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Reset any items that were mid-processing when server shut down
    # These will be retried automatically
    reset_count = reset_all_processing_items(conn)
    if reset_count > 0:
      print(f"🔄 Reset {reset_count} interrupted generation(s) - will be retried")
    # Clean up old completed items
    deleted_count = clear_completed_items(conn)
    if deleted_count > 0:
      print(f"🧹 Cleaned up {deleted_count} old completed queue item(s)")
  finally:
    conn.close()

  # Start the queue worker
  start_queue_worker()

  # Start the global web renderer
  try:
    start_global_renderer(port=WEB_SERVER_PORT)
  except Exception as e:
    print(f"⚠️  Failed to start web renderer: {e}")
    print("   Rendering will start on demand")

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web renderer
    print("🛑 Stopping web renderer...")
    stop_global_renderer()

  return 0


if __name__ == "__main__":
  exit(main())

```

```python
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
  R          - Toggle render/generation mode
  G          - Generate selected quadrants
  S          - Toggle select tool
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.make_rectangle_plan import (
  Point,
  RectBounds,
  create_rectangle_plan,
  get_plan_summary,
  validate_plan,
)
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cancel_processing_items,
  cancel_queue_item_by_id,
  clear_completed_items,
  clear_pending_queue,
  get_all_processing_items,
  get_next_pending_item_for_available_model,
  get_pending_queue,
  get_queue_position_for_model,
  get_queue_status,
  get_queue_status_by_model,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
  reset_all_processing_items,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
  latlng_to_quadrant_coords,
)

from isometric_nyc.e2e_generation.bounds import load_bounds

# Global boundary GeoJSON - loaded at startup
BOUNDARY_GEOJSON: dict | None = None
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)


# =============================================================================
# Logging Configuration - Suppress noisy tile request logs
# =============================================================================
class TileRequestFilter(logging.Filter):
  """Filter out noisy tile and static file requests from logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    message = record.getMessage()
    # Filter out tile requests, static files, and api/status polling
    if "/tile/" in message:
      return False
    if "/static/" in message:
      return False
    if "/api/status" in message:
      return False
    return True


# Apply filter to werkzeug logger (Flask's HTTP request logger)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(TileRequestFilter())

# Generation lock - protects generation_state updates
generation_lock = threading.Lock()

# Per-model generation states
# Key is model_id (None for default), value is state dict
model_generation_states: dict[str | None, dict] = {}

# Legacy global generation_state for backwards compatibility with API
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
  "current_item_id": None,
  "model_id": None,
}

# Track which models are currently processing
busy_models: set[str | None] = set()
busy_models_lock = threading.Lock()

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Cancellation flag - set to True to cancel all generations
generation_cancelled = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def ensure_flagged_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the flagged column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "flagged" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'flagged' column to quadrants table")


def ensure_is_water_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


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


def get_quadrant_info(x: int, y: int, use_render: bool = False) -> dict:
  """
  Get info about a quadrant including whether it has data, is flagged, and water status.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed)
     1: Water tile
  """
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      water_status = row[2]
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": water_status == 1,  # True if water
        "is_explicit_not_water": water_status == -1,  # True if explicitly not water
        "water_status": water_status,  # Raw value: -1, 0, or 1
      }
    return {
      "has_data": False,
      "flagged": False,
      "is_water": False,
      "is_explicit_not_water": False,
      "water_status": 0,
    }
  finally:
    conn.close()


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

  # Check which tiles have data, flagged status, and water status
  tiles = {}
  flagged_tiles = {}
  water_tiles = {}
  explicit_not_water_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      water_tiles[(dx, dy)] = info["is_water"]
      explicit_not_water_tiles[(dx, dy)] = info["is_explicit_not_water"]

  # Get model configuration for the frontend
  models_config = []
  default_model_id = None
  if APP_CONFIG:
    models_config = [m.to_dict() for m in APP_CONFIG.models]
    default_model_id = APP_CONFIG.default_model_id

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    flagged_tiles=flagged_tiles,
    water_tiles=water_tiles,
    explicit_not_water_tiles=explicit_not_water_tiles,
    generation_dir=str(GENERATION_DIR),
    models_config=json.dumps(models_config),
    default_model_id=default_model_id,
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

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = shared_get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation.
    prompt: Optional additional prompt text for generation

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants,
    prompt=prompt,
  )


def render_quadrant_with_renderer(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
) -> bytes | None:
  """
  Render a quadrant using the global web renderer.

  Returns the PNG bytes of the rendered quadrant.
  """
  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    save_quadrant_render,
  )
  from isometric_nyc.e2e_generation.web_renderer import get_web_renderer

  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  renderer = get_web_renderer(port=WEB_SERVER_PORT)

  # Render the tile
  quadrant_images = renderer.render_quadrant(
    quadrant_x=x,
    quadrant_y=y,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    camera_azimuth_degrees=config["camera_azimuth_degrees"],
    camera_elevation_degrees=config["camera_elevation_degrees"],
    view_height_meters=config.get("view_height_meters", 200),
  )

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), png_bytes in quadrant_images.items():
    qx, qy = x + dx, y + dy
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


def process_queue_item_from_db(item_id: int) -> dict:
  """Process a single queue item from the database."""
  global generation_state

  conn = get_db_connection()
  try:
    # Get item details
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT item_type, quadrants, model_id, context_quadrants, prompt
      FROM generation_queue
      WHERE id = ?
      """,
      (item_id,),
    )
    row = cursor.fetchone()
    if not row:
      return {"success": False, "error": "Item not found"}

    item_type = QueueItemType(row[0])
    quadrants = json.loads(row[1])
    model_id = row[2]
    context_quadrants_raw = json.loads(row[3]) if row[3] else None
    prompt = row[4]

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Calculate context lazily if not explicitly provided
    # This ensures we use the most up-to-date context based on what's
    # actually generated at execution time (not queue time)
    if context_quadrants_raw:
      context_quadrants = [(q[0], q[1]) for q in context_quadrants_raw]
      print(
        f"   📋 Using explicit context from queue: {len(context_quadrants)} quadrant(s)"
      )
    else:
      # Calculate context lazily based on current generation state
      context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
      if context_quadrants:
        print(f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s)")
      else:
        print(
          "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
        )

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = (
      "starting" if item_type == QueueItemType.GENERATE else "rendering"
    )
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(
      f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}"
    )
    if model_id:
      print(f"   Model: {model_id}")
    if context_quadrants:
      print(f"   Context: {context_quadrants}")
    if prompt:
      print(f"   Prompt: {prompt}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt
        )

        if result["success"]:
          print(f"✅ Generation complete: {result['message']}")
          generation_state["status"] = "complete"
          generation_state["message"] = result["message"]
          mark_item_complete(conn, item_id, result["message"])
          return result

        # Generation failed
        if gen_attempt < max_generation_retries:
          print(
            f"⚠️  Generation failed (attempt {gen_attempt}/{max_generation_retries}): "
            f"{result['error']}"
          )
          print(f"⏳ Waiting {generation_retry_delay}s before retrying generation...")
          update_generation_state(
            "retrying",
            f"Generation failed, retrying (attempt {gen_attempt + 1}/{max_generation_retries})...",
          )
          time.sleep(generation_retry_delay)
        else:
          # All retries exhausted
          print(
            f"❌ Generation failed after {max_generation_retries} attempts: "
            f"{result['error']}"
          )
          generation_state["status"] = "error"
          generation_state["error"] = result["error"]
          mark_item_error(conn, item_id, result["error"])
          return result

      # Should not reach here, but just in case
      return result

    else:  # render
      update_generation_state("rendering", "Starting render...")

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant_with_renderer(conn, config, qx, qy)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      result_message = f"Rendered {rendered_count} quadrant(s)"
      update_generation_state("complete", result_message)
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")
      mark_item_complete(conn, item_id, result_message)

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    mark_item_error(conn, item_id, str(e))
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def process_model_item(item_id: int, model_id: str | None):
  """Process a single queue item for a specific model in its own thread."""
  global generation_state, generation_cancelled

  try:
    # Check cancellation before starting
    if generation_cancelled:
      print(f"⚠️  Item {item_id} cancelled before processing")
      return

    process_queue_item_from_db(item_id)

  except Exception as e:
    print(f"❌ Model worker error for {model_id}: {e}")
    traceback.print_exc()
  finally:
    # Remove model from busy set
    with busy_models_lock:
      busy_models.discard(model_id)

    # Update global state if this was the active model
    with generation_lock:
      if generation_state.get("model_id") == model_id:
        generation_state["is_generating"] = False
        generation_state["current_item_id"] = None

    # Remove from per-model states
    if model_id in model_generation_states:
      del model_generation_states[model_id]


def queue_worker():
  """Background worker that processes the generation queue from the database.

  This worker supports parallel processing of different models - each model
  can have one active generation at a time, but different models can run
  concurrently.
  """
  global generation_state, queue_worker_running, generation_cancelled

  print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
      # Check if we were cancelled
      if generation_cancelled:
        print("⚠️  Generation cancelled, resetting flags...")
        generation_cancelled = False
        with generation_lock:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None
        with busy_models_lock:
          busy_models.clear()
        model_generation_states.clear()
        time.sleep(0.5)
        continue

      conn = get_db_connection()

      # Get current busy models
      with busy_models_lock:
        current_busy = busy_models.copy()

      # Get next pending item for an available model
      item = get_next_pending_item_for_available_model(conn, current_busy)

      if item is None:
        # No items available (either queue empty or all models busy)
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      model_id = item.model_id
      conn.close()
      conn = None

      # Mark this model as busy
      with busy_models_lock:
        if model_id in busy_models:
          # Another thread grabbed this model, skip
          continue
        busy_models.add(model_id)

      # Update global state for display (use most recent)
      with generation_lock:
        generation_state["is_generating"] = True
        generation_state["model_id"] = model_id

      # Initialize per-model state
      model_generation_states[model_id] = {
        "is_generating": True,
        "item_id": item_id,
        "started_at": time.time(),
      }

      # Spawn a thread to process this model's item
      model_name = model_id or "default"
      worker_thread = threading.Thread(
        target=process_model_item,
        args=(item_id, model_id),
        name=f"model-worker-{model_name}",
        daemon=True,
      )
      worker_thread.start()

      print(f"🚀 Started worker for model '{model_name}' (item {item_id})")

      # Small delay before checking for more work
      time.sleep(0.2)

    except Exception as e:
      print(f"❌ Queue worker error: {e}")
      traceback.print_exc()
      time.sleep(1.0)
    finally:
      if conn:
        conn.close()

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue_db(
  quadrants: list[tuple[int, int]],
  item_type: str,
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
      context_quadrants,
      prompt,
    )

    # Get model-specific queue position
    model_position = get_queue_position_for_model(conn, queue_item.id, model_id)

    # Get total queue length for backwards compatibility
    pending = get_pending_queue(conn)
    total_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": model_position,  # Position within this model's queue
      "total_position": total_position,  # Overall queue position
      "model_id": model_id,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {model_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)
    model_status = get_queue_status_by_model(conn)

    # Get list of currently busy models
    with busy_models_lock:
      active_models = list(busy_models)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
      # Per-model queue info
      "queue_by_model": model_status["by_model"],
      "processing_models": model_status["processing_models"],
      # All currently processing models (for parallel processing)
      "active_models": active_models,
      "active_model_count": len(active_models),
      # All quadrants being processed across all models
      "all_processing_quadrants": model_status["all_processing_quadrants"],
    }

    # Set is_generating based on whether any models are active
    response["is_generating"] = len(active_models) > 0

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
  """
  API endpoint to clear all items from the generation queue,
  including cancelling any currently running generation.

  Returns:
    {
      "success": true,
      "cleared_count": N,
      "cancelled_count": M,
      "message": "Cleared N pending item(s), cancelled M in-progress item(s)"
    }
  """
  global generation_state, generation_cancelled

  print(f"\n{'=' * 60}")
  print("🗑️  Clear queue request received")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # Clear pending items
    cleared_count = clear_pending_queue(conn)

    # Cancel any in-progress items
    cancelled_count = cancel_processing_items(conn)

    # Set the cancellation flag so the worker knows to stop
    if cancelled_count > 0:
      generation_cancelled = True

    # Clear busy models
    with busy_models_lock:
      busy_models.clear()

    # Clear per-model states
    model_generation_states.clear()

    # Reset the generation state
    generation_state["is_generating"] = False
    generation_state["quadrants"] = []
    generation_state["status"] = "idle"
    generation_state["message"] = "Queue cleared"
    generation_state["error"] = None
    generation_state["current_item_id"] = None

    if cleared_count > 0 or cancelled_count > 0:
      print(
        f"✅ Cleared {cleared_count} pending, cancelled {cancelled_count} in-progress"
      )
    else:
      print("ℹ️  Queue was already empty")

    message_parts = []
    if cleared_count > 0:
      message_parts.append(f"Cleared {cleared_count} pending item(s)")
    if cancelled_count > 0:
      message_parts.append(f"cancelled {cancelled_count} in-progress item(s)")
    message = ", ".join(message_parts) if message_parts else "Queue was already empty"

    return jsonify(
      {
        "success": True,
        "cleared_count": cleared_count,
        "cancelled_count": cancelled_count,
        "message": message,
      }
    )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/queue/cancel/<int:item_id>", methods=["POST"])
def api_cancel_queue_item(item_id: int):
  """
  API endpoint to cancel a specific queue item by its ID.

  Returns:
    {
      "success": true,
      "cancelled": true,
      "item_id": N,
      "message": "Cancelled queue item N"
    }
  """
  global generation_cancelled

  print(f"\n{'=' * 60}")
  print(f"🗑️  Cancel queue item request received: item_id={item_id}")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # First check if this item was processing (not just pending)
    cursor = conn.cursor()
    cursor.execute(
      "SELECT status, model_id FROM generation_queue WHERE id = ?",
      (item_id,),
    )
    row = cursor.fetchone()
    was_processing = row and row[0] == "processing"
    cancelled_model_id = row[1] if row else None

    cancelled = cancel_queue_item_by_id(conn, item_id)

    if cancelled:
      print(f"✅ Cancelled queue item {item_id}")

      # Only set the global cancellation flag if this was a PROCESSING item
      # Pending items just get marked as cancelled in the database
      if was_processing:
        generation_cancelled = True
        print("   ⚠️  Item was processing, signaling cancellation")

        # Also remove this model from busy set so it can pick up new work
        if cancelled_model_id:
          with busy_models_lock:
            busy_models.discard(cancelled_model_id)

      return jsonify(
        {
          "success": True,
          "cancelled": True,
          "item_id": item_id,
          "message": f"Cancelled queue item {item_id}",
        }
      )
    else:
      print(f"ℹ️  Queue item {item_id} not found or already completed")
      return jsonify(
        {
          "success": True,
          "cancelled": False,
          "item_id": item_id,
          "message": f"Queue item {item_id} not found or already completed",
        }
      )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify(
    {
      "models": [m.to_dict() for m in APP_CONFIG.models],
      "default_model_id": APP_CONFIG.default_model_id,
    }
  )


@app.route("/api/nyc-boundary")
def api_nyc_boundary():
  """
  API endpoint to get the NYC boundary GeoJSON with coordinate transformation info.

  Returns the NYC borough boundaries along with the generation config needed
  to transform lat/lng coordinates to quadrant (x, y) coordinates.
  """
  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    # Pre-compute boundary points in quadrant coordinates for the frontend
    # This avoids complex math in JavaScript
    boundary_in_quadrants = {"type": "FeatureCollection", "features": []}

    for feature in BOUNDARY_GEOJSON["features"]:
      new_feature = {
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": {"type": feature["geometry"]["type"], "coordinates": []},
      }

      # Process each ring of the polygon
      for ring in feature["geometry"]["coordinates"]:
        new_ring = []
        for coord in ring:
          lng, lat = coord[0], coord[1]
          qx, qy = latlng_to_quadrant_coords(config, lat, lng)
          new_ring.append([qx, qy])
        new_feature["geometry"]["coordinates"].append(new_ring)

      boundary_in_quadrants["features"].append(new_feature)

    return jsonify(
      {
        "boundary": boundary_in_quadrants,
        "seed": config["seed"],
      }
    )
  finally:
    conn.close()


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/delete-render", methods=["POST"])
def api_delete_render():
  """API endpoint to delete render data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the render column (set to NULL) but keep the row
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted render data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/flag", methods=["POST"])
def api_flag():
  """API endpoint to flag/unflag selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get flag value (default to True/1 for flagging, False/0 for unflagging)
  flag_value = 1 if data.get("flag", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the flagged column exists
    ensure_flagged_column_exists(conn)

    flagged_count = 0
    for qx, qy in quadrants:
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET flagged = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (flag_value, qx, qy),
      )
      if cursor.rowcount > 0:
        flagged_count += 1

    conn.commit()

    action = "Flagged" if flag_value else "Unflagged"
    return jsonify(
      {
        "success": True,
        "message": f"{action} {flagged_count} quadrant{'s' if flagged_count != 1 else ''}",
        "count": flagged_count,
        "flagged": bool(flag_value),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/water", methods=["POST"])
def api_water():
  """
  API endpoint to mark/unmark selected quadrants as water tiles.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed by script)
     1: Water tile

  Request body:
    {
      "quadrants": [[x, y], ...],
      "is_water": true/false,  // true=water(1), false=not water(0)
      "explicit_not_water": true  // Optional: if true, sets to -1 (protected)
    }
  """
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Determine water value:
  # - explicit_not_water=true → -1 (protected from auto-detection)
  # - is_water=true → 1 (water)
  # - is_water=false → 0 (not water, can be auto-changed)
  if data.get("explicit_not_water", False):
    water_value = -1
    action = "Marked as explicitly NOT water (protected)"
  elif data.get("is_water", True):
    water_value = 1
    action = "Marked as water"
  else:
    water_value = 0
    action = "Unmarked as water"

  conn = get_db_connection()

  try:
    # Ensure the is_water column exists
    ensure_is_water_column_exists(conn)

    water_count = 0
    for qx, qy in quadrants:
      # First ensure the quadrant exists in the database
      cursor = conn.execute(
        "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (qx, qy),
      )
      if cursor.fetchone() is None:
        # Quadrant doesn't exist, skip it
        continue

      cursor = conn.execute(
        """
        UPDATE quadrants
        SET is_water = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (water_value, qx, qy),
      )
      if cursor.rowcount > 0:
        water_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "water_status": water_value,
        "is_water": water_value == 1,
        "is_explicit_not_water": water_value == -1,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  print(f"\n{'=' * 60}")
  print(f"🎨 Render request: {selected_quadrants}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "render")
  return jsonify(result), 202  # 202 Accepted


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """
  API endpoint to generate tiles for selected quadrants.

  Request body:
    {
      "quadrants": [[x, y], ...] or [{"x": x, "y": y}, ...],
      "model_id": "optional-model-id",
      "context": [[x, y], ...] or [{"x": x, "y": y}, ...]  // Optional context quadrants
    }

  The context quadrants are used to provide surrounding pixel art context for
  the generation. If a context quadrant has an existing generation, that will
  be used; otherwise the render content will be used.
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Get optional model_id from request
  model_id = data.get("model_id")

  # Convert quadrants to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Parse optional context quadrants
  context_quadrants = None
  context_raw = data.get("context")
  if context_raw:
    if not isinstance(context_raw, list):
      return jsonify(
        {
          "success": False,
          "error": "context must be a list of quadrant coordinates",
        }
      ), 400

    context_quadrants = []
    for q in context_raw:
      if isinstance(q, list) and len(q) == 2:
        context_quadrants.append((int(q[0]), int(q[1])))
      elif isinstance(q, dict) and "x" in q and "y" in q:
        context_quadrants.append((int(q["x"]), int(q["y"])))
      else:
        return jsonify(
          {
            "success": False,
            "error": f"Invalid context quadrant format: {q}",
          }
        ), 400

  # Parse optional prompt
  prompt = data.get("prompt")
  if prompt and not isinstance(prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "prompt must be a string",
      }
    ), 400

  # Clean up prompt (strip whitespace, None if empty)
  if prompt:
    prompt = prompt.strip()
    if not prompt:
      prompt = None

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt
  )
  return jsonify(result), 202  # 202 Accepted


# =============================================================================
# Rectangle Generation API
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def load_queued_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """
  Load all quadrants from pending and processing queue items.

  These quadrants are scheduled for generation and should be considered
  when planning new rectangles to avoid seam issues.
  """
  queued: set[Point] = set()

  # Get pending items
  pending_items = get_pending_queue(conn)
  for item in pending_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  # Get processing items
  processing_items = get_all_processing_items(conn)
  for item in processing_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  return queued


@app.route("/api/generate-rectangle", methods=["POST"])
def api_generate_rectangle():
  """
  API endpoint to generate all quadrants within a rectangle.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "model_id": "optional-model-id"    // Optional model ID
    }

  Returns:
    {
      "success": true,
      "plan_summary": {...},
      "queued_count": N,
      "message": "Queued N generation steps"
    }
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl = Point(int(tl_raw[0]), int(tl_raw[1]))
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl = Point(int(tl_raw["x"]), int(tl_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br = Point(int(br_raw[0]), int(br_raw[1]))
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br = Point(int(br_raw["x"]), int(br_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  # Get optional model_id
  model_id = data.get("model_id")

  print(f"\n{'=' * 60}")
  print(f"📐 Rectangle generation request: ({tl.x},{tl.y}) to ({br.x},{br.y})")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Load existing generated quadrants and pending/processing quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)
    queued = load_queued_quadrants(conn)

    if queued:
      print(f"   Considering {len(queued)} queued/processing quadrant(s) for seam avoidance")

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated, queued)

    # Validate the plan
    is_valid, errors = validate_plan(plan)
    if not is_valid:
      print(f"❌ Invalid plan generated: {errors}")
      return jsonify(
        {
          "success": False,
          "error": f"Internal error: invalid plan generated - {errors}",
        }
      ), 500

    # Get plan summary for response
    summary = get_plan_summary(plan)

    if len(plan.steps) == 0:
      print("ℹ️  No quadrants to generate (all already generated)")
      return jsonify(
        {
          "success": True,
          "plan_summary": summary,
          "queued_count": 0,
          "message": "No quadrants to generate - all already generated",
        }
      )

    # Queue all generation steps
    queued_count = 0
    for step in plan.steps:
      quadrants = [(q.x, q.y) for q in step.quadrants]
      add_to_queue(conn, QueueItemType.GENERATE, quadrants, model_id)
      queued_count += 1

    # Ensure queue worker is running
    start_queue_worker()

    print(f"✅ Queued {queued_count} generation step(s)")
    print(f"   Steps by type: {summary['steps_by_type']}")

    return jsonify(
      {
        "success": True,
        "plan_summary": summary,
        "queued_count": queued_count,
        "message": f"Queued {queued_count} generation step(s) for {summary['total_quadrants']} quadrant(s)",
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG, BOUNDARY_GEOJSON

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
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to app_config.json (default: looks in the e2e_generation directory)",
  )
  parser.add_argument(
    "--bounds",
    type=Path,
    default=None,
    help="Path to custom bounds GeoJSON file (default: NYC boundary)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Load app configuration
  APP_CONFIG = load_app_config(args.config)
  print(f"📦 Loaded {len(APP_CONFIG.models)} model(s) from configuration")
  for model in APP_CONFIG.models:
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Reset any items that were mid-processing when server shut down
    # These will be retried automatically
    reset_count = reset_all_processing_items(conn)
    if reset_count > 0:
      print(f"🔄 Reset {reset_count} interrupted generation(s) - will be retried")
    # Clean up old completed items
    deleted_count = clear_completed_items(conn)
    if deleted_count > 0:
      print(f"🧹 Cleaned up {deleted_count} old completed queue item(s)")
  finally:
    conn.close()

  # Start the queue worker
  start_queue_worker()

  # Start the global web renderer
  try:
    start_global_renderer(port=WEB_SERVER_PORT)
  except Exception as e:
    print(f"⚠️  Failed to start web renderer: {e}")
    print("   Rendering will start on demand")

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web renderer
    print("🛑 Stopping web renderer...")
    stop_global_renderer()

  return 0


if __name__ == "__main__":
  exit(main())

```

```python
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
  R          - Toggle render/generation mode
  G          - Generate selected quadrants
  S          - Toggle select tool
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.make_rectangle_plan import (
  Point,
  RectBounds,
  create_rectangle_plan,
  get_plan_summary,
  validate_plan,
)
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cancel_processing_items,
  cancel_queue_item_by_id,
  clear_completed_items,
  clear_pending_queue,
  get_all_processing_items,
  get_next_pending_item_for_available_model,
  get_pending_queue,
  get_queue_position_for_model,
  get_queue_status,
  get_queue_status_by_model,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
  reset_all_processing_items,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
  latlng_to_quadrant_coords,
)

from isometric_nyc.e2e_generation.bounds import load_bounds

# Global boundary GeoJSON - loaded at startup
BOUNDARY_GEOJSON: dict | None = None
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)


# =============================================================================
# Logging Configuration - Suppress noisy tile request logs
# =============================================================================
class TileRequestFilter(logging.Filter):
  """Filter out noisy tile and static file requests from logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    message = record.getMessage()
    # Filter out tile requests, static files, and api/status polling
    if "/tile/" in message:
      return False
    if "/static/" in message:
      return False
    if "/api/status" in message:
      return False
    return True


# Apply filter to werkzeug logger (Flask's HTTP request logger)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(TileRequestFilter())

# Generation lock - protects generation_state updates
generation_lock = threading.Lock()

# Per-model generation states
# Key is model_id (None for default), value is state dict
model_generation_states: dict[str | None, dict] = {}

# Legacy global generation_state for backwards compatibility with API
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
  "current_item_id": None,
  "model_id": None,
}

# Track which models are currently processing
busy_models: set[str | None] = set()
busy_models_lock = threading.Lock()

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Cancellation flag - set to True to cancel all generations
generation_cancelled = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def ensure_flagged_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the flagged column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "flagged" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'flagged' column to quadrants table")


def ensure_is_water_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


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


def get_quadrant_info(x: int, y: int, use_render: bool = False) -> dict:
  """
  Get info about a quadrant including whether it has data, is flagged, and water status.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed)
     1: Water tile
  """
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      water_status = row[2]
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": water_status == 1,  # True if water
        "is_explicit_not_water": water_status == -1,  # True if explicitly not water
        "water_status": water_status,  # Raw value: -1, 0, or 1
      }
    return {
      "has_data": False,
      "flagged": False,
      "is_water": False,
      "is_explicit_not_water": False,
      "water_status": 0,
    }
  finally:
    conn.close()


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

  # Check which tiles have data, flagged status, and water status
  tiles = {}
  flagged_tiles = {}
  water_tiles = {}
  explicit_not_water_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      water_tiles[(dx, dy)] = info["is_water"]
      explicit_not_water_tiles[(dx, dy)] = info["is_explicit_not_water"]

  # Get model configuration for the frontend
  models_config = []
  default_model_id = None
  if APP_CONFIG:
    models_config = [m.to_dict() for m in APP_CONFIG.models]
    default_model_id = APP_CONFIG.default_model_id

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    flagged_tiles=flagged_tiles,
    water_tiles=water_tiles,
    explicit_not_water_tiles=explicit_not_water_tiles,
    generation_dir=str(GENERATION_DIR),
    models_config=json.dumps(models_config),
    default_model_id=default_model_id,
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

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = shared_get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation.
    prompt: Optional additional prompt text for generation

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants,
    prompt=prompt,
  )


def render_quadrant_with_renderer(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
) -> bytes | None:
  """
  Render a quadrant using the global web renderer.

  Returns the PNG bytes of the rendered quadrant.
  """
  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    save_quadrant_render,
  )
  from isometric_nyc.e2e_generation.web_renderer import get_web_renderer

  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  renderer = get_web_renderer(port=WEB_SERVER_PORT)

  # Render the tile
  quadrant_images = renderer.render_quadrant(
    quadrant_x=x,
    quadrant_y=y,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    camera_azimuth_degrees=config["camera_azimuth_degrees"],
    camera_elevation_degrees=config["camera_elevation_degrees"],
    view_height_meters=config.get("view_height_meters", 200),
  )

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), png_bytes in quadrant_images.items():
    qx, qy = x + dx, y + dy
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


def process_queue_item_from_db(item_id: int) -> dict:
  """Process a single queue item from the database."""
  global generation_state

  conn = get_db_connection()
  try:
    # Get item details
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT item_type, quadrants, model_id, context_quadrants, prompt
      FROM generation_queue
      WHERE id = ?
      """,
      (item_id,),
    )
    row = cursor.fetchone()
    if not row:
      return {"success": False, "error": "Item not found"}

    item_type = QueueItemType(row[0])
    quadrants = json.loads(row[1])
    model_id = row[2]
    context_quadrants_raw = json.loads(row[3]) if row[3] else None
    prompt = row[4]

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Calculate context lazily if not explicitly provided
    # This ensures we use the most up-to-date context based on what's
    # actually generated at execution time (not queue time)
    if context_quadrants_raw:
      context_quadrants = [(q[0], q[1]) for q in context_quadrants_raw]
      print(
        f"   📋 Using explicit context from queue: {len(context_quadrants)} quadrant(s)"
      )
    else:
      # Calculate context lazily based on current generation state
      context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
      if context_quadrants:
        print(f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s)")
      else:
        print(
          "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
        )

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = (
      "starting" if item_type == QueueItemType.GENERATE else "rendering"
    )
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(
      f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}"
    )
    if model_id:
      print(f"   Model: {model_id}")
    if context_quadrants:
      print(f"   Context: {context_quadrants}")
    if prompt:
      print(f"   Prompt: {prompt}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt
        )

        if result["success"]:
          print(f"✅ Generation complete: {result['message']}")
          generation_state["status"] = "complete"
          generation_state["message"] = result["message"]
          mark_item_complete(conn, item_id, result["message"])
          return result

        # Generation failed
        if gen_attempt < max_generation_retries:
          print(
            f"⚠️  Generation failed (attempt {gen_attempt}/{max_generation_retries}): "
            f"{result['error']}"
          )
          print(f"⏳ Waiting {generation_retry_delay}s before retrying generation...")
          update_generation_state(
            "retrying",
            f"Generation failed, retrying (attempt {gen_attempt + 1}/{max_generation_retries})...",
          )
          time.sleep(generation_retry_delay)
        else:
          # All retries exhausted
          print(
            f"❌ Generation failed after {max_generation_retries} attempts: "
            f"{result['error']}"
          )
          generation_state["status"] = "error"
          generation_state["error"] = result["error"]
          mark_item_error(conn, item_id, result["error"])
          return result

      # Should not reach here, but just in case
      return result

    else:  # render
      update_generation_state("rendering", "Starting render...")

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant_with_renderer(conn, config, qx, qy)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      result_message = f"Rendered {rendered_count} quadrant(s)"
      update_generation_state("complete", result_message)
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")
      mark_item_complete(conn, item_id, result_message)

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    mark_item_error(conn, item_id, str(e))
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def process_model_item(item_id: int, model_id: str | None):
  """Process a single queue item for a specific model in its own thread."""
  global generation_state, generation_cancelled

  try:
    # Check cancellation before starting
    if generation_cancelled:
      print(f"⚠️  Item {item_id} cancelled before processing")
      return

    process_queue_item_from_db(item_id)

  except Exception as e:
    print(f"❌ Model worker error for {model_id}: {e}")
    traceback.print_exc()
  finally:
    # Remove model from busy set
    with busy_models_lock:
      busy_models.discard(model_id)

    # Update global state if this was the active model
    with generation_lock:
      if generation_state.get("model_id") == model_id:
        generation_state["is_generating"] = False
        generation_state["current_item_id"] = None

    # Remove from per-model states
    if model_id in model_generation_states:
      del model_generation_states[model_id]


def queue_worker():
  """Background worker that processes the generation queue from the database.

  This worker supports parallel processing of different models - each model
  can have one active generation at a time, but different models can run
  concurrently.
  """
  global generation_state, queue_worker_running, generation_cancelled

  print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
      # Check if we were cancelled
      if generation_cancelled:
        print("⚠️  Generation cancelled, resetting flags...")
        generation_cancelled = False
        with generation_lock:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None
        with busy_models_lock:
          busy_models.clear()
        model_generation_states.clear()
        time.sleep(0.5)
        continue

      conn = get_db_connection()

      # Get current busy models
      with busy_models_lock:
        current_busy = busy_models.copy()

      # Get next pending item for an available model
      item = get_next_pending_item_for_available_model(conn, current_busy)

      if item is None:
        # No items available (either queue empty or all models busy)
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      model_id = item.model_id
      conn.close()
      conn = None

      # Mark this model as busy
      with busy_models_lock:
        if model_id in busy_models:
          # Another thread grabbed this model, skip
          continue
        busy_models.add(model_id)

      # Update global state for display (use most recent)
      with generation_lock:
        generation_state["is_generating"] = True
        generation_state["model_id"] = model_id

      # Initialize per-model state
      model_generation_states[model_id] = {
        "is_generating": True,
        "item_id": item_id,
        "started_at": time.time(),
      }

      # Spawn a thread to process this model's item
      model_name = model_id or "default"
      worker_thread = threading.Thread(
        target=process_model_item,
        args=(item_id, model_id),
        name=f"model-worker-{model_name}",
        daemon=True,
      )
      worker_thread.start()

      print(f"🚀 Started worker for model '{model_name}' (item {item_id})")

      # Small delay before checking for more work
      time.sleep(0.2)

    except Exception as e:
      print(f"❌ Queue worker error: {e}")
      traceback.print_exc()
      time.sleep(1.0)
    finally:
      if conn:
        conn.close()

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue_db(
  quadrants: list[tuple[int, int]],
  item_type: str,
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
      context_quadrants,
      prompt,
    )

    # Get model-specific queue position
    model_position = get_queue_position_for_model(conn, queue_item.id, model_id)

    # Get total queue length for backwards compatibility
    pending = get_pending_queue(conn)
    total_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": model_position,  # Position within this model's queue
      "total_position": total_position,  # Overall queue position
      "model_id": model_id,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {model_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)
    model_status = get_queue_status_by_model(conn)

    # Get list of currently busy models
    with busy_models_lock:
      active_models = list(busy_models)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
      # Per-model queue info
      "queue_by_model": model_status["by_model"],
      "processing_models": model_status["processing_models"],
      # All currently processing models (for parallel processing)
      "active_models": active_models,
      "active_model_count": len(active_models),
      # All quadrants being processed across all models
      "all_processing_quadrants": model_status["all_processing_quadrants"],
    }

    # Set is_generating based on whether any models are active
    response["is_generating"] = len(active_models) > 0

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
  """
  API endpoint to clear all items from the generation queue,
  including cancelling any currently running generation.

  Returns:
    {
      "success": true,
      "cleared_count": N,
      "cancelled_count": M,
      "message": "Cleared N pending item(s), cancelled M in-progress item(s)"
    }
  """
  global generation_state, generation_cancelled

  print(f"\n{'=' * 60}")
  print("🗑️  Clear queue request received")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # Clear pending items
    cleared_count = clear_pending_queue(conn)

    # Cancel any in-progress items
    cancelled_count = cancel_processing_items(conn)

    # Set the cancellation flag so the worker knows to stop
    if cancelled_count > 0:
      generation_cancelled = True

    # Clear busy models
    with busy_models_lock:
      busy_models.clear()

    # Clear per-model states
    model_generation_states.clear()

    # Reset the generation state
    generation_state["is_generating"] = False
    generation_state["quadrants"] = []
    generation_state["status"] = "idle"
    generation_state["message"] = "Queue cleared"
    generation_state["error"] = None
    generation_state["current_item_id"] = None

    if cleared_count > 0 or cancelled_count > 0:
      print(
        f"✅ Cleared {cleared_count} pending, cancelled {cancelled_count} in-progress"
      )
    else:
      print("ℹ️  Queue was already empty")

    message_parts = []
    if cleared_count > 0:
      message_parts.append(f"Cleared {cleared_count} pending item(s)")
    if cancelled_count > 0:
      message_parts.append(f"cancelled {cancelled_count} in-progress item(s)")
    message = ", ".join(message_parts) if message_parts else "Queue was already empty"

    return jsonify(
      {
        "success": True,
        "cleared_count": cleared_count,
        "cancelled_count": cancelled_count,
        "message": message,
      }
    )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/queue/cancel/<int:item_id>", methods=["POST"])
def api_cancel_queue_item(item_id: int):
  """
  API endpoint to cancel a specific queue item by its ID.

  Returns:
    {
      "success": true,
      "cancelled": true,
      "item_id": N,
      "message": "Cancelled queue item N"
    }
  """
  global generation_cancelled

  print(f"\n{'=' * 60}")
  print(f"🗑️  Cancel queue item request received: item_id={item_id}")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # First check if this item was processing (not just pending)
    cursor = conn.cursor()
    cursor.execute(
      "SELECT status, model_id FROM generation_queue WHERE id = ?",
      (item_id,),
    )
    row = cursor.fetchone()
    was_processing = row and row[0] == "processing"
    cancelled_model_id = row[1] if row else None

    cancelled = cancel_queue_item_by_id(conn, item_id)

    if cancelled:
      print(f"✅ Cancelled queue item {item_id}")

      # Only set the global cancellation flag if this was a PROCESSING item
      # Pending items just get marked as cancelled in the database
      if was_processing:
        generation_cancelled = True
        print("   ⚠️  Item was processing, signaling cancellation")

        # Also remove this model from busy set so it can pick up new work
        if cancelled_model_id:
          with busy_models_lock:
            busy_models.discard(cancelled_model_id)

      return jsonify(
        {
          "success": True,
          "cancelled": True,
          "item_id": item_id,
          "message": f"Cancelled queue item {item_id}",
        }
      )
    else:
      print(f"ℹ️  Queue item {item_id} not found or already completed")
      return jsonify(
        {
          "success": True,
          "cancelled": False,
          "item_id": item_id,
          "message": f"Queue item {item_id} not found or already completed",
        }
      )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify(
    {
      "models": [m.to_dict() for m in APP_CONFIG.models],
      "default_model_id": APP_CONFIG.default_model_id,
    }
  )


@app.route("/api/nyc-boundary")
def api_nyc_boundary():
  """
  API endpoint to get the NYC boundary GeoJSON with coordinate transformation info.

  Returns the NYC borough boundaries along with the generation config needed
  to transform lat/lng coordinates to quadrant (x, y) coordinates.
  """
  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    # Pre-compute boundary points in quadrant coordinates for the frontend
    # This avoids complex math in JavaScript
    boundary_in_quadrants = {"type": "FeatureCollection", "features": []}

    for feature in BOUNDARY_GEOJSON["features"]:
      new_feature = {
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": {"type": feature["geometry"]["type"], "coordinates": []},
      }

      # Process each ring of the polygon
      for ring in feature["geometry"]["coordinates"]:
        new_ring = []
        for coord in ring:
          lng, lat = coord[0], coord[1]
          qx, qy = latlng_to_quadrant_coords(config, lat, lng)
          new_ring.append([qx, qy])
        new_feature["geometry"]["coordinates"].append(new_ring)

      boundary_in_quadrants["features"].append(new_feature)

    return jsonify(
      {
        "boundary": boundary_in_quadrants,
        "seed": config["seed"],
      }
    )
  finally:
    conn.close()


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/delete-render", methods=["POST"])
def api_delete_render():
  """API endpoint to delete render data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the render column (set to NULL) but keep the row
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted render data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/flag", methods=["POST"])
def api_flag():
  """API endpoint to flag/unflag selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get flag value (default to True/1 for flagging, False/0 for unflagging)
  flag_value = 1 if data.get("flag", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the flagged column exists
    ensure_flagged_column_exists(conn)

    flagged_count = 0
    for qx, qy in quadrants:
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET flagged = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (flag_value, qx, qy),
      )
      if cursor.rowcount > 0:
        flagged_count += 1

    conn.commit()

    action = "Flagged" if flag_value else "Unflagged"
    return jsonify(
      {
        "success": True,
        "message": f"{action} {flagged_count} quadrant{'s' if flagged_count != 1 else ''}",
        "count": flagged_count,
        "flagged": bool(flag_value),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/water", methods=["POST"])
def api_water():
  """
  API endpoint to mark/unmark selected quadrants as water tiles.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed by script)
     1: Water tile

  Request body:
    {
      "quadrants": [[x, y], ...],
      "is_water": true/false,  // true=water(1), false=not water(0)
      "explicit_not_water": true  // Optional: if true, sets to -1 (protected)
    }
  """
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Determine water value:
  # - explicit_not_water=true → -1 (protected from auto-detection)
  # - is_water=true → 1 (water)
  # - is_water=false → 0 (not water, can be auto-changed)
  if data.get("explicit_not_water", False):
    water_value = -1
    action = "Marked as explicitly NOT water (protected)"
  elif data.get("is_water", True):
    water_value = 1
    action = "Marked as water"
  else:
    water_value = 0
    action = "Unmarked as water"

  conn = get_db_connection()

  try:
    # Ensure the is_water column exists
    ensure_is_water_column_exists(conn)

    water_count = 0
    for qx, qy in quadrants:
      # First ensure the quadrant exists in the database
      cursor = conn.execute(
        "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (qx, qy),
      )
      if cursor.fetchone() is None:
        # Quadrant doesn't exist, skip it
        continue

      cursor = conn.execute(
        """
        UPDATE quadrants
        SET is_water = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (water_value, qx, qy),
      )
      if cursor.rowcount > 0:
        water_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "water_status": water_value,
        "is_water": water_value == 1,
        "is_explicit_not_water": water_value == -1,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  print(f"\n{'=' * 60}")
  print(f"🎨 Render request: {selected_quadrants}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "render")
  return jsonify(result), 202  # 202 Accepted


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """
  API endpoint to generate tiles for selected quadrants.

  Request body:
    {
      "quadrants": [[x, y], ...] or [{"x": x, "y": y}, ...],
      "model_id": "optional-model-id",
      "context": [[x, y], ...] or [{"x": x, "y": y}, ...]  // Optional context quadrants
    }

  The context quadrants are used to provide surrounding pixel art context for
  the generation. If a context quadrant has an existing generation, that will
  be used; otherwise the render content will be used.
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Get optional model_id from request
  model_id = data.get("model_id")

  # Convert quadrants to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Parse optional context quadrants
  context_quadrants = None
  context_raw = data.get("context")
  if context_raw:
    if not isinstance(context_raw, list):
      return jsonify(
        {
          "success": False,
          "error": "context must be a list of quadrant coordinates",
        }
      ), 400

    context_quadrants = []
    for q in context_raw:
      if isinstance(q, list) and len(q) == 2:
        context_quadrants.append((int(q[0]), int(q[1])))
      elif isinstance(q, dict) and "x" in q and "y" in q:
        context_quadrants.append((int(q["x"]), int(q["y"])))
      else:
        return jsonify(
          {
            "success": False,
            "error": f"Invalid context quadrant format: {q}",
          }
        ), 400

  # Parse optional prompt
  prompt = data.get("prompt")
  if prompt and not isinstance(prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "prompt must be a string",
      }
    ), 400

  # Clean up prompt (strip whitespace, None if empty)
  if prompt:
    prompt = prompt.strip()
    if not prompt:
      prompt = None

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt
  )
  return jsonify(result), 202  # 202 Accepted


# =============================================================================
# Rectangle Generation API
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def load_queued_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """
  Load all quadrants from pending and processing queue items.

  These quadrants are scheduled for generation and should be considered
  when planning new rectangles to avoid seam issues.
  """
  queued: set[Point] = set()

  # Get pending items
  pending_items = get_pending_queue(conn)
  for item in pending_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  # Get processing items
  processing_items = get_all_processing_items(conn)
  for item in processing_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  return queued


@app.route("/api/generate-rectangle", methods=["POST"])
def api_generate_rectangle():
  """
  API endpoint to generate all quadrants within a rectangle.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "model_id": "optional-model-id"    // Optional model ID
    }

  Returns:
    {
      "success": true,
      "plan_summary": {...},
      "queued_count": N,
      "message": "Queued N generation steps"
    }
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl = Point(int(tl_raw[0]), int(tl_raw[1]))
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl = Point(int(tl_raw["x"]), int(tl_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br = Point(int(br_raw[0]), int(br_raw[1]))
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br = Point(int(br_raw["x"]), int(br_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  # Get optional model_id
  model_id = data.get("model_id")

  print(f"\n{'=' * 60}")
  print(f"📐 Rectangle generation request: ({tl.x},{tl.y}) to ({br.x},{br.y})")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Load existing generated quadrants and pending/processing quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)
    queued = load_queued_quadrants(conn)

    if queued:
      print(f"   Considering {len(queued)} queued/processing quadrant(s) for seam avoidance")

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated, queued)

    # Validate the plan
    is_valid, errors = validate_plan(plan)
    if not is_valid:
      print(f"❌ Invalid plan generated: {errors}")
      return jsonify(
        {
          "success": False,
          "error": f"Internal error: invalid plan generated - {errors}",
        }
      ), 500

    # Get plan summary for response
    summary = get_plan_summary(plan)

    if len(plan.steps) == 0:
      print("ℹ️  No quadrants to generate (all already generated)")
      return jsonify(
        {
          "success": True,
          "plan_summary": summary,
          "queued_count": 0,
          "message": "No quadrants to generate - all already generated",
        }
      )

    # Queue all generation steps
    queued_count = 0
    for step in plan.steps:
      quadrants = [(q.x, q.y) for q in step.quadrants]
      add_to_queue(conn, QueueItemType.GENERATE, quadrants, model_id)
      queued_count += 1

    # Ensure queue worker is running
    start_queue_worker()

    print(f"✅ Queued {queued_count} generation step(s)")
    print(f"   Steps by type: {summary['steps_by_type']}")

    return jsonify(
      {
        "success": True,
        "plan_summary": summary,
        "queued_count": queued_count,
        "message": f"Queued {queued_count} generation step(s) for {summary['total_quadrants']} quadrant(s)",
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG, BOUNDARY_GEOJSON

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
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to app_config.json (default: looks in the e2e_generation directory)",
  )
  parser.add_argument(
    "--bounds",
    type=Path,
    default=None,
    help="Path to custom bounds GeoJSON file (default: NYC boundary)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Load app configuration
  APP_CONFIG = load_app_config(args.config)
  print(f"📦 Loaded {len(APP_CONFIG.models)} model(s) from configuration")
  for model in APP_CONFIG.models:
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Load boundary GeoJSON
  bounds_path = args.bounds.resolve() if args.bounds else None
  BOUNDARY_GEOJSON = load_bounds(bounds_path)
  bounds_name = bounds_path.name if bounds_path else "NYC (default)"
  print(f"📍 Boundary: {bounds_name}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Reset any items that were mid-processing when server shut down
    # These will be retried automatically
    reset_count = reset_all_processing_items(conn)
    if reset_count > 0:
      print(f"🔄 Reset {reset_count} interrupted generation(s) - will be retried")
    # Clean up old completed items
    deleted_count = clear_completed_items(conn)
    if deleted_count > 0:
      print(f"🧹 Cleaned up {deleted_count} old completed queue item(s)")
  finally:
    conn.close()

  # Start the queue worker
  start_queue_worker()

  # Start the global web renderer
  try:
    start_global_renderer(port=WEB_SERVER_PORT)
  except Exception as e:
    print(f"⚠️  Failed to start web renderer: {e}")
    print("   Rendering will start on demand")

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web renderer
    print("🛑 Stopping web renderer...")
    stop_global_renderer()

  return 0


if __name__ == "__main__":
  exit(main())

```

```python
"""
Create Bounds - Interactive polygon editor for defining generation boundaries.

This script creates an interactive web-based polygon editor that allows you to:
- Draw and edit polygon boundaries on a map
- View existing generated and pending tiles
- Load and edit existing boundary files
- Save new boundaries to the bounds directory

Usage:
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir] --load bounds/custom.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --load: Path to an existing bounds JSON file to edit

Output:
  Opens a web browser with the polygon editor
  Saves new bounds to e2e_generation/bounds/ directory
"""

import argparse
import json
import sqlite3
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlparse

from isometric_nyc.e2e_generation.bounds import get_bounds_dir, load_bounds, save_bounds
from isometric_nyc.e2e_generation.debug_map import (
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

  # Bounds directory path for saving
  bounds_dir_str = str(bounds_dir) if bounds_dir else str(get_bounds_dir())

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

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
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
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
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

```

Now let me check for linting errors in the files I've modified:

Let me fix the linting errors:

```python
"""
Create Bounds - Interactive polygon editor for defining generation boundaries.

This script creates an interactive web-based polygon editor that allows you to:
- Draw and edit polygon boundaries on a map
- View existing generated and pending tiles
- Load and edit existing boundary files
- Save new boundaries to the bounds directory

Usage:
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir] --load bounds/custom.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --load: Path to an existing bounds JSON file to edit

Output:
  Opens a web browser with the polygon editor
  Saves new bounds to e2e_generation/bounds/ directory
"""

import argparse
import json
import sqlite3
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import get_bounds_dir, load_bounds, save_bounds
from isometric_nyc.e2e_generation.debug_map import (
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

  # Bounds directory path for saving
  bounds_dir_str = str(bounds_dir) if bounds_dir else str(get_bounds_dir())

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

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
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
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
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


```

```python
"""
Create Bounds - Interactive polygon editor for defining generation boundaries.

This script creates an interactive web-based polygon editor that allows you to:
- Draw and edit polygon boundaries on a map
- View existing generated and pending tiles
- Load and edit existing boundary files
- Save new boundaries to the bounds directory

Usage:
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir] --load bounds/custom.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --load: Path to an existing bounds JSON file to edit

Output:
  Opens a web browser with the polygon editor
  Saves new bounds to e2e_generation/bounds/ directory
"""

import argparse
import json
import sqlite3
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import get_bounds_dir, load_bounds, save_bounds
from isometric_nyc.e2e_generation.debug_map import (
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

    // Initialize map
    const map = L.map('map', {{
      zoomControl: true,
      attributionControl: true
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
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
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


```

```python
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
  R          - Toggle render/generation mode
  G          - Generate selected quadrants
  S          - Toggle select tool
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.make_rectangle_plan import (
  Point,
  RectBounds,
  create_rectangle_plan,
  get_plan_summary,
  validate_plan,
)
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cancel_processing_items,
  cancel_queue_item_by_id,
  clear_completed_items,
  clear_pending_queue,
  get_all_processing_items,
  get_next_pending_item_for_available_model,
  get_pending_queue,
  get_queue_position_for_model,
  get_queue_status,
  get_queue_status_by_model,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
  reset_all_processing_items,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
  latlng_to_quadrant_coords,
)
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)


# =============================================================================
# Logging Configuration - Suppress noisy tile request logs
# =============================================================================
class TileRequestFilter(logging.Filter):
  """Filter out noisy tile and static file requests from logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    message = record.getMessage()
    # Filter out tile requests, static files, and api/status polling
    if "/tile/" in message:
      return False
    if "/static/" in message:
      return False
    if "/api/status" in message:
      return False
    return True


# Apply filter to werkzeug logger (Flask's HTTP request logger)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(TileRequestFilter())

# Generation lock - protects generation_state updates
generation_lock = threading.Lock()

# Per-model generation states
# Key is model_id (None for default), value is state dict
model_generation_states: dict[str | None, dict] = {}

# Legacy global generation_state for backwards compatibility with API
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
  "current_item_id": None,
  "model_id": None,
}

# Track which models are currently processing
busy_models: set[str | None] = set()
busy_models_lock = threading.Lock()

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Cancellation flag - set to True to cancel all generations
generation_cancelled = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def ensure_flagged_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the flagged column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "flagged" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'flagged' column to quadrants table")


def ensure_is_water_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


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


def get_quadrant_info(x: int, y: int, use_render: bool = False) -> dict:
  """
  Get info about a quadrant including whether it has data, is flagged, and water status.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed)
     1: Water tile
  """
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      water_status = row[2]
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": water_status == 1,  # True if water
        "is_explicit_not_water": water_status == -1,  # True if explicitly not water
        "water_status": water_status,  # Raw value: -1, 0, or 1
      }
    return {
      "has_data": False,
      "flagged": False,
      "is_water": False,
      "is_explicit_not_water": False,
      "water_status": 0,
    }
  finally:
    conn.close()


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

  # Check which tiles have data, flagged status, and water status
  tiles = {}
  flagged_tiles = {}
  water_tiles = {}
  explicit_not_water_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      water_tiles[(dx, dy)] = info["is_water"]
      explicit_not_water_tiles[(dx, dy)] = info["is_explicit_not_water"]

  # Get model configuration for the frontend
  models_config = []
  default_model_id = None
  if APP_CONFIG:
    models_config = [m.to_dict() for m in APP_CONFIG.models]
    default_model_id = APP_CONFIG.default_model_id

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    flagged_tiles=flagged_tiles,
    water_tiles=water_tiles,
    explicit_not_water_tiles=explicit_not_water_tiles,
    generation_dir=str(GENERATION_DIR),
    models_config=json.dumps(models_config),
    default_model_id=default_model_id,
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

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = shared_get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation.
    prompt: Optional additional prompt text for generation

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants,
    prompt=prompt,
  )


def render_quadrant_with_renderer(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
) -> bytes | None:
  """
  Render a quadrant using the global web renderer.

  Returns the PNG bytes of the rendered quadrant.
  """
  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    save_quadrant_render,
  )
  from isometric_nyc.e2e_generation.web_renderer import get_web_renderer

  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  renderer = get_web_renderer(port=WEB_SERVER_PORT)

  # Render the tile
  quadrant_images = renderer.render_quadrant(
    quadrant_x=x,
    quadrant_y=y,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    camera_azimuth_degrees=config["camera_azimuth_degrees"],
    camera_elevation_degrees=config["camera_elevation_degrees"],
    view_height_meters=config.get("view_height_meters", 200),
  )

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), png_bytes in quadrant_images.items():
    qx, qy = x + dx, y + dy
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


def process_queue_item_from_db(item_id: int) -> dict:
  """Process a single queue item from the database."""
  global generation_state

  conn = get_db_connection()
  try:
    # Get item details
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT item_type, quadrants, model_id, context_quadrants, prompt
      FROM generation_queue
      WHERE id = ?
      """,
      (item_id,),
    )
    row = cursor.fetchone()
    if not row:
      return {"success": False, "error": "Item not found"}

    item_type = QueueItemType(row[0])
    quadrants = json.loads(row[1])
    model_id = row[2]
    context_quadrants_raw = json.loads(row[3]) if row[3] else None
    prompt = row[4]

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Calculate context lazily if not explicitly provided
    # This ensures we use the most up-to-date context based on what's
    # actually generated at execution time (not queue time)
    if context_quadrants_raw:
      context_quadrants = [(q[0], q[1]) for q in context_quadrants_raw]
      print(
        f"   📋 Using explicit context from queue: {len(context_quadrants)} quadrant(s)"
      )
    else:
      # Calculate context lazily based on current generation state
      context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
      if context_quadrants:
        print(f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s)")
      else:
        print(
          "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
        )

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = (
      "starting" if item_type == QueueItemType.GENERATE else "rendering"
    )
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(
      f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}"
    )
    if model_id:
      print(f"   Model: {model_id}")
    if context_quadrants:
      print(f"   Context: {context_quadrants}")
    if prompt:
      print(f"   Prompt: {prompt}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt
        )

        if result["success"]:
          print(f"✅ Generation complete: {result['message']}")
          generation_state["status"] = "complete"
          generation_state["message"] = result["message"]
          mark_item_complete(conn, item_id, result["message"])
          return result

        # Generation failed
        if gen_attempt < max_generation_retries:
          print(
            f"⚠️  Generation failed (attempt {gen_attempt}/{max_generation_retries}): "
            f"{result['error']}"
          )
          print(f"⏳ Waiting {generation_retry_delay}s before retrying generation...")
          update_generation_state(
            "retrying",
            f"Generation failed, retrying (attempt {gen_attempt + 1}/{max_generation_retries})...",
          )
          time.sleep(generation_retry_delay)
        else:
          # All retries exhausted
          print(
            f"❌ Generation failed after {max_generation_retries} attempts: "
            f"{result['error']}"
          )
          generation_state["status"] = "error"
          generation_state["error"] = result["error"]
          mark_item_error(conn, item_id, result["error"])
          return result

      # Should not reach here, but just in case
      return result

    else:  # render
      update_generation_state("rendering", "Starting render...")

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant_with_renderer(conn, config, qx, qy)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      result_message = f"Rendered {rendered_count} quadrant(s)"
      update_generation_state("complete", result_message)
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")
      mark_item_complete(conn, item_id, result_message)

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    mark_item_error(conn, item_id, str(e))
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def process_model_item(item_id: int, model_id: str | None):
  """Process a single queue item for a specific model in its own thread."""
  global generation_state, generation_cancelled

  try:
    # Check cancellation before starting
    if generation_cancelled:
      print(f"⚠️  Item {item_id} cancelled before processing")
      return

    process_queue_item_from_db(item_id)

  except Exception as e:
    print(f"❌ Model worker error for {model_id}: {e}")
    traceback.print_exc()
  finally:
    # Remove model from busy set
    with busy_models_lock:
      busy_models.discard(model_id)

    # Update global state if this was the active model
    with generation_lock:
      if generation_state.get("model_id") == model_id:
        generation_state["is_generating"] = False
        generation_state["current_item_id"] = None

    # Remove from per-model states
    if model_id in model_generation_states:
      del model_generation_states[model_id]


def queue_worker():
  """Background worker that processes the generation queue from the database.

  This worker supports parallel processing of different models - each model
  can have one active generation at a time, but different models can run
  concurrently.
  """
  global generation_state, queue_worker_running, generation_cancelled

  print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
      # Check if we were cancelled
      if generation_cancelled:
        print("⚠️  Generation cancelled, resetting flags...")
        generation_cancelled = False
        with generation_lock:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None
        with busy_models_lock:
          busy_models.clear()
        model_generation_states.clear()
        time.sleep(0.5)
        continue

      conn = get_db_connection()

      # Get current busy models
      with busy_models_lock:
        current_busy = busy_models.copy()

      # Get next pending item for an available model
      item = get_next_pending_item_for_available_model(conn, current_busy)

      if item is None:
        # No items available (either queue empty or all models busy)
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      model_id = item.model_id
      conn.close()
      conn = None

      # Mark this model as busy
      with busy_models_lock:
        if model_id in busy_models:
          # Another thread grabbed this model, skip
          continue
        busy_models.add(model_id)

      # Update global state for display (use most recent)
      with generation_lock:
        generation_state["is_generating"] = True
        generation_state["model_id"] = model_id

      # Initialize per-model state
      model_generation_states[model_id] = {
        "is_generating": True,
        "item_id": item_id,
        "started_at": time.time(),
      }

      # Spawn a thread to process this model's item
      model_name = model_id or "default"
      worker_thread = threading.Thread(
        target=process_model_item,
        args=(item_id, model_id),
        name=f"model-worker-{model_name}",
        daemon=True,
      )
      worker_thread.start()

      print(f"🚀 Started worker for model '{model_name}' (item {item_id})")

      # Small delay before checking for more work
      time.sleep(0.2)

    except Exception as e:
      print(f"❌ Queue worker error: {e}")
      traceback.print_exc()
      time.sleep(1.0)
    finally:
      if conn:
        conn.close()

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue_db(
  quadrants: list[tuple[int, int]],
  item_type: str,
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
      context_quadrants,
      prompt,
    )

    # Get model-specific queue position
    model_position = get_queue_position_for_model(conn, queue_item.id, model_id)

    # Get total queue length for backwards compatibility
    pending = get_pending_queue(conn)
    total_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": model_position,  # Position within this model's queue
      "total_position": total_position,  # Overall queue position
      "model_id": model_id,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {model_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)
    model_status = get_queue_status_by_model(conn)

    # Get list of currently busy models
    with busy_models_lock:
      active_models = list(busy_models)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
      # Per-model queue info
      "queue_by_model": model_status["by_model"],
      "processing_models": model_status["processing_models"],
      # All currently processing models (for parallel processing)
      "active_models": active_models,
      "active_model_count": len(active_models),
      # All quadrants being processed across all models
      "all_processing_quadrants": model_status["all_processing_quadrants"],
    }

    # Set is_generating based on whether any models are active
    response["is_generating"] = len(active_models) > 0

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
  """
  API endpoint to clear all items from the generation queue,
  including cancelling any currently running generation.

  Returns:
    {
      "success": true,
      "cleared_count": N,
      "cancelled_count": M,
      "message": "Cleared N pending item(s), cancelled M in-progress item(s)"
    }
  """
  global generation_state, generation_cancelled

  print(f"\n{'=' * 60}")
  print("🗑️  Clear queue request received")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # Clear pending items
    cleared_count = clear_pending_queue(conn)

    # Cancel any in-progress items
    cancelled_count = cancel_processing_items(conn)

    # Set the cancellation flag so the worker knows to stop
    if cancelled_count > 0:
      generation_cancelled = True

    # Clear busy models
    with busy_models_lock:
      busy_models.clear()

    # Clear per-model states
    model_generation_states.clear()

    # Reset the generation state
    generation_state["is_generating"] = False
    generation_state["quadrants"] = []
    generation_state["status"] = "idle"
    generation_state["message"] = "Queue cleared"
    generation_state["error"] = None
    generation_state["current_item_id"] = None

    if cleared_count > 0 or cancelled_count > 0:
      print(
        f"✅ Cleared {cleared_count} pending, cancelled {cancelled_count} in-progress"
      )
    else:
      print("ℹ️  Queue was already empty")

    message_parts = []
    if cleared_count > 0:
      message_parts.append(f"Cleared {cleared_count} pending item(s)")
    if cancelled_count > 0:
      message_parts.append(f"cancelled {cancelled_count} in-progress item(s)")
    message = ", ".join(message_parts) if message_parts else "Queue was already empty"

    return jsonify(
      {
        "success": True,
        "cleared_count": cleared_count,
        "cancelled_count": cancelled_count,
        "message": message,
      }
    )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/queue/cancel/<int:item_id>", methods=["POST"])
def api_cancel_queue_item(item_id: int):
  """
  API endpoint to cancel a specific queue item by its ID.

  Returns:
    {
      "success": true,
      "cancelled": true,
      "item_id": N,
      "message": "Cancelled queue item N"
    }
  """
  global generation_cancelled

  print(f"\n{'=' * 60}")
  print(f"🗑️  Cancel queue item request received: item_id={item_id}")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # First check if this item was processing (not just pending)
    cursor = conn.cursor()
    cursor.execute(
      "SELECT status, model_id FROM generation_queue WHERE id = ?",
      (item_id,),
    )
    row = cursor.fetchone()
    was_processing = row and row[0] == "processing"
    cancelled_model_id = row[1] if row else None

    cancelled = cancel_queue_item_by_id(conn, item_id)

    if cancelled:
      print(f"✅ Cancelled queue item {item_id}")

      # Only set the global cancellation flag if this was a PROCESSING item
      # Pending items just get marked as cancelled in the database
      if was_processing:
        generation_cancelled = True
        print("   ⚠️  Item was processing, signaling cancellation")

        # Also remove this model from busy set so it can pick up new work
        if cancelled_model_id:
          with busy_models_lock:
            busy_models.discard(cancelled_model_id)

      return jsonify(
        {
          "success": True,
          "cancelled": True,
          "item_id": item_id,
          "message": f"Cancelled queue item {item_id}",
        }
      )
    else:
      print(f"ℹ️  Queue item {item_id} not found or already completed")
      return jsonify(
        {
          "success": True,
          "cancelled": False,
          "item_id": item_id,
          "message": f"Queue item {item_id} not found or already completed",
        }
      )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify(
    {
      "models": [m.to_dict() for m in APP_CONFIG.models],
      "default_model_id": APP_CONFIG.default_model_id,
    }
  )


@app.route("/api/nyc-boundary")
def api_nyc_boundary():
  """
  API endpoint to get the NYC boundary GeoJSON with coordinate transformation info.

  Returns the NYC borough boundaries along with the generation config needed
  to transform lat/lng coordinates to quadrant (x, y) coordinates.
  """
  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    # Pre-compute boundary points in quadrant coordinates for the frontend
    # This avoids complex math in JavaScript
    boundary_in_quadrants = {"type": "FeatureCollection", "features": []}

    for feature in BOUNDARY_GEOJSON["features"]:
      new_feature = {
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": {"type": feature["geometry"]["type"], "coordinates": []},
      }

      # Process each ring of the polygon
      for ring in feature["geometry"]["coordinates"]:
        new_ring = []
        for coord in ring:
          lng, lat = coord[0], coord[1]
          qx, qy = latlng_to_quadrant_coords(config, lat, lng)
          new_ring.append([qx, qy])
        new_feature["geometry"]["coordinates"].append(new_ring)

      boundary_in_quadrants["features"].append(new_feature)

    return jsonify(
      {
        "boundary": boundary_in_quadrants,
        "seed": config["seed"],
      }
    )
  finally:
    conn.close()


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/delete-render", methods=["POST"])
def api_delete_render():
  """API endpoint to delete render data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the render column (set to NULL) but keep the row
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted render data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/flag", methods=["POST"])
def api_flag():
  """API endpoint to flag/unflag selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get flag value (default to True/1 for flagging, False/0 for unflagging)
  flag_value = 1 if data.get("flag", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the flagged column exists
    ensure_flagged_column_exists(conn)

    flagged_count = 0
    for qx, qy in quadrants:
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET flagged = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (flag_value, qx, qy),
      )
      if cursor.rowcount > 0:
        flagged_count += 1

    conn.commit()

    action = "Flagged" if flag_value else "Unflagged"
    return jsonify(
      {
        "success": True,
        "message": f"{action} {flagged_count} quadrant{'s' if flagged_count != 1 else ''}",
        "count": flagged_count,
        "flagged": bool(flag_value),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/water", methods=["POST"])
def api_water():
  """
  API endpoint to mark/unmark selected quadrants as water tiles.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed by script)
     1: Water tile

  Request body:
    {
      "quadrants": [[x, y], ...],
      "is_water": true/false,  // true=water(1), false=not water(0)
      "explicit_not_water": true  // Optional: if true, sets to -1 (protected)
    }
  """
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Determine water value:
  # - explicit_not_water=true → -1 (protected from auto-detection)
  # - is_water=true → 1 (water)
  # - is_water=false → 0 (not water, can be auto-changed)
  if data.get("explicit_not_water", False):
    water_value = -1
    action = "Marked as explicitly NOT water (protected)"
  elif data.get("is_water", True):
    water_value = 1
    action = "Marked as water"
  else:
    water_value = 0
    action = "Unmarked as water"

  conn = get_db_connection()

  try:
    # Ensure the is_water column exists
    ensure_is_water_column_exists(conn)

    water_count = 0
    for qx, qy in quadrants:
      # First ensure the quadrant exists in the database
      cursor = conn.execute(
        "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (qx, qy),
      )
      if cursor.fetchone() is None:
        # Quadrant doesn't exist, skip it
        continue

      cursor = conn.execute(
        """
        UPDATE quadrants
        SET is_water = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (water_value, qx, qy),
      )
      if cursor.rowcount > 0:
        water_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "water_status": water_value,
        "is_water": water_value == 1,
        "is_explicit_not_water": water_value == -1,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  print(f"\n{'=' * 60}")
  print(f"🎨 Render request: {selected_quadrants}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "render")
  return jsonify(result), 202  # 202 Accepted


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """
  API endpoint to generate tiles for selected quadrants.

  Request body:
    {
      "quadrants": [[x, y], ...] or [{"x": x, "y": y}, ...],
      "model_id": "optional-model-id",
      "context": [[x, y], ...] or [{"x": x, "y": y}, ...]  // Optional context quadrants
    }

  The context quadrants are used to provide surrounding pixel art context for
  the generation. If a context quadrant has an existing generation, that will
  be used; otherwise the render content will be used.
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Get optional model_id from request
  model_id = data.get("model_id")

  # Convert quadrants to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Parse optional context quadrants
  context_quadrants = None
  context_raw = data.get("context")
  if context_raw:
    if not isinstance(context_raw, list):
      return jsonify(
        {
          "success": False,
          "error": "context must be a list of quadrant coordinates",
        }
      ), 400

    context_quadrants = []
    for q in context_raw:
      if isinstance(q, list) and len(q) == 2:
        context_quadrants.append((int(q[0]), int(q[1])))
      elif isinstance(q, dict) and "x" in q and "y" in q:
        context_quadrants.append((int(q["x"]), int(q["y"])))
      else:
        return jsonify(
          {
            "success": False,
            "error": f"Invalid context quadrant format: {q}",
          }
        ), 400

  # Parse optional prompt
  prompt = data.get("prompt")
  if prompt and not isinstance(prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "prompt must be a string",
      }
    ), 400

  # Clean up prompt (strip whitespace, None if empty)
  if prompt:
    prompt = prompt.strip()
    if not prompt:
      prompt = None

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt
  )
  return jsonify(result), 202  # 202 Accepted


# =============================================================================
# Rectangle Generation API
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def load_queued_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """
  Load all quadrants from pending and processing queue items.

  These quadrants are scheduled for generation and should be considered
  when planning new rectangles to avoid seam issues.
  """
  queued: set[Point] = set()

  # Get pending items
  pending_items = get_pending_queue(conn)
  for item in pending_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  # Get processing items
  processing_items = get_all_processing_items(conn)
  for item in processing_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  return queued


@app.route("/api/generate-rectangle", methods=["POST"])
def api_generate_rectangle():
  """
  API endpoint to generate all quadrants within a rectangle.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "model_id": "optional-model-id"    // Optional model ID
    }

  Returns:
    {
      "success": true,
      "plan_summary": {...},
      "queued_count": N,
      "message": "Queued N generation steps"
    }
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl = Point(int(tl_raw[0]), int(tl_raw[1]))
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl = Point(int(tl_raw["x"]), int(tl_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br = Point(int(br_raw[0]), int(br_raw[1]))
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br = Point(int(br_raw["x"]), int(br_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  # Get optional model_id
  model_id = data.get("model_id")

  print(f"\n{'=' * 60}")
  print(f"📐 Rectangle generation request: ({tl.x},{tl.y}) to ({br.x},{br.y})")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Load existing generated quadrants and pending/processing quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)
    queued = load_queued_quadrants(conn)

    if queued:
      print(f"   Considering {len(queued)} queued/processing quadrant(s) for seam avoidance")

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated, queued)

    # Validate the plan
    is_valid, errors = validate_plan(plan)
    if not is_valid:
      print(f"❌ Invalid plan generated: {errors}")
      return jsonify(
        {
          "success": False,
          "error": f"Internal error: invalid plan generated - {errors}",
        }
      ), 500

    # Get plan summary for response
    summary = get_plan_summary(plan)

    if len(plan.steps) == 0:
      print("ℹ️  No quadrants to generate (all already generated)")
      return jsonify(
        {
          "success": True,
          "plan_summary": summary,
          "queued_count": 0,
          "message": "No quadrants to generate - all already generated",
        }
      )

    # Queue all generation steps
    queued_count = 0
    for step in plan.steps:
      quadrants = [(q.x, q.y) for q in step.quadrants]
      add_to_queue(conn, QueueItemType.GENERATE, quadrants, model_id)
      queued_count += 1

    # Ensure queue worker is running
    start_queue_worker()

    print(f"✅ Queued {queued_count} generation step(s)")
    print(f"   Steps by type: {summary['steps_by_type']}")

    return jsonify(
      {
        "success": True,
        "plan_summary": summary,
        "queued_count": queued_count,
        "message": f"Queued {queued_count} generation step(s) for {summary['total_quadrants']} quadrant(s)",
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG, BOUNDARY_GEOJSON

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
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to app_config.json (default: looks in the e2e_generation directory)",
  )
  parser.add_argument(
    "--bounds",
    type=Path,
    default=None,
    help="Path to custom bounds GeoJSON file (default: NYC boundary)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Load app configuration
  APP_CONFIG = load_app_config(args.config)
  print(f"📦 Loaded {len(APP_CONFIG.models)} model(s) from configuration")
  for model in APP_CONFIG.models:
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Load boundary GeoJSON
  bounds_path = args.bounds.resolve() if args.bounds else None
  BOUNDARY_GEOJSON = load_bounds(bounds_path)
  bounds_name = bounds_path.name if bounds_path else "NYC (default)"
  print(f"📍 Boundary: {bounds_name}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Reset any items that were mid-processing when server shut down
    # These will be retried automatically
    reset_count = reset_all_processing_items(conn)
    if reset_count > 0:
      print(f"🔄 Reset {reset_count} interrupted generation(s) - will be retried")
    # Clean up old completed items
    deleted_count = clear_completed_items(conn)
    if deleted_count > 0:
      print(f"🧹 Cleaned up {deleted_count} old completed queue item(s)")
  finally:
    conn.close()

  # Start the queue worker
  start_queue_worker()

  # Start the global web renderer
  try:
    start_global_renderer(port=WEB_SERVER_PORT)
  except Exception as e:
    print(f"⚠️  Failed to start web renderer: {e}")
    print("   Rendering will start on demand")

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web renderer
    print("🛑 Stopping web renderer...")
    stop_global_renderer()

  return 0


if __name__ == "__main__":
  exit(main())

```

```python
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
  R          - Toggle render/generation mode
  G          - Generate selected quadrants
  S          - Toggle select tool
"""

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from isometric_nyc.e2e_generation.generate_omni import run_generation_for_quadrants
from isometric_nyc.e2e_generation.make_rectangle_plan import (
  Point,
  RectBounds,
  create_rectangle_plan,
  get_plan_summary,
  validate_plan,
)
from isometric_nyc.e2e_generation.model_config import AppConfig, load_app_config
from isometric_nyc.e2e_generation.queue_db import (
  QueueItemType,
  add_to_queue,
  cancel_processing_items,
  cancel_queue_item_by_id,
  clear_completed_items,
  clear_pending_queue,
  get_all_processing_items,
  get_next_pending_item_for_available_model,
  get_pending_queue,
  get_queue_position_for_model,
  get_queue_status,
  get_queue_status_by_model,
  init_queue_table,
  mark_item_complete,
  mark_item_error,
  mark_item_processing,
  reset_all_processing_items,
)
from isometric_nyc.e2e_generation.replace_color import hex_to_rgb
from isometric_nyc.e2e_generation.replace_color import (
  process_quadrant as process_color_replacement,
)
from isometric_nyc.e2e_generation.bounds import load_bounds
from isometric_nyc.e2e_generation.shared import (
  DEFAULT_WEB_PORT,
  get_generation_config,
  latlng_to_quadrant_coords,
)
from isometric_nyc.e2e_generation.web_renderer import (
  start_global_renderer,
  stop_global_renderer,
)

# Global boundary GeoJSON - loaded at startup
BOUNDARY_GEOJSON: dict | None = None

# Load environment variables
load_dotenv()

# Setup Flask with template and static folders relative to this file
VIEWER_DIR = Path(__file__).parent
app = Flask(
  __name__,
  template_folder=str(VIEWER_DIR / "templates"),
  static_folder=str(VIEWER_DIR / "static"),
)


# =============================================================================
# Logging Configuration - Suppress noisy tile request logs
# =============================================================================
class TileRequestFilter(logging.Filter):
  """Filter out noisy tile and static file requests from logs."""

  def filter(self, record: logging.LogRecord) -> bool:
    message = record.getMessage()
    # Filter out tile requests, static files, and api/status polling
    if "/tile/" in message:
      return False
    if "/static/" in message:
      return False
    if "/api/status" in message:
      return False
    return True


# Apply filter to werkzeug logger (Flask's HTTP request logger)
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(TileRequestFilter())

# Generation lock - protects generation_state updates
generation_lock = threading.Lock()

# Per-model generation states
# Key is model_id (None for default), value is state dict
model_generation_states: dict[str | None, dict] = {}

# Legacy global generation_state for backwards compatibility with API
generation_state = {
  "is_generating": False,
  "quadrants": [],  # List of quadrant coords being generated
  "status": "idle",  # idle, validating, rendering, uploading, generating, saving, complete, error
  "message": "",
  "error": None,
  "started_at": None,
  "current_item_id": None,
  "model_id": None,
}

# Track which models are currently processing
busy_models: set[str | None] = set()
busy_models_lock = threading.Lock()

# Queue worker thread
queue_worker_thread: threading.Thread | None = None
queue_worker_running = False

# Cancellation flag - set to True to cancel all generations
generation_cancelled = False

# Will be set by main()
GENERATION_DIR: Path | None = None
WEB_SERVER_PORT: int = DEFAULT_WEB_PORT
APP_CONFIG: AppConfig | None = None


def get_db_connection() -> sqlite3.Connection:
  """Get a connection to the quadrants database."""
  if GENERATION_DIR is None:
    raise RuntimeError("GENERATION_DIR not set")
  db_path = GENERATION_DIR / "quadrants.db"
  return sqlite3.connect(db_path)


def ensure_flagged_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the flagged column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "flagged" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN flagged INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'flagged' column to quadrants table")


def ensure_is_water_column_exists(conn: sqlite3.Connection) -> None:
  """Ensure the is_water column exists in the quadrants table (migration)."""
  cursor = conn.cursor()
  # Check if column exists
  cursor.execute("PRAGMA table_info(quadrants)")
  columns = [row[1] for row in cursor.fetchall()]
  if "is_water" not in columns:
    cursor.execute("ALTER TABLE quadrants ADD COLUMN is_water INTEGER DEFAULT 0")
    conn.commit()
    print("📝 Added 'is_water' column to quadrants table")


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


def get_quadrant_info(x: int, y: int, use_render: bool = False) -> dict:
  """
  Get info about a quadrant including whether it has data, is flagged, and water status.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed)
     1: Water tile
  """
  conn = get_db_connection()
  try:
    # Ensure columns exist
    ensure_flagged_column_exists(conn)
    ensure_is_water_column_exists(conn)

    cursor = conn.cursor()
    column = "render" if use_render else "generation"
    cursor.execute(
      f"""
      SELECT {column} IS NOT NULL, COALESCE(flagged, 0), COALESCE(is_water, 0)
      FROM quadrants
      WHERE quadrant_x = ? AND quadrant_y = ?
      """,
      (x, y),
    )
    row = cursor.fetchone()
    if row:
      water_status = row[2]
      return {
        "has_data": bool(row[0]),
        "flagged": bool(row[1]),
        "is_water": water_status == 1,  # True if water
        "is_explicit_not_water": water_status == -1,  # True if explicitly not water
        "water_status": water_status,  # Raw value: -1, 0, or 1
      }
    return {
      "has_data": False,
      "flagged": False,
      "is_water": False,
      "is_explicit_not_water": False,
      "water_status": 0,
    }
  finally:
    conn.close()


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

  # Check which tiles have data, flagged status, and water status
  tiles = {}
  flagged_tiles = {}
  water_tiles = {}
  explicit_not_water_tiles = {}
  for dx in range(nx):
    for dy in range(ny):
      qx, qy = x + dx, y + dy
      info = get_quadrant_info(qx, qy, use_render=show_render)
      tiles[(dx, dy)] = info["has_data"]
      flagged_tiles[(dx, dy)] = info["flagged"]
      water_tiles[(dx, dy)] = info["is_water"]
      explicit_not_water_tiles[(dx, dy)] = info["is_explicit_not_water"]

  # Get model configuration for the frontend
  models_config = []
  default_model_id = None
  if APP_CONFIG:
    models_config = [m.to_dict() for m in APP_CONFIG.models]
    default_model_id = APP_CONFIG.default_model_id

  return render_template(
    "viewer.html",
    x=x,
    y=y,
    nx=nx,
    ny=ny,
    size_px=size_px,
    show_lines=show_lines,
    show_coords=show_coords,
    show_render=show_render,
    tiles=tiles,
    flagged_tiles=flagged_tiles,
    water_tiles=water_tiles,
    explicit_not_water_tiles=explicit_not_water_tiles,
    generation_dir=str(GENERATION_DIR),
    models_config=json.dumps(models_config),
    default_model_id=default_model_id,
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

  # Generate ETag from content hash for caching
  etag = hashlib.md5(data).hexdigest()

  # Check if client has cached version
  if_none_match = request.headers.get("If-None-Match")
  if if_none_match and if_none_match == etag:
    return Response(status=304)  # Not Modified

  response = Response(data, mimetype="image/png")
  response.headers["ETag"] = etag
  response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
  return response


# =============================================================================
# Generation API
# =============================================================================


def update_generation_state(
  status: str, message: str = "", error: str | None = None
) -> None:
  """Update the global generation state."""
  global generation_state
  generation_state["status"] = status
  generation_state["message"] = message
  if error:
    generation_state["error"] = error


def calculate_context_quadrants(
  conn: sqlite3.Connection,
  selected_quadrants: list[tuple[int, int]],
) -> list[tuple[int, int]]:
  """
  Calculate context quadrants lazily at execution time.

  This determines which adjacent quadrants have existing generations
  that can provide context for the current generation.

  For a valid generation, we need at least a 2x2 block where all 4 quadrants
  are either being generated or already generated.

  Args:
    conn: Database connection
    selected_quadrants: The quadrants being generated

  Returns:
    List of quadrant coordinates that have existing generations and can
    provide context for the current generation.
  """
  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation as shared_get_quadrant_generation,
  )

  selected_set = set(selected_quadrants)
  context = []

  # Find all quadrants adjacent to the selection that have generations
  # Check all potential 2x2 blocks that include any selected quadrant
  checked = set()

  for qx, qy in selected_quadrants:
    # Check all neighbors that could form a 2x2 block with this quadrant
    # A quadrant can be in 4 different 2x2 blocks (as TL, TR, BL, BR corner)
    potential_context = [
      # Neighbors for 2x2 where (qx, qy) is top-left
      (qx + 1, qy),
      (qx, qy + 1),
      (qx + 1, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is top-right
      (qx - 1, qy),
      (qx - 1, qy + 1),
      (qx, qy + 1),
      # Neighbors for 2x2 where (qx, qy) is bottom-left
      (qx, qy - 1),
      (qx + 1, qy - 1),
      (qx + 1, qy),
      # Neighbors for 2x2 where (qx, qy) is bottom-right
      (qx - 1, qy - 1),
      (qx, qy - 1),
      (qx - 1, qy),
    ]

    for nx, ny in potential_context:
      coord = (nx, ny)
      if coord in checked or coord in selected_set:
        continue
      checked.add(coord)

      # Check if this quadrant has an existing generation
      gen = shared_get_quadrant_generation(conn, nx, ny)
      if gen is not None:
        context.append(coord)

  return context


def run_generation(
  conn: sqlite3.Connection,
  config: dict,
  selected_quadrants: list[tuple[int, int]],
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """
  Run the full generation pipeline for selected quadrants.

  This is a wrapper around run_generation_for_quadrants that ensures
  the web server is running and updates the global generation state.

  Args:
    conn: Database connection
    config: Generation config dict
    selected_quadrants: List of (x, y) quadrant coordinates to generate
    model_id: Optional model ID for generation
    context_quadrants: Optional list of (x, y) quadrant coordinates to use as
      context. These quadrants provide surrounding pixel art context for the
      generation.
    prompt: Optional additional prompt text for generation

  Returns dict with success status and message/error.
  """
  # Get model configuration if specified
  model_config = None
  if model_id and APP_CONFIG:
    model_config = APP_CONFIG.get_model(model_id)
  elif APP_CONFIG:
    model_config = APP_CONFIG.get_default_model()

  # Create status callback that updates global state
  def status_callback(status: str, message: str) -> None:
    update_generation_state(status, message)

  # Use the shared library function
  return run_generation_for_quadrants(
    conn=conn,
    config=config,
    selected_quadrants=selected_quadrants,
    port=WEB_SERVER_PORT,
    status_callback=status_callback,
    model_config=model_config,
    context_quadrants=context_quadrants,
    prompt=prompt,
  )


def render_quadrant_with_renderer(
  conn: sqlite3.Connection,
  config: dict,
  x: int,
  y: int,
) -> bytes | None:
  """
  Render a quadrant using the global web renderer.

  Returns the PNG bytes of the rendered quadrant.
  """
  from isometric_nyc.e2e_generation.shared import (
    ensure_quadrant_exists,
    save_quadrant_render,
  )
  from isometric_nyc.e2e_generation.web_renderer import get_web_renderer

  # Ensure the quadrant exists in the database
  quadrant = ensure_quadrant_exists(conn, config, x, y)

  print(f"   🎨 Rendering tile for quadrant ({x}, {y})...")

  renderer = get_web_renderer(port=WEB_SERVER_PORT)

  # Render the tile
  quadrant_images = renderer.render_quadrant(
    quadrant_x=x,
    quadrant_y=y,
    lat=quadrant["lat"],
    lng=quadrant["lng"],
    width_px=config["width_px"],
    height_px=config["height_px"],
    camera_azimuth_degrees=config["camera_azimuth_degrees"],
    camera_elevation_degrees=config["camera_elevation_degrees"],
    view_height_meters=config.get("view_height_meters", 200),
  )

  # Save all quadrants to database
  result_bytes = None
  for (dx, dy), png_bytes in quadrant_images.items():
    qx, qy = x + dx, y + dy
    save_quadrant_render(conn, config, qx, qy, png_bytes)
    print(f"      ✓ Saved render for ({qx}, {qy})")

    # Return the specific quadrant we were asked for
    if qx == x and qy == y:
      result_bytes = png_bytes

  return result_bytes


def process_queue_item_from_db(item_id: int) -> dict:
  """Process a single queue item from the database."""
  global generation_state

  conn = get_db_connection()
  try:
    # Get item details
    cursor = conn.cursor()
    cursor.execute(
      """
      SELECT item_type, quadrants, model_id, context_quadrants, prompt
      FROM generation_queue
      WHERE id = ?
      """,
      (item_id,),
    )
    row = cursor.fetchone()
    if not row:
      return {"success": False, "error": "Item not found"}

    item_type = QueueItemType(row[0])
    quadrants = json.loads(row[1])
    model_id = row[2]
    context_quadrants_raw = json.loads(row[3]) if row[3] else None
    prompt = row[4]

    # Convert to list of tuples
    selected_quadrants = [(q[0], q[1]) for q in quadrants]

    # Calculate context lazily if not explicitly provided
    # This ensures we use the most up-to-date context based on what's
    # actually generated at execution time (not queue time)
    if context_quadrants_raw:
      context_quadrants = [(q[0], q[1]) for q in context_quadrants_raw]
      print(
        f"   📋 Using explicit context from queue: {len(context_quadrants)} quadrant(s)"
      )
    else:
      # Calculate context lazily based on current generation state
      context_quadrants = calculate_context_quadrants(conn, selected_quadrants)
      if context_quadrants:
        print(f"   📋 Calculated lazy context: {len(context_quadrants)} quadrant(s)")
      else:
        print(
          "   📋 No context quadrants (2x2 self-contained or no adjacent generations)"
        )

    # Mark item as processing
    mark_item_processing(conn, item_id)

    # Initialize generation state
    generation_state["is_generating"] = True
    generation_state["quadrants"] = selected_quadrants
    generation_state["status"] = (
      "starting" if item_type == QueueItemType.GENERATE else "rendering"
    )
    generation_state["message"] = f"Starting {item_type.value}..."
    generation_state["error"] = None
    generation_state["started_at"] = time.time()
    generation_state["current_item_id"] = item_id
    generation_state["model_id"] = model_id

    print(f"\n{'=' * 60}")
    emoji = "🎯" if item_type == QueueItemType.GENERATE else "🎨"
    print(
      f"{emoji} {item_type.value.title()} request (item {item_id}): {selected_quadrants}"
    )
    if model_id:
      print(f"   Model: {model_id}")
    if context_quadrants:
      print(f"   Context: {context_quadrants}")
    if prompt:
      print(f"   Prompt: {prompt}")
    print(f"{'=' * 60}")

    config = get_generation_config(conn)

    if item_type == QueueItemType.GENERATE:
      # Retry logic for generation - retry up to 3 times
      max_generation_retries = 3
      generation_retry_delay = 5.0  # seconds between generation retries

      for gen_attempt in range(1, max_generation_retries + 1):
        result = run_generation(
          conn, config, selected_quadrants, model_id, context_quadrants, prompt
        )

        if result["success"]:
          print(f"✅ Generation complete: {result['message']}")
          generation_state["status"] = "complete"
          generation_state["message"] = result["message"]
          mark_item_complete(conn, item_id, result["message"])
          return result

        # Generation failed
        if gen_attempt < max_generation_retries:
          print(
            f"⚠️  Generation failed (attempt {gen_attempt}/{max_generation_retries}): "
            f"{result['error']}"
          )
          print(f"⏳ Waiting {generation_retry_delay}s before retrying generation...")
          update_generation_state(
            "retrying",
            f"Generation failed, retrying (attempt {gen_attempt + 1}/{max_generation_retries})...",
          )
          time.sleep(generation_retry_delay)
        else:
          # All retries exhausted
          print(
            f"❌ Generation failed after {max_generation_retries} attempts: "
            f"{result['error']}"
          )
          generation_state["status"] = "error"
          generation_state["error"] = result["error"]
          mark_item_error(conn, item_id, result["error"])
          return result

      # Should not reach here, but just in case
      return result

    else:  # render
      update_generation_state("rendering", "Starting render...")

      rendered_count = 0
      total = len(selected_quadrants)

      for i, (qx, qy) in enumerate(selected_quadrants):
        update_generation_state(
          "rendering", f"Rendering quadrant ({qx}, {qy})... ({i + 1}/{total})"
        )
        print(f"   🎨 Rendering quadrant ({qx}, {qy})...")

        try:
          render_bytes = render_quadrant_with_renderer(conn, config, qx, qy)
          if render_bytes:
            rendered_count += 1
            print(f"      ✓ Rendered quadrant ({qx}, {qy})")
          else:
            print(f"      ⚠️ No render output for ({qx}, {qy})")
        except Exception as e:
          print(f"      ❌ Failed to render ({qx}, {qy}): {e}")
          traceback.print_exc()

      result_message = f"Rendered {rendered_count} quadrant(s)"
      update_generation_state("complete", result_message)
      print(f"✅ Render complete: {rendered_count}/{total} quadrants")
      mark_item_complete(conn, item_id, result_message)

      return {
        "success": True,
        "message": f"Rendered {rendered_count} quadrant{'s' if rendered_count != 1 else ''}",
        "quadrants": selected_quadrants,
      }

  except Exception as e:
    traceback.print_exc()
    generation_state["status"] = "error"
    generation_state["error"] = str(e)
    mark_item_error(conn, item_id, str(e))
    return {"success": False, "error": str(e)}
  finally:
    conn.close()


def process_model_item(item_id: int, model_id: str | None):
  """Process a single queue item for a specific model in its own thread."""
  global generation_state, generation_cancelled

  try:
    # Check cancellation before starting
    if generation_cancelled:
      print(f"⚠️  Item {item_id} cancelled before processing")
      return

    process_queue_item_from_db(item_id)

  except Exception as e:
    print(f"❌ Model worker error for {model_id}: {e}")
    traceback.print_exc()
  finally:
    # Remove model from busy set
    with busy_models_lock:
      busy_models.discard(model_id)

    # Update global state if this was the active model
    with generation_lock:
      if generation_state.get("model_id") == model_id:
        generation_state["is_generating"] = False
        generation_state["current_item_id"] = None

    # Remove from per-model states
    if model_id in model_generation_states:
      del model_generation_states[model_id]


def queue_worker():
  """Background worker that processes the generation queue from the database.

  This worker supports parallel processing of different models - each model
  can have one active generation at a time, but different models can run
  concurrently.
  """
  global generation_state, queue_worker_running, generation_cancelled

  print("🔄 Queue worker started (parallel model support)")

  while queue_worker_running:
    conn = None
    try:
      # Check if we were cancelled
      if generation_cancelled:
        print("⚠️  Generation cancelled, resetting flags...")
        generation_cancelled = False
        with generation_lock:
          generation_state["is_generating"] = False
          generation_state["current_item_id"] = None
        with busy_models_lock:
          busy_models.clear()
        model_generation_states.clear()
        time.sleep(0.5)
        continue

      conn = get_db_connection()

      # Get current busy models
      with busy_models_lock:
        current_busy = busy_models.copy()

      # Get next pending item for an available model
      item = get_next_pending_item_for_available_model(conn, current_busy)

      if item is None:
        # No items available (either queue empty or all models busy)
        conn.close()
        time.sleep(0.5)
        continue

      item_id = item.id
      model_id = item.model_id
      conn.close()
      conn = None

      # Mark this model as busy
      with busy_models_lock:
        if model_id in busy_models:
          # Another thread grabbed this model, skip
          continue
        busy_models.add(model_id)

      # Update global state for display (use most recent)
      with generation_lock:
        generation_state["is_generating"] = True
        generation_state["model_id"] = model_id

      # Initialize per-model state
      model_generation_states[model_id] = {
        "is_generating": True,
        "item_id": item_id,
        "started_at": time.time(),
      }

      # Spawn a thread to process this model's item
      model_name = model_id or "default"
      worker_thread = threading.Thread(
        target=process_model_item,
        args=(item_id, model_id),
        name=f"model-worker-{model_name}",
        daemon=True,
      )
      worker_thread.start()

      print(f"🚀 Started worker for model '{model_name}' (item {item_id})")

      # Small delay before checking for more work
      time.sleep(0.2)

    except Exception as e:
      print(f"❌ Queue worker error: {e}")
      traceback.print_exc()
      time.sleep(1.0)
    finally:
      if conn:
        conn.close()

  print("🛑 Queue worker stopped")


def start_queue_worker():
  """Start the queue worker thread if not already running."""
  global queue_worker_thread, queue_worker_running

  if queue_worker_thread is not None and queue_worker_thread.is_alive():
    return  # Already running

  queue_worker_running = True
  queue_worker_thread = threading.Thread(target=queue_worker, daemon=True)
  queue_worker_thread.start()


def stop_queue_worker():
  """Stop the queue worker thread."""
  global queue_worker_running
  queue_worker_running = False


def add_to_queue_db(
  quadrants: list[tuple[int, int]],
  item_type: str,
  model_id: str | None = None,
  context_quadrants: list[tuple[int, int]] | None = None,
  prompt: str | None = None,
) -> dict:
  """Add a generation/render request to the database queue."""
  conn = get_db_connection()
  try:
    queue_item = add_to_queue(
      conn,
      QueueItemType(item_type),
      quadrants,
      model_id,
      context_quadrants,
      prompt,
    )

    # Get model-specific queue position
    model_position = get_queue_position_for_model(conn, queue_item.id, model_id)

    # Get total queue length for backwards compatibility
    pending = get_pending_queue(conn)
    total_position = len(pending)

    # Ensure the queue worker is running
    start_queue_worker()

    return {
      "success": True,
      "queued": True,
      "position": model_position,  # Position within this model's queue
      "total_position": total_position,  # Overall queue position
      "model_id": model_id,
      "item_id": queue_item.id,
      "message": f"Added to queue at position {model_position}",
    }
  finally:
    conn.close()


@app.route("/api/status")
def api_status():
  """API endpoint to check generation status including queue info."""
  conn = get_db_connection()
  try:
    queue_status = get_queue_status(conn)
    model_status = get_queue_status_by_model(conn)

    # Get list of currently busy models
    with busy_models_lock:
      active_models = list(busy_models)

    # Build the response
    response = {
      **generation_state,
      "queue": queue_status["pending_items"],
      "queue_length": queue_status["pending_count"],
      # Per-model queue info
      "queue_by_model": model_status["by_model"],
      "processing_models": model_status["processing_models"],
      # All currently processing models (for parallel processing)
      "active_models": active_models,
      "active_model_count": len(active_models),
      # All quadrants being processed across all models
      "all_processing_quadrants": model_status["all_processing_quadrants"],
    }

    # Set is_generating based on whether any models are active
    response["is_generating"] = len(active_models) > 0

    # Include current processing item info if available
    if queue_status["current_item"]:
      response["current_item"] = queue_status["current_item"]

    return jsonify(response)
  finally:
    conn.close()


@app.route("/api/queue/clear", methods=["POST"])
def api_clear_queue():
  """
  API endpoint to clear all items from the generation queue,
  including cancelling any currently running generation.

  Returns:
    {
      "success": true,
      "cleared_count": N,
      "cancelled_count": M,
      "message": "Cleared N pending item(s), cancelled M in-progress item(s)"
    }
  """
  global generation_state, generation_cancelled

  print(f"\n{'=' * 60}")
  print("🗑️  Clear queue request received")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # Clear pending items
    cleared_count = clear_pending_queue(conn)

    # Cancel any in-progress items
    cancelled_count = cancel_processing_items(conn)

    # Set the cancellation flag so the worker knows to stop
    if cancelled_count > 0:
      generation_cancelled = True

    # Clear busy models
    with busy_models_lock:
      busy_models.clear()

    # Clear per-model states
    model_generation_states.clear()

    # Reset the generation state
    generation_state["is_generating"] = False
    generation_state["quadrants"] = []
    generation_state["status"] = "idle"
    generation_state["message"] = "Queue cleared"
    generation_state["error"] = None
    generation_state["current_item_id"] = None

    if cleared_count > 0 or cancelled_count > 0:
      print(
        f"✅ Cleared {cleared_count} pending, cancelled {cancelled_count} in-progress"
      )
    else:
      print("ℹ️  Queue was already empty")

    message_parts = []
    if cleared_count > 0:
      message_parts.append(f"Cleared {cleared_count} pending item(s)")
    if cancelled_count > 0:
      message_parts.append(f"cancelled {cancelled_count} in-progress item(s)")
    message = ", ".join(message_parts) if message_parts else "Queue was already empty"

    return jsonify(
      {
        "success": True,
        "cleared_count": cleared_count,
        "cancelled_count": cancelled_count,
        "message": message,
      }
    )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/queue/cancel/<int:item_id>", methods=["POST"])
def api_cancel_queue_item(item_id: int):
  """
  API endpoint to cancel a specific queue item by its ID.

  Returns:
    {
      "success": true,
      "cancelled": true,
      "item_id": N,
      "message": "Cancelled queue item N"
    }
  """
  global generation_cancelled

  print(f"\n{'=' * 60}")
  print(f"🗑️  Cancel queue item request received: item_id={item_id}")
  print(f"{'=' * 60}")

  conn = get_db_connection()
  try:
    # First check if this item was processing (not just pending)
    cursor = conn.cursor()
    cursor.execute(
      "SELECT status, model_id FROM generation_queue WHERE id = ?",
      (item_id,),
    )
    row = cursor.fetchone()
    was_processing = row and row[0] == "processing"
    cancelled_model_id = row[1] if row else None

    cancelled = cancel_queue_item_by_id(conn, item_id)

    if cancelled:
      print(f"✅ Cancelled queue item {item_id}")

      # Only set the global cancellation flag if this was a PROCESSING item
      # Pending items just get marked as cancelled in the database
      if was_processing:
        generation_cancelled = True
        print("   ⚠️  Item was processing, signaling cancellation")

        # Also remove this model from busy set so it can pick up new work
        if cancelled_model_id:
          with busy_models_lock:
            busy_models.discard(cancelled_model_id)

      return jsonify(
        {
          "success": True,
          "cancelled": True,
          "item_id": item_id,
          "message": f"Cancelled queue item {item_id}",
        }
      )
    else:
      print(f"ℹ️  Queue item {item_id} not found or already completed")
      return jsonify(
        {
          "success": True,
          "cancelled": False,
          "item_id": item_id,
          "message": f"Queue item {item_id} not found or already completed",
        }
      )
  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/models")
def api_models():
  """API endpoint to get available models."""
  if APP_CONFIG is None:
    return jsonify({"models": [], "default_model_id": None})

  return jsonify(
    {
      "models": [m.to_dict() for m in APP_CONFIG.models],
      "default_model_id": APP_CONFIG.default_model_id,
    }
  )


@app.route("/api/nyc-boundary")
def api_nyc_boundary():
  """
  API endpoint to get the NYC boundary GeoJSON with coordinate transformation info.

  Returns the NYC borough boundaries along with the generation config needed
  to transform lat/lng coordinates to quadrant (x, y) coordinates.
  """
  conn = get_db_connection()
  try:
    config = get_generation_config(conn)

    # Pre-compute boundary points in quadrant coordinates for the frontend
    # This avoids complex math in JavaScript
    boundary_in_quadrants = {"type": "FeatureCollection", "features": []}

    for feature in BOUNDARY_GEOJSON["features"]:
      new_feature = {
        "type": "Feature",
        "properties": feature["properties"],
        "geometry": {"type": feature["geometry"]["type"], "coordinates": []},
      }

      # Process each ring of the polygon
      for ring in feature["geometry"]["coordinates"]:
        new_ring = []
        for coord in ring:
          lng, lat = coord[0], coord[1]
          qx, qy = latlng_to_quadrant_coords(config, lat, lng)
          new_ring.append([qx, qy])
        new_feature["geometry"]["coordinates"].append(new_ring)

      boundary_in_quadrants["features"].append(new_feature)

    return jsonify(
      {
        "boundary": boundary_in_quadrants,
        "seed": config["seed"],
      }
    )
  finally:
    conn.close()


@app.route("/api/delete", methods=["POST"])
def api_delete():
  """API endpoint to delete generation data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the generation column (set to NULL) but keep the row
      # Columns are quadrant_x and quadrant_y
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET generation = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted generation data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/delete-render", methods=["POST"])
def api_delete_render():
  """API endpoint to delete render data for selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Connect to database (quadrants.db, not tiles.db)
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    deleted_count = 0
    for qx, qy in quadrants:
      # Clear the render column (set to NULL) but keep the row
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET render = NULL
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (qx, qy),
      )
      if cursor.rowcount > 0:
        deleted_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"Deleted render data for {deleted_count} quadrant{'s' if deleted_count != 1 else ''}",
        "deleted": deleted_count,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/flag", methods=["POST"])
def api_flag():
  """API endpoint to flag/unflag selected quadrants."""
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Get flag value (default to True/1 for flagging, False/0 for unflagging)
  flag_value = 1 if data.get("flag", True) else 0

  conn = get_db_connection()

  try:
    # Ensure the flagged column exists
    ensure_flagged_column_exists(conn)

    flagged_count = 0
    for qx, qy in quadrants:
      cursor = conn.execute(
        """
        UPDATE quadrants
        SET flagged = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (flag_value, qx, qy),
      )
      if cursor.rowcount > 0:
        flagged_count += 1

    conn.commit()

    action = "Flagged" if flag_value else "Unflagged"
    return jsonify(
      {
        "success": True,
        "message": f"{action} {flagged_count} quadrant{'s' if flagged_count != 1 else ''}",
        "count": flagged_count,
        "flagged": bool(flag_value),
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


@app.route("/api/water", methods=["POST"])
def api_water():
  """
  API endpoint to mark/unmark selected quadrants as water tiles.

  Water status values:
    -1: Explicitly NOT water (protected from auto-detection)
     0: Not water (auto-detected, can be changed by script)
     1: Water tile

  Request body:
    {
      "quadrants": [[x, y], ...],
      "is_water": true/false,  // true=water(1), false=not water(0)
      "explicit_not_water": true  // Optional: if true, sets to -1 (protected)
    }
  """
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify({"success": False, "error": "No quadrants specified"})

  quadrants = data["quadrants"]
  if not quadrants:
    return jsonify({"success": False, "error": "Empty quadrants list"})

  # Determine water value:
  # - explicit_not_water=true → -1 (protected from auto-detection)
  # - is_water=true → 1 (water)
  # - is_water=false → 0 (not water, can be auto-changed)
  if data.get("explicit_not_water", False):
    water_value = -1
    action = "Marked as explicitly NOT water (protected)"
  elif data.get("is_water", True):
    water_value = 1
    action = "Marked as water"
  else:
    water_value = 0
    action = "Unmarked as water"

  conn = get_db_connection()

  try:
    # Ensure the is_water column exists
    ensure_is_water_column_exists(conn)

    water_count = 0
    for qx, qy in quadrants:
      # First ensure the quadrant exists in the database
      cursor = conn.execute(
        "SELECT 1 FROM quadrants WHERE quadrant_x = ? AND quadrant_y = ?",
        (qx, qy),
      )
      if cursor.fetchone() is None:
        # Quadrant doesn't exist, skip it
        continue

      cursor = conn.execute(
        """
        UPDATE quadrants
        SET is_water = ?
        WHERE quadrant_x = ? AND quadrant_y = ?
        """,
        (water_value, qx, qy),
      )
      if cursor.rowcount > 0:
        water_count += 1

    conn.commit()

    return jsonify(
      {
        "success": True,
        "message": f"{action}: {water_count} quadrant{'s' if water_count != 1 else ''}",
        "count": water_count,
        "water_status": water_value,
        "is_water": water_value == 1,
        "is_explicit_not_water": water_value == -1,
      }
    )
  except Exception as e:
    return jsonify({"success": False, "error": str(e)})
  finally:
    conn.close()


# Hardcoded water replacement color
WATER_REPLACEMENT_COLOR = "#4A6372"
DEFAULT_SOFTNESS = 30.0  # Lower = more precise color matching


@app.route("/api/fix-water", methods=["POST"])
def api_fix_water():
  """API endpoint to fix water color in a quadrant using soft blending."""
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  # Required fields
  x = data.get("x")
  y = data.get("y")
  target_color = data.get("target_color")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  if not target_color:
    return jsonify({"success": False, "error": "Missing target_color"}), 400

  # Optional fields
  softness = data.get("softness", DEFAULT_SOFTNESS)

  # Parse colors
  try:
    target_rgb = hex_to_rgb(target_color)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid target color: {e}"}), 400

  try:
    replacement_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)
  except ValueError as e:
    return jsonify({"success": False, "error": f"Invalid replacement color: {e}"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fix request: quadrant ({x}, {y})")
  print(f"   Target color: {target_color} -> RGB{target_rgb}")
  print(f"   Replacement color: {WATER_REPLACEMENT_COLOR} -> RGB{replacement_rgb}")
  print(f"   Softness: {softness}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)
    exports_dir = Path(GENERATION_DIR) / "exports"

    success = process_color_replacement(
      conn=conn,
      config=config,
      x=int(x),
      y=int(y),
      target_color=target_rgb,
      replacement_color=replacement_rgb,
      softness=float(softness),
      dry_run=False,  # Apply directly to database
      exports_dir=exports_dir,
    )

    if success:
      print(f"✅ Water fix complete for quadrant ({x}, {y})")
      return jsonify(
        {
          "success": True,
          "message": f"Fixed water color in quadrant ({x}, {y})",
          "quadrant": {"x": x, "y": y},
          "target_color": target_color,
          "replacement_color": WATER_REPLACEMENT_COLOR,
        }
      )
    else:
      print(f"❌ Water fix failed for quadrant ({x}, {y})")
      return jsonify(
        {"success": False, "error": f"Failed to process quadrant ({x}, {y})"}
      ), 400

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/water-fill", methods=["POST"])
def api_water_fill():
  """API endpoint to fill an entire quadrant with the water color."""

  from PIL import Image

  from isometric_nyc.e2e_generation.shared import (
    get_quadrant_generation,
    image_to_png_bytes,
    png_bytes_to_image,
    save_quadrant_generation,
  )

  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No data provided"}), 400

  x = data.get("x")
  y = data.get("y")

  if x is None or y is None:
    return jsonify({"success": False, "error": "Missing x or y coordinate"}), 400

  print(f"\n{'=' * 60}")
  print(f"💧 Water fill request: quadrant ({x}, {y})")
  print(f"   Fill color: {WATER_REPLACEMENT_COLOR}")
  print(f"{'=' * 60}")

  # Connect to database
  db_path = Path(GENERATION_DIR) / "quadrants.db"
  conn = sqlite3.connect(db_path)

  try:
    config = get_generation_config(conn)

    # Get existing generation to determine size, or use config defaults
    generation_bytes = get_quadrant_generation(conn, int(x), int(y))
    if generation_bytes is not None:
      # Get dimensions from existing image
      existing_img = png_bytes_to_image(generation_bytes)
      width, height = existing_img.size
      mode = existing_img.mode
    else:
      # No existing generation - use quadrant size from config
      # Quadrant is half the tile size
      width = config.get("width_px", 512) // 2
      height = config.get("height_px", 512) // 2
      mode = "RGBA"
      print(f"   No existing generation - creating new {width}x{height} image")

    # Parse water color
    water_rgb = hex_to_rgb(WATER_REPLACEMENT_COLOR)

    # Create solid color image
    if mode == "RGBA":
      fill_color = (*water_rgb, 255)  # Add full alpha
    else:
      fill_color = water_rgb

    filled_img = Image.new(mode, (width, height), fill_color)

    # Save to database
    png_bytes = image_to_png_bytes(filled_img)
    save_quadrant_generation(conn, config, int(x), int(y), png_bytes)

    print(f"✅ Water fill complete for quadrant ({x}, {y})")
    return jsonify(
      {
        "success": True,
        "message": f"Filled quadrant ({x}, {y}) with water color",
        "quadrant": {"x": x, "y": y},
        "color": WATER_REPLACEMENT_COLOR,
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


@app.route("/api/render", methods=["POST"])
def api_render():
  """API endpoint to render tiles for selected quadrants."""
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Convert to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  print(f"\n{'=' * 60}")
  print(f"🎨 Render request: {selected_quadrants}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(selected_quadrants, "render")
  return jsonify(result), 202  # 202 Accepted


@app.route("/api/generate", methods=["POST"])
def api_generate():
  """
  API endpoint to generate tiles for selected quadrants.

  Request body:
    {
      "quadrants": [[x, y], ...] or [{"x": x, "y": y}, ...],
      "model_id": "optional-model-id",
      "context": [[x, y], ...] or [{"x": x, "y": y}, ...]  // Optional context quadrants
    }

  The context quadrants are used to provide surrounding pixel art context for
  the generation. If a context quadrant has an existing generation, that will
  be used; otherwise the render content will be used.
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data or "quadrants" not in data:
    return jsonify(
      {
        "success": False,
        "error": "Missing 'quadrants' in request body",
      }
    ), 400

  quadrants = data["quadrants"]
  if not isinstance(quadrants, list) or len(quadrants) == 0:
    return jsonify(
      {
        "success": False,
        "error": "quadrants must be a non-empty list",
      }
    ), 400

  # Get optional model_id from request
  model_id = data.get("model_id")

  # Convert quadrants to list of tuples
  selected_quadrants = []
  for q in quadrants:
    if isinstance(q, list) and len(q) == 2:
      selected_quadrants.append((int(q[0]), int(q[1])))
    elif isinstance(q, dict) and "x" in q and "y" in q:
      selected_quadrants.append((int(q["x"]), int(q["y"])))
    else:
      return jsonify(
        {
          "success": False,
          "error": f"Invalid quadrant format: {q}",
        }
      ), 400

  # Parse optional context quadrants
  context_quadrants = None
  context_raw = data.get("context")
  if context_raw:
    if not isinstance(context_raw, list):
      return jsonify(
        {
          "success": False,
          "error": "context must be a list of quadrant coordinates",
        }
      ), 400

    context_quadrants = []
    for q in context_raw:
      if isinstance(q, list) and len(q) == 2:
        context_quadrants.append((int(q[0]), int(q[1])))
      elif isinstance(q, dict) and "x" in q and "y" in q:
        context_quadrants.append((int(q["x"]), int(q["y"])))
      else:
        return jsonify(
          {
            "success": False,
            "error": f"Invalid context quadrant format: {q}",
          }
        ), 400

  # Parse optional prompt
  prompt = data.get("prompt")
  if prompt and not isinstance(prompt, str):
    return jsonify(
      {
        "success": False,
        "error": "prompt must be a string",
      }
    ), 400

  # Clean up prompt (strip whitespace, None if empty)
  if prompt:
    prompt = prompt.strip()
    if not prompt:
      prompt = None

  print(f"\n{'=' * 60}")
  print(f"🎯 Generation request: {selected_quadrants}")
  if model_id:
    print(f"   Model: {model_id}")
  if context_quadrants:
    print(f"   Context: {context_quadrants}")
  if prompt:
    print(f"   Prompt: {prompt}")
  print(f"{'=' * 60}")

  # Always add to queue (database-backed queue handles everything)
  result = add_to_queue_db(
    selected_quadrants, "generate", model_id, context_quadrants, prompt
  )
  return jsonify(result), 202  # 202 Accepted


# =============================================================================
# Rectangle Generation API
# =============================================================================


def load_generated_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """Load all quadrants that have generations from the database."""
  cursor = conn.cursor()
  cursor.execute(
    "SELECT quadrant_x, quadrant_y FROM quadrants WHERE generation IS NOT NULL"
  )
  return {Point(row[0], row[1]) for row in cursor.fetchall()}


def load_queued_quadrants(conn: sqlite3.Connection) -> set[Point]:
  """
  Load all quadrants from pending and processing queue items.

  These quadrants are scheduled for generation and should be considered
  when planning new rectangles to avoid seam issues.
  """
  queued: set[Point] = set()

  # Get pending items
  pending_items = get_pending_queue(conn)
  for item in pending_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  # Get processing items
  processing_items = get_all_processing_items(conn)
  for item in processing_items:
    if item.item_type == QueueItemType.GENERATE:
      for qx, qy in item.quadrants:
        queued.add(Point(qx, qy))

  return queued


@app.route("/api/generate-rectangle", methods=["POST"])
def api_generate_rectangle():
  """
  API endpoint to generate all quadrants within a rectangle.

  Request body:
    {
      "tl": [x, y] or {"x": x, "y": y},  // Top-left corner
      "br": [x, y] or {"x": x, "y": y},  // Bottom-right corner
      "model_id": "optional-model-id"    // Optional model ID
    }

  Returns:
    {
      "success": true,
      "plan_summary": {...},
      "queued_count": N,
      "message": "Queued N generation steps"
    }
  """
  global generation_state

  # Parse request
  data = request.get_json()
  if not data:
    return jsonify({"success": False, "error": "No JSON body provided"}), 400

  # Parse top-left coordinate
  tl_raw = data.get("tl")
  if not tl_raw:
    return jsonify(
      {"success": False, "error": "Missing 'tl' (top-left) coordinate"}
    ), 400

  try:
    if isinstance(tl_raw, list) and len(tl_raw) == 2:
      tl = Point(int(tl_raw[0]), int(tl_raw[1]))
    elif isinstance(tl_raw, dict) and "x" in tl_raw and "y" in tl_raw:
      tl = Point(int(tl_raw["x"]), int(tl_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'tl' format: {tl_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'tl' coordinate: {e}"}), 400

  # Parse bottom-right coordinate
  br_raw = data.get("br")
  if not br_raw:
    return jsonify(
      {"success": False, "error": "Missing 'br' (bottom-right) coordinate"}
    ), 400

  try:
    if isinstance(br_raw, list) and len(br_raw) == 2:
      br = Point(int(br_raw[0]), int(br_raw[1]))
    elif isinstance(br_raw, dict) and "x" in br_raw and "y" in br_raw:
      br = Point(int(br_raw["x"]), int(br_raw["y"]))
    else:
      return jsonify({"success": False, "error": f"Invalid 'br' format: {br_raw}"}), 400
  except (ValueError, TypeError) as e:
    return jsonify({"success": False, "error": f"Invalid 'br' coordinate: {e}"}), 400

  # Validate bounds
  if tl.x > br.x or tl.y > br.y:
    return jsonify(
      {
        "success": False,
        "error": "Invalid bounds: top-left must be above and to the left of bottom-right",
      }
    ), 400

  # Get optional model_id
  model_id = data.get("model_id")

  print(f"\n{'=' * 60}")
  print(f"📐 Rectangle generation request: ({tl.x},{tl.y}) to ({br.x},{br.y})")
  if model_id:
    print(f"   Model: {model_id}")
  print(f"{'=' * 60}")

  # Load existing generated quadrants and pending/processing quadrants
  conn = get_db_connection()
  try:
    generated = load_generated_quadrants(conn)
    queued = load_queued_quadrants(conn)

    if queued:
      print(f"   Considering {len(queued)} queued/processing quadrant(s) for seam avoidance")

    # Create the rectangle plan
    bounds = RectBounds(tl, br)
    plan = create_rectangle_plan(bounds, generated, queued)

    # Validate the plan
    is_valid, errors = validate_plan(plan)
    if not is_valid:
      print(f"❌ Invalid plan generated: {errors}")
      return jsonify(
        {
          "success": False,
          "error": f"Internal error: invalid plan generated - {errors}",
        }
      ), 500

    # Get plan summary for response
    summary = get_plan_summary(plan)

    if len(plan.steps) == 0:
      print("ℹ️  No quadrants to generate (all already generated)")
      return jsonify(
        {
          "success": True,
          "plan_summary": summary,
          "queued_count": 0,
          "message": "No quadrants to generate - all already generated",
        }
      )

    # Queue all generation steps
    queued_count = 0
    for step in plan.steps:
      quadrants = [(q.x, q.y) for q in step.quadrants]
      add_to_queue(conn, QueueItemType.GENERATE, quadrants, model_id)
      queued_count += 1

    # Ensure queue worker is running
    start_queue_worker()

    print(f"✅ Queued {queued_count} generation step(s)")
    print(f"   Steps by type: {summary['steps_by_type']}")

    return jsonify(
      {
        "success": True,
        "plan_summary": summary,
        "queued_count": queued_count,
        "message": f"Queued {queued_count} generation step(s) for {summary['total_quadrants']} quadrant(s)",
      }
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"success": False, "error": str(e)}), 500
  finally:
    conn.close()


def main():
  global GENERATION_DIR, WEB_SERVER_PORT, APP_CONFIG, BOUNDARY_GEOJSON

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
    help="Port to run the Flask server on (default: 8080)",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
  )
  parser.add_argument(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    help=f"Port for the Vite web server used for rendering (default: {DEFAULT_WEB_PORT})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Path to app_config.json (default: looks in the e2e_generation directory)",
  )
  parser.add_argument(
    "--bounds",
    type=Path,
    default=None,
    help="Path to custom bounds GeoJSON file (default: NYC boundary)",
  )

  args = parser.parse_args()

  GENERATION_DIR = args.generation_dir.resolve()
  WEB_SERVER_PORT = args.web_port

  if not GENERATION_DIR.exists():
    print(f"❌ Error: Directory not found: {GENERATION_DIR}")
    return 1

  db_path = GENERATION_DIR / "quadrants.db"
  if not db_path.exists():
    print(f"❌ Error: Database not found: {db_path}")
    return 1

  # Load app configuration
  APP_CONFIG = load_app_config(args.config)
  print(f"📦 Loaded {len(APP_CONFIG.models)} model(s) from configuration")
  for model in APP_CONFIG.models:
    default_marker = (
      " (default)" if model.model_id == APP_CONFIG.default_model_id else ""
    )
    has_key = "✓" if model.api_key else "✗"
    print(f"   {has_key} {model.name} ({model.model_id}){default_marker}")

  # Load boundary GeoJSON
  bounds_path = args.bounds.resolve() if args.bounds else None
  BOUNDARY_GEOJSON = load_bounds(bounds_path)
  bounds_name = bounds_path.name if bounds_path else "NYC (default)"
  print(f"📍 Boundary: {bounds_name}")

  # Initialize the generation queue table
  conn = get_db_connection()
  try:
    init_queue_table(conn)
    # Reset any items that were mid-processing when server shut down
    # These will be retried automatically
    reset_count = reset_all_processing_items(conn)
    if reset_count > 0:
      print(f"🔄 Reset {reset_count} interrupted generation(s) - will be retried")
    # Clean up old completed items
    deleted_count = clear_completed_items(conn)
    if deleted_count > 0:
      print(f"🧹 Cleaned up {deleted_count} old completed queue item(s)")
  finally:
    conn.close()

  # Start the queue worker
  start_queue_worker()

  # Start the global web renderer
  try:
    start_global_renderer(port=WEB_SERVER_PORT)
  except Exception as e:
    print(f"⚠️  Failed to start web renderer: {e}")
    print("   Rendering will start on demand")

  print("🎨 Starting tile viewer...")
  print(f"   Generation dir: {GENERATION_DIR}")
  print(f"   Flask server: http://{args.host}:{args.port}/")
  print(f"   Web render port: {WEB_SERVER_PORT}")
  print("   Press Ctrl+C to stop")

  try:
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
  finally:
    # Clean up queue worker
    print("\n🛑 Stopping queue worker...")
    stop_queue_worker()

    # Clean up web renderer
    print("🛑 Stopping web renderer...")
    stop_global_renderer()

  return 0


if __name__ == "__main__":
  exit(main())

```

I've successfully implemented all the requirements from `tasks/035_boundaries.md`. Here's a summary of what was created:

## Summary of Changes

### 1. Created `e2e_generation/bounds/` directory with:

- **`nyc.json`** - Extracted NYC GeoJSON boundary as the default boundary file
- **`__init__.py`** - Utilities for loading, saving, and managing bounds files:
  - `load_bounds(path)` - Load a boundary from file (defaults to NYC)
  - `save_bounds(bounds, name)` - Save a boundary to the bounds directory
  - `get_bounds_dir()` - Get the bounds directory path
  - `list_bounds_files()` - List all available bounds files

### 2. Updated `debug_map.py`:

- Now imports boundary from the bounds module instead of inline JSON
- Added `--bounds` / `-b` CLI argument to specify alternate bounds file
- Updated `create_debug_map()` to accept `bounds_path` parameter

### 3. Updated `app.py`:

- Now imports boundary from the bounds module
- Added `--bounds` CLI argument to specify alternate bounds file
- Boundary is loaded at startup and used by the `/api/nyc-boundary` endpoint

### 4. Created `create_bounds.py` - Interactive polygon editor app:

**Features:**
- Leaflet-based map with dark theme
- Shows generated and pending tiles on the map
- Displays NYC boundary by default
- Draggable vertices with visual markers
- Double-click on edge to add a new vertex
- Double-click on vertex to delete it (minimum 3 vertices enforced)
- Self-intersection validation with warnings
- "Reset" button to create a 50% screen rectangle
- "Clear" button to remove all vertices
- Save button with name input → saves to `e2e_generation/bounds/[name].json`
- Visibility toggles for tiles, pending, and NYC boundary

**Usage:**
```bash
# Create new bounds
uv run python src/isometric_nyc/e2e_generation/create_bounds.py generations/v01

# Edit existing bounds
uv run python src/isometric_nyc/e2e_generation/create_bounds.py generations/v01 --load bounds/my-bounds.json
```

```shellscript
# Create new bounds
uv run python src/isometric_nyc/e2e_generation/create_bounds.py generations/v01

# Edit existing bounds
uv run python src/isometric_nyc/e2e_generation/create_bounds.py generations/v01 --load bounds/my-bounds.json
```

---

**User**

Ensure that double-clicking doesn't zoom the map

---

**Cursor**

```python
"""
Create Bounds - Interactive polygon editor for defining generation boundaries.

This script creates an interactive web-based polygon editor that allows you to:
- Draw and edit polygon boundaries on a map
- View existing generated and pending tiles
- Load and edit existing boundary files
- Save new boundaries to the bounds directory

Usage:
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir]
  uv run python src/isometric_nyc/e2e_generation/create_bounds.py [generation_dir] --load bounds/custom.json

Arguments:
  generation_dir: Path to the generation directory (default: generations/v01)
  --load: Path to an existing bounds JSON file to edit

Output:
  Opens a web browser with the polygon editor
  Saves new bounds to e2e_generation/bounds/ directory
"""

import argparse
import json
import sqlite3
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from isometric_nyc.e2e_generation.bounds import get_bounds_dir, load_bounds, save_bounds
from isometric_nyc.e2e_generation.debug_map import (
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
    default=Path("generations/v01"),
    help="Path to the generation directory (default: generations/v01)",
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

```

Added `doubleClickZoom: false` to the Leaflet map options. Now double-clicking will only trigger vertex add/delete actions without zooming the map.

