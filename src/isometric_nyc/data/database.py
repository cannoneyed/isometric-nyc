import json
import sqlite3
from pathlib import Path
from typing import Optional

from isometric_nyc.models.building import BuildingData

DB_PATH = Path("buildings.db")


def init_db():
  """Initialize the SQLite database with the buildings table."""
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS buildings (
            address TEXT PRIMARY KEY,
            bin TEXT,
            footprint_geometry TEXT,
            roof_height REAL,
            satellite_image_url TEXT,
            street_view_image_url TEXT,
            raw_metadata TEXT
        )
    """)
  conn.commit()
  conn.close()


def save_building(building: BuildingData):
  """Save or update a building record."""
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  footprint_json = (
    json.dumps(building.footprint_geometry) if building.footprint_geometry else None
  )
  metadata_json = json.dumps(building.raw_metadata) if building.raw_metadata else None

  cursor.execute(
    """
        INSERT OR REPLACE INTO buildings (
            address, bin, footprint_geometry, roof_height, 
            satellite_image_url, street_view_image_url, raw_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (
      building.address,
      building.bin,
      footprint_json,
      building.roof_height,
      building.satellite_image_url,
      building.street_view_image_url,
      metadata_json,
    ),
  )
  conn.commit()
  conn.close()


def get_building(address: str) -> Optional[BuildingData]:
  """Retrieve a building by address."""
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  cursor = conn.cursor()

  cursor.execute("SELECT * FROM buildings WHERE address = ?", (address,))
  row = cursor.fetchone()
  conn.close()

  if row:
    return BuildingData(
      address=row["address"],
      bin=row["bin"],
      footprint_geometry=json.loads(row["footprint_geometry"])
      if row["footprint_geometry"]
      else None,
      roof_height=row["roof_height"],
      satellite_image_url=row["satellite_image_url"],
      street_view_image_url=row["street_view_image_url"],
      raw_metadata=json.loads(row["raw_metadata"]) if row["raw_metadata"] else None,
    )
  return None
