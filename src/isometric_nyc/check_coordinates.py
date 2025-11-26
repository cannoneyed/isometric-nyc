import psycopg2
from shapely.wkb import loads as load_wkb

from isometric_nyc.db import get_db_config


def check_raw_coords():
  conn = psycopg2.connect(**get_db_config())
  with conn.cursor() as cur:
    # Get one raw geometry from the table
    sql = "SELECT ST_AsBinary(geometry) FROM citydb.geometry_data LIMIT 1;"
    cur.execute(sql)
    row = cur.fetchone()

    if row:
      geom = load_wkb(bytes(row[0]))
      # Get the first point of the geometry
      if geom.geom_type == "Polygon":
        x, y = geom.exterior.coords[0][0], geom.exterior.coords[0][1]
      elif geom.geom_type == "MultiPolygon":
        x, y = geom.geoms[0].exterior.coords[0][0], geom.geoms[0].exterior.coords[0][1]
      elif geom.geom_type == "MultiSurface":
        # Handle MultiSurface (common in CityGML)
        x, y = geom.geoms[0].exterior.coords[0][0], geom.geoms[0].exterior.coords[0][1]
      else:
        x, y = 0, 0

      print(f"\n📢 RAW COORDINATE SAMPLE: X={x:.2f}, Y={y:.2f}")

      # --- DIAGNOSIS ---
      if 580000 < x < 600000:
        print("✅ DIAGNOSIS: Data is in UTM 18N (Meters).")
        print("👉 ACTION: Set FORCE_SRID = 32618 in your script.")
      elif 900000 < x < 1050000:
        print("✅ DIAGNOSIS: Data is in NY State Plane (Feet).")
        print("👉 ACTION: Set FORCE_SRID = 2263 in your script.")
      else:
        print("⚠️  Unknown Projection. Use these SRID values to find the match.")
    else:
      print("❌ Table is empty.")


def main():
  check_raw_coords()


if __name__ == "__main__":
  main()
