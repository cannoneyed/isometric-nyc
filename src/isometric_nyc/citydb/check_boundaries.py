import psycopg2

from isometric_nyc.city_db.db import get_db_config


def check_bounds():
  conn = psycopg2.connect(**get_db_config())
  with conn.cursor() as cur:
    print("🌍 Calculating Database Extents (this might take a moment)...")
    # ST_Extent returns the bounding box of the entire geometry column
    cur.execute("SELECT ST_Extent(geometry) FROM citydb.geometry_data;")
    row = cur.fetchone()

    if row and row[0]:
      print(f"✅ Full Map Bounds: {row[0]}")

      # Let's also count the buildings to ensure we aren't looking at an empty map
      print("📊 Counting Buildings...")
      cur.execute("SELECT count(*) FROM citydb.feature WHERE objectclass_id = 26;")
      count = cur.fetchone()[0]
      print(f"   Total Buildings: {count}")

      if count < 100:
        print(
          "⚠️  WARNING: Very few buildings found. Import might have failed or only imported metadata."
        )
    else:
      print("❌ Map is empty. No geometry found.")


def main():
  check_bounds()


if __name__ == "__main__":
  main()
