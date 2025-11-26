import psycopg2
from tabulate import tabulate

from isometric_nyc.db import get_db_config


def get_columns(conn, table_name):
  with conn.cursor() as cur:
    # SQL to get column names from information_schema
    cur.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'citydb' 
            AND table_name = '{table_name}'
            ORDER BY column_name;
        """)
    return cur.fetchall()


def inspect():
  try:
    conn = psycopg2.connect(**get_db_config())
    print("✅ Connected. Analyzing Schema...\n")

    # 1. Inspect 'feature' table (The likely home of the link)
    print("--- Table: citydb.feature ---")
    cols = get_columns(conn, "feature")
    if cols:
      print(tabulate(cols, headers=["Column", "Type"]))
    else:
      print("❌ Table not found!")

    print("\n" + "=" * 30 + "\n")

    # 2. Inspect 'geometry_data' table (The geometry storage)
    print("--- Table: citydb.geometry_data ---")
    cols = get_columns(conn, "geometry_data")
    if cols:
      print(tabulate(cols, headers=["Column", "Type"]))
    else:
      print("❌ Table not found!")

    conn.close()

  except Exception as e:
    print(f"Error: {e}")


def main():
  inspect()


if __name__ == "__main__":
  main()
