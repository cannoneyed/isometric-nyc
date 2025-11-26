import psycopg2
from tabulate import tabulate

from isometric_nyc.db import get_db_config


def inspect_db():
  try:
    config = get_db_config()
    conn = psycopg2.connect(**config)
    cur = conn.cursor()

    print(f"✅ Connected to database: {config['database']}")

    # 1. Check Search Path
    cur.execute("SHOW search_path;")
    print(f"🔎 Current Search Path: {cur.fetchone()[0]}")

    # 2. List All Schemas
    print("\n📂 AVAILABLE SCHEMAS:")
    cur.execute(
      "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast');"
    )
    schemas = [row[0] for row in cur.fetchall()]
    print(schemas)

    # 3. List All Tables (and their Schema)
    print("\n🗄️ FOUND TABLES (First 50):")
    cur.execute("""
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            LIMIT 50;
        """)

    rows = cur.fetchall()
    if rows:
      print(tabulate(rows, headers=["Schema", "Table"]))

      # Check specifically for our missing friend
      found_sg = any(r[1] == "surface_geometry" for r in rows)
      if found_sg:
        schema_loc = next(r[0] for r in rows if r[1] == "surface_geometry")
        print(f"\n🎉 FOUND IT! 'surface_geometry' is in schema: '{schema_loc}'")
        print(f"👉 Update your query to use: {schema_loc}.surface_geometry")
      else:
        print("\n❌ 'surface_geometry' table NOT FOUND in any schema.")
        print(
          "   Likely cause: The 3DCityDB scripts created the DB structure, but the tables failed to initialize."
        )
    else:
      print("❌ No tables found at all. The database might be empty.")

    conn.close()

  except Exception as e:
    print(f"💥 Connection Failed: {e}")


def main():
  inspect_db()


if __name__ == "__main__":
  main()
