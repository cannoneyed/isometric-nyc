import psycopg2
from tabulate import tabulate

from isometric_nyc.db import get_db_config


def db_census(conn):
  with conn.cursor() as cur:
    print("📊 Running Database Census...")
    # Count all features grouped by their Class ID
    sql = """
        SELECT objectclass_id, count(*) 
        FROM citydb.feature 
        GROUP BY objectclass_id 
        ORDER BY count(*) DESC;
        """
    cur.execute(sql)
    rows = cur.fetchall()

    if rows:
      print(tabulate(rows, headers=["Class ID", "Count"]))
      print("\nCHEAT SHEET:")
      print("26 = Building")
      print("43-46 = Roads/Transportation")
      print("8-9 = Water/Relief")
      print("21 = CityFurniture")
      print("3 = LandUse")
    else:
      print("❌ The 'feature' table is COMPLETELY EMPTY.")
      print(
        "   If 'geometry_data' has rows but 'feature' does not, your database is corrupted or orphaned."
      )


def identify_classes(conn):
  conn = psycopg2.connect(**get_db_config())
  with conn.cursor() as cur:
    # Check specific IDs found in your census
    ids_to_check = (709, 712, 710, 901)

    sql = f"""
        SELECT id, classname
        FROM citydb.objectclass
        WHERE id IN {ids_to_check};
        """

    try:
      cur.execute(sql)
      rows = cur.fetchall()
      print(tabulate(rows, headers=["ID", "Class Name"]))
    except Exception as e:
      print(f"Error: {e}")
      # Fallback: if 'alias' column doesn't exist in your version
      cur.execute(
        f"SELECT id, classname FROM citydb.objectclass WHERE id IN {ids_to_check}"
      )
      print(tabulate(cur.fetchall(), headers=["ID", "Class Name"]))


def main():
  conn = psycopg2.connect(**get_db_config())
  db_census(conn)
  identify_classes(conn)


if __name__ == "__main__":
  main()
