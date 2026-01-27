import os

from dotenv import load_dotenv


def get_db_config():
  load_dotenv()
  return {
    "host": "localhost",
    "port": os.getenv("CITYDB_PORT"),
    "database": os.getenv("CITYDB_NAME"),
    "user": os.getenv("CITYDB_USER"),
    "password": os.getenv("CITYDB_PASSWORD"),
  }
