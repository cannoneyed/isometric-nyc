import argparse
import json

from isometric_nyc.data.database import get_building


def main():
  parser = argparse.ArgumentParser(description="Inspect a building in the database.")
  parser.add_argument("address", help="The address to inspect")
  args = parser.parse_args()

  building = get_building(args.address)
  if building:
    # model_dump() is for Pydantic v2
    print(json.dumps(building.model_dump(), indent=2, default=str))
  else:
    print(f"No building found for address: {args.address}")


if __name__ == "__main__":
  main()
