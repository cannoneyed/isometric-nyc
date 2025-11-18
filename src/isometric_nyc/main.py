import argparse
import os
import sys

from dotenv import load_dotenv

from isometric_nyc.data.database import init_db, save_building
from isometric_nyc.data.google_maps import GoogleMapsClient
from isometric_nyc.data.nyc_opendata import NYCOpenDataClient
from isometric_nyc.models.building import BuildingData


def main():
  load_dotenv()

  parser = argparse.ArgumentParser(description="Generate isometric NYC building data.")
  parser.add_argument("address", help="The NYC address to process")
  args = parser.parse_args()

  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    print("Error: GOOGLE_MAPS_API_KEY not found in environment variables.")
    sys.exit(1)

  app_token = os.getenv("NYC_OPENDATA_APP_TOKEN")
  if not app_token:
    print("Error: NYC_OPENDATA_APP_TOKEN not found in environment variables.")
    sys.exit(1)

  print(f"Processing address: {args.address}")

  # Initialize clients
  gmaps = GoogleMapsClient(api_key)
  nyc = NYCOpenDataClient(app_token)
  init_db()

  # 1. Geocode
  print("Geocoding...")
  coords = gmaps.geocode(args.address)
  if not coords:
    print("Error: Could not geocode address.")
    sys.exit(1)
  lat, lng = coords
  print(f"Coordinates: {lat}, {lng}")

  # 2. OpenData
  print("Fetching building footprint...")
  footprint_data = nyc.get_building_footprint(lat, lng)
  if not footprint_data:
    print("Warning: No building footprint found at this location.")
  else:
    print(f"Found building BIN: {footprint_data.get('bin')}")

  # 3. Google Maps Images
  print("Generating image URLs...")
  sat_url = gmaps.get_satellite_image_url(lat, lng)
  sv_url = gmaps.get_street_view_image_url(lat, lng)

  # 4. Save
  building = BuildingData(
    address=args.address,
    bin=footprint_data.get("bin") if footprint_data else None,
    footprint_geometry=footprint_data.get("the_geom") if footprint_data else None,
    roof_height=float(footprint_data.get("heightroof", 0))
    if footprint_data and "heightroof" in footprint_data
    else None,
    satellite_image_url=sat_url,
    street_view_image_url=sv_url,
    raw_metadata=footprint_data,
  )

  save_building(building)
  print("Saved building data to database.")


if __name__ == "__main__":
  main()
