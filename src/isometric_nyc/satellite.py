import io
import os
import sys

import PIL
import requests

from isometric_nyc.data.google_maps import GoogleMapsClient


def get_satellite_image(lat, lon, zoom=19, size="1280x1280"):
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    print("Error: GOOGLE_MAPS_API_KEY not found in environment variables.")
    sys.exit(1)

  gmaps = GoogleMapsClient(api_key)
  sat_url = gmaps.get_satellite_image_url(lat, lon, zoom=zoom, size=size)

  # Download the actual image
  response = requests.get(sat_url)
  image = PIL.Image.open(io.BytesIO(response.content))
  return image
