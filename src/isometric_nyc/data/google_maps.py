from typing import Optional, Tuple

import googlemaps


class GoogleMapsClient:
  def __init__(self, api_key: str):
    self.client = googlemaps.Client(key=api_key)

  def geocode(self, address: str) -> Optional[Tuple[float, float]]:
    """Geocode an address to (lat, lng)."""
    result = self.client.geocode(address)
    if not result:
      return None
    location = result[0]["geometry"]["location"]
    return location["lat"], location["lng"]

  def get_satellite_image_url(
    self, lat: float, lng: float, zoom: int = 19, size: str = "600x600"
  ) -> str:
    """Generate a static map URL for satellite view."""
    # Note: This generates a URL, it doesn't download the image.
    # For signed URLs or downloading, we'd need more logic.
    # For now, we construct the URL manually or use the client if it supports it (it mostly does signing).
    # The python client 'static_map' method returns the raw image data (generator).
    # But for the DB we might want the URL or the blob. The plan said "URL".
    # Let's construct the URL for now as it's easier to store/view.
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    return f"{base_url}?center={lat},{lng}&zoom={zoom}&size={size}&maptype=satellite&key={self.client.key}"

  def get_street_view_image_url(
    self, lat: float, lng: float, size: str = "600x600"
  ) -> str:
    """Generate a Street View static API URL."""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    return f"{base_url}?location={lat},{lng}&size={size}&key={self.client.key}"
