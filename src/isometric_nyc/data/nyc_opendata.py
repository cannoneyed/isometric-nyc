from typing import Any, Dict, Optional

from sodapy import Socrata

# NYC Building Footprints dataset ID
DATASET_ID = "5zhs-2jue"


class NYCOpenDataClient:
  def __init__(self, app_token: Optional[str] = None):
    self.client = Socrata("data.cityofnewyork.us", app_token)

  def get_building_footprint(self, lat: float, lng: float) -> Optional[Dict[str, Any]]:
    """
    Find the building footprint that contains the given point.
    Returns the first match found.
    """
    # SoQL query to find polygon containing the point
    # We use 'intersects(the_geom, 'POINT(lng lat)')'
    # Note: WKT is POINT(x y) -> POINT(lng lat)
    query = f"intersects(the_geom, 'POINT({lng} {lat})')"

    results = self.client.get(DATASET_ID, where=query, limit=1)

    if results:
      return results[0]
    return None
