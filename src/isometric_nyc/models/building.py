from typing import Any, Dict, Optional

from pydantic import BaseModel


class BuildingData(BaseModel):
  address: str
  bin: Optional[str] = None
  footprint_geometry: Optional[Dict[str, Any]] = None  # GeoJSON or similar
  roof_height: Optional[float] = None
  satellite_image_url: Optional[str] = None
  street_view_image_url: Optional[str] = None
  raw_metadata: Optional[Dict[str, Any]] = None
