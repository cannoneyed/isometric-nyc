#!/usr/bin/env python3
"""
Script to verify and fix lat/lng coordinates in view.json using Google Maps Geocoding API.

Usage:
    uv run python src/isometric_nyc/synthetic_data/verify_coordinates.py [--dry-run]

Requires GOOGLE_MAPS_API_KEY environment variable to be set.
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


def geocode_location(name: str, api_key: str) -> tuple[float, float] | None:
  """
  Use Google Maps Geocoding API to get lat/lng for a location name.
  Adds "NYC" or "New York" context to improve accuracy.
  """
  # Add NYC context to the query for better results
  query = f"{name}, New York City, NY"

  url = "https://maps.googleapis.com/maps/api/geocode/json"
  params = {
    "address": query,
    "key": api_key,
  }

  try:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data["status"] == "OK" and data["results"]:
      location = data["results"][0]["geometry"]["location"]
      return location["lat"], location["lng"]
    else:
      print(
        f"  Warning: Could not geocode '{name}': {data.get('status', 'Unknown error')}"
      )
      return None
  except requests.RequestException as e:
    print(f"  Error geocoding '{name}': {e}")
    return None


def calculate_distance_meters(
  lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
  """Calculate approximate distance between two points in meters using Haversine formula."""
  import math

  R = 6371000  # Earth's radius in meters

  phi1 = math.radians(lat1)
  phi2 = math.radians(lat2)
  delta_phi = math.radians(lat2 - lat1)
  delta_lambda = math.radians(lon2 - lon1)

  a = (
    math.sin(delta_phi / 2) ** 2
    + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
  )
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

  return R * c


def main() -> None:
  """Main function to verify and fix coordinates."""
  dry_run = "--dry-run" in sys.argv

  # Get API key
  api_key = os.getenv("GOOGLE_MAPS_API_KEY")
  if not api_key:
    print("Error: GOOGLE_MAPS_API_KEY environment variable not set.")
    sys.exit(1)

  # Load view.json
  script_dir = Path(__file__).parent
  view_json_path = script_dir / "view.json"

  with open(view_json_path) as f:
    data = json.load(f)

  locations = data.get("locations", [])

  print(f"Verifying {len(locations)} locations...")
  print("=" * 80)

  updates_needed = []

  for i, loc in enumerate(locations):
    name = loc["name"]
    current_lat = loc["lat"]
    current_lon = loc["lon"]

    print(f"\n[{i + 1}/{len(locations)}] {name}")
    print(f"  Current: ({current_lat}, {current_lon})")

    result = geocode_location(name, api_key)

    if result:
      new_lat, new_lon = result
      distance = calculate_distance_meters(current_lat, current_lon, new_lat, new_lon)

      print(f"  Google:  ({new_lat}, {new_lon})")
      print(f"  Distance: {distance:.0f} meters")

      # Flag if distance is more than 100 meters
      if distance > 100:
        print("  ⚠️  MISMATCH - updating coordinates")
        updates_needed.append(
          {
            "index": i,
            "name": name,
            "old_lat": current_lat,
            "old_lon": current_lon,
            "new_lat": new_lat,
            "new_lon": new_lon,
            "distance": distance,
          }
        )
      else:
        print("  ✓ OK")

  print("\n" + "=" * 80)
  print(f"\nSummary: {len(updates_needed)} locations need updates")

  if updates_needed:
    print("\nLocations to update:")
    for update in updates_needed:
      print(
        f"  - {update['name']}: ({update['old_lat']}, {update['old_lon']}) -> ({update['new_lat']}, {update['new_lon']}) [{update['distance']:.0f}m off]"
      )

    if dry_run:
      print("\n[DRY RUN] No changes made. Remove --dry-run to apply changes.")
    else:
      # Apply updates
      for update in updates_needed:
        locations[update["index"]]["lat"] = update["new_lat"]
        locations[update["index"]]["lon"] = update["new_lon"]

      # Write back to file
      with open(view_json_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")  # Add trailing newline

      print(f"\n✓ Updated {len(updates_needed)} locations in {view_json_path}")
  else:
    print("\nAll coordinates are correct!")


if __name__ == "__main__":
  main()
