## Simplest Generation

We will implement a system to generate isometric NYC buildings by first gathering the necessary data.

### Plan

1.  **Data Gathering Function**:
    *   Input: Address string.
    *   **Geocoding**: Convert address to coordinates (lat/lng).
    *   **NYC OpenData**:
        *   Dataset: Building Footprints (Socrata ID: `5zhs-2jue`).
        *   Query: Find building footprint intersecting the coordinates.
        *   Fields: `the_geom` (footprint), `heightroof` (roof height).
    *   **Google Maps**:
        *   Satellite Image: Google Static Maps API.
        *   Street View: Google Street View Static API.

2.  **Storage**:
    *   **Database**: SQLite (`buildings.db`).
    *   **Schema**:
        *   `address` (Primary Key)
        *   `bin` (Building Identification Number)
        *   `footprint_geometry` (JSON/WKT)
        *   `roof_height` (Float)
        *   `satellite_image_url` (Text)
        *   `street_view_image_url` (Text)
        *   `raw_metadata` (JSON)

3.  **Implementation Steps**:
    *   Install dependencies: `sodapy`, `googlemaps`, `pydantic`.
    *   Create `src/isometric_nyc/data/` module for API clients and DB access.
    *   Create `src/isometric_nyc/models/` for data structures.
    *   Update `main.py` to orchestrate the process.

### Requirements
*   `GOOGLE_MAPS_API_KEY` environment variable.
*   Optional: `NYC_OPENDATA_APP_TOKEN` for higher rate limits.
