# Isometric NYC

This project aims to generate a massive isometric pixel art view of New York City using the latest and greatest AI tools available.

## Setup

In order to run the isometric-nyc command, you will need API credentials for the services in a `.env` file in the root directory of the project.

```bash
GOOGLE_MAPS_API_KEY=...
NYC_OPENDATA_APP_TOKEN=...
```

This repo uses `uv` for dependency management and environment handling. To install dependencies, run:

```bash
uv sync
```

## Scripts

```bash
uv run isometric-nyc "350 5th Ave, New York, NY"
```
