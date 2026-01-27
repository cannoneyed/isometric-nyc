# Generations

This directory contains the generations data for generated tile quadrants. Each project's generations are stored in its own subdir (e.g. `nyc`), with the actual data stored in a SQLite DB with the following schema:

## Tables

### `quadrants`

Primary table storing tile quadrant data and generated images.

| Column | Type | Description |
|--------|------|-------------|
| `quadrant_x` | INTEGER | X coordinate of the quadrant |
| `quadrant_y` | INTEGER | Y coordinate of the quadrant |
| `lat` | REAL | Latitude of the quadrant center |
| `lng` | REAL | Longitude of the quadrant center |
| `render` | BLOB | Original rendered image (PNG) |
| `generation` | BLOB | AI-generated isometric image (PNG) |
| `is_generated` | INTEGER | Computed: 1 if generation exists, else 0 |
| `notes` | TEXT | User notes for the quadrant |
| `flagged` | INTEGER | 1 if flagged for review (default: 0) |
| `is_water` | INTEGER | 1 if quadrant is water tile (default: 0) |
| `starred` | INTEGER | 1 if starred/favorited (default: 0) |
| `is_reference` | INTEGER | 1 if used as reference tile (default: 0) |
| `water_mask` | BLOB | Water mask image (PNG) (WIP) |
| `water_type` | TEXT | Type of water body (unused / WIP) |
| `dark_mode` | BLOB | Dark mode variant image (PNG) (WIP / unused) |

**Indexes:** `idx_quadrants_coords` (lat, lng), `idx_quadrants_tile` (tile_row, tile_col), `idx_quadrants_generated` (is_generated)

### `metadata`

Key-value store for database metadata.

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT | Metadata key (PK) |
| `value` | TEXT | Metadata value |

### `generation_queue`

Queue for generation jobs driven by the e2e generation web app.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing ID (PK) |
| `item_type` | TEXT | Type of generation item |
| `quadrants` | TEXT | JSON list of quadrant coordinates |
| `context_quadrants` | TEXT | JSON list of context quadrant coordinates |
| `model_id` | TEXT | AI model identifier |
| `prompt` | TEXT | Generation prompt |
| `negative_prompt` | TEXT | Negative prompt for generation |
| `status` | TEXT | Job status: pending, running, completed, failed |
| `created_at` | REAL | Unix timestamp of creation |
| `started_at` | REAL | Unix timestamp when started |
| `completed_at` | REAL | Unix timestamp when completed |
| `error_message` | TEXT | Error message if failed |
| `result_message` | TEXT | Result/success message |

**Indexes:** `idx_queue_status` (status)
