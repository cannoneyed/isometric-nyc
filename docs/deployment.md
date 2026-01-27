# Deployment

The web app can be deployed to GitHub Pages with DZI tile assets served from Cloudflare R2.

### Prerequisites

1. **Install rclone** (required for uploading DZI tiles):

```bash
brew install rclone
```

2. **Configure rclone for R2**:

```bash
rclone config
# Create new remote named 'r2'
# Choose 'Cloudflare R2' as provider
# Enter your R2 access key and secret
# Set endpoint to your R2 endpoint URL
```

3. **Create an R2 bucket** named `isometric-nyc` in your Cloudflare dashboard

### Deploy Script

The deployment script (`src/app/deploy.py`) handles three tasks:

| Step         | Flag             | Description                              |
| ------------ | ---------------- | ---------------------------------------- |
| Build        | `--build`        | Builds the Vite app to `dist/`           |
| GitHub Pages | `--github-pages` | Pushes HTML/JS/CSS to `gh-pages` branch  |
| Cloudflare R2| `--r2`           | Uploads DZI tiles to R2                  |

### Usage

```bash
# Build only
uv run python src/app/deploy.py --build

# Deploy to GitHub Pages only
uv run python src/app/deploy.py --github-pages

# Upload DZI tiles to R2 only
uv run python src/app/deploy.py --r2

# Full deployment (build + GitHub Pages + R2)
uv run python src/app/deploy.py --all

# Dry run to preview what would be deployed
uv run python src/app/deploy.py --all --dry-run
```

### Configuration

| Environment Variable    | Description                          |
| ----------------------- | ------------------------------------ |
| `GITHUB_PAGES_DOMAIN`   | Custom domain for GitHub Pages (opt) |

### Architecture

- **GitHub Pages**: Serves the static HTML/JS/CSS (lightweight, fast)
- **Cloudflare R2**: Serves DZI tiles (many small WebP files, CDN-optimized)

The web app automatically loads tiles from the configured R2 public URL.

### What Gets Uploaded to R2

The `--r2` flag uploads:

- `tiles.dzi` - DZI descriptor file
- `tiles_metadata.json` - Custom metadata (grid dimensions, origin, etc.)
- `tiles_files/` - Tile pyramid (~60,000+ WebP files)

Files are uploaded to `isometric-nyc/` prefix in the R2 bucket.

### Caching

Caching is handled by a worker that adds appropriate cache headers and entries in `src/app/worker/src/index.ts`. Note that this worker must be configured properly in the CloudFlare dashboard.
